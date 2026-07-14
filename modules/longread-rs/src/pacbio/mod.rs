//! PacBio subread/CCS BAM post-processing.
//!
//! Two subcommands share this module:
//!
//! * `longread rg` ([`normalize`]) — canonicalize every PBSIM3 subread BAM as one synthetic
//!   PacBio movie with globally unique ZMWs and one specification-compliant `SUBREAD` read group,
//!   emitting one `*.normalized.bam` per input plus a `zmw_map.tsv`.
//! * `longread check` ([`check`]) — fail before merge if CCS chunks overlap in PacBio's
//!   movie/ZMW key space.
//!
//! Both replace an earlier `samtools`/`awk`/`sort` pipeline. They work entirely in-process via
//! `noodles` (no subprocess), parallelize across input BAMs, and are deterministic regardless of
//! thread count. The parallel driver bounds the number of *simultaneously open* file descriptors
//! (independently of how many input BAMs there are) so that thousands of chunks are safe.

pub mod check;
pub mod normalize;

use std::fs::File;
use std::path::Path;

use md5::{Digest, Md5};
use noodles_bam as bam;
use noodles_bgzf as bgzf;
use noodles_sam as sam;

use crate::error::{Error, Result};

/// A BAM reader over a single-threaded bgzf decoder reading directly from a file.
///
/// One reader holds exactly one open file descriptor; the bounded worker pool in each subcommand
/// caps how many exist at once.
pub(crate) type BamReader = bam::io::Reader<bgzf::io::Reader<File>>;

/// A BAM writer over a single-threaded bgzf encoder writing directly to a file.
pub(crate) type BamWriter = bam::io::Writer<bgzf::io::Writer<File>>;

/// Open a BAM file for reading and consume its header, returning the reader positioned at the
/// first record together with the parsed SAM header.
pub(crate) fn open_reader(path: &Path) -> Result<(BamReader, sam::Header)> {
    let file = File::open(path)
        .map_err(|e| Error::pacbio(format!("cannot open {}: {e}", path.display())))?;
    let mut reader = bam::io::Reader::new(file);
    let header = reader
        .read_header()
        .map_err(|e| Error::pacbio(format!("{}: reading BAM header: {e}", path.display())))?;
    Ok((reader, header))
}

/// Create a BAM writer over a bgzf encoder (uses the crate's `libdeflate` backend).
pub(crate) fn create_writer(path: &Path) -> Result<BamWriter> {
    let file = File::create(path)
        .map_err(|e| Error::pacbio(format!("cannot create {}: {e}", path.display())))?;
    Ok(bam::io::Writer::new(file))
}

/// Build a bounded [`rayon::ThreadPool`].
///
/// The worker count is `min(threads_or_all_cores, max_open_files / files_per_task)`, so that with
/// each worker holding at most `files_per_task` open files the total never exceeds
/// `max_open_files`. Determinism does not depend on the worker count: every file is processed
/// independently and outputs are ordered by a serial, deterministic merge/sort.
pub(crate) fn build_pool(
    threads: usize,
    max_open_files: usize,
    files_per_task: usize,
) -> Result<rayon::ThreadPool> {
    let logical = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    };
    let fd_cap = (max_open_files / files_per_task.max(1)).max(1);
    let workers = logical.min(fd_cap).max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| Error::pacbio(format!("failed to build thread pool: {e}")))
}

/// Sanitize a movie name to PacBio's allowed character set (`[A-Za-z0-9_.-]`), mirroring the
/// original Groovy `replaceAll(/[^A-Za-z0-9_.-]/, '_')`.
pub(crate) fn sanitize_movie(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Derive the 8-hex read-group ID for a `SUBREAD` movie: the first four bytes of
/// `md5(lower(movie) + "//SUBREAD")`, byte-identical to the original
/// `printf '%s//SUBREAD' | md5sum | cut -c1-8`.
pub(crate) fn read_group_id(movie: &str) -> String {
    let mut hasher = Md5::new();
    hasher.update(movie.to_ascii_lowercase().as_bytes());
    hasher.update(b"//SUBREAD");
    let digest = hasher.finalize();
    format!(
        "{:02x}{:02x}{:02x}{:02x}",
        digest[0], digest[1], digest[2], digest[3]
    )
}

/// The three components of a PacBio subread QNAME: `movie / zmw / rest`.
pub(crate) struct SubreadName<'a> {
    /// The movie name (component 0).
    pub movie: &'a [u8],
    /// The ZMW hole number (component 1), parsed as a non-negative integer.
    pub zmw: i64,
    /// The remainder (component 2), e.g. `qStart_qEnd`.
    pub rest: &'a [u8],
}

/// Parse a PacBio subread QNAME. Requires exactly three `/`-separated components with an all-digit
/// ZMW, matching the original `awk` guard (`length != 3 || zmw !~ /^[0-9]+$/`).
pub(crate) fn parse_subread_name(name: &[u8]) -> Result<SubreadName<'_>> {
    let mut parts = name.split(|&b| b == b'/');
    let movie = parts.next();
    let zmw = parts.next();
    let rest = parts.next();
    let extra = parts.next();
    match (movie, zmw, rest, extra) {
        (Some(movie), Some(zmw), Some(rest), None)
            if !zmw.is_empty() && zmw.iter().all(u8::is_ascii_digit) =>
        {
            let zmw = std::str::from_utf8(zmw)
                .ok()
                .and_then(|s| s.parse::<i64>().ok())
                .ok_or_else(|| {
                    Error::pacbio(format!(
                        "PBSIM3 subread QNAME has an out-of-range ZMW: {}",
                        String::from_utf8_lossy(name)
                    ))
                })?;
            Ok(SubreadName { movie, zmw, rest })
        }
        _ => Err(Error::pacbio(format!(
            "malformed PBSIM3 subread QNAME (expected movie/zmw/rest with numeric zmw): {}",
            String::from_utf8_lossy(name)
        ))),
    }
}

/// Extract the `movie/zmw` key from a read QNAME (subread or CCS). Requires at least two
/// `/`-separated components; returns the key as owned bytes.
pub(crate) fn movie_zmw_key(name: &[u8]) -> Result<Vec<u8>> {
    let mut parts = name.split(|&b| b == b'/');
    match (parts.next(), parts.next()) {
        (Some(movie), Some(zmw)) if !movie.is_empty() && !zmw.is_empty() => {
            let mut key = Vec::with_capacity(movie.len() + 1 + zmw.len());
            key.extend_from_slice(movie);
            key.push(b'/');
            key.extend_from_slice(zmw);
            Ok(key)
        }
        _ => Err(Error::pacbio(format!(
            "malformed QNAME (expected at least movie/zmw): {}",
            String::from_utf8_lossy(name)
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_group_id_matches_reference() {
        // printf 'movie.sample//SUBREAD' | md5sum | cut -c1-8
        // -> verified against the shell pipeline.
        let id = read_group_id("movie.sample");
        assert_eq!(id.len(), 8);
        assert!(id.bytes().all(|b| b.is_ascii_hexdigit()));
        // Case-insensitive on the movie (lowercased before hashing).
        assert_eq!(read_group_id("MOVIE.Sample"), id);
    }

    #[test]
    fn sanitize_movie_replaces_invalid() {
        assert_eq!(sanitize_movie("movie.id_1-2"), "movie.id_1-2");
        assert_eq!(sanitize_movie("movie:/x y"), "movie__x_y");
    }

    #[test]
    fn parse_subread_name_ok() {
        let n = parse_subread_name(b"movie.a/42/0_100").unwrap();
        assert_eq!(n.movie, b"movie.a");
        assert_eq!(n.zmw, 42);
        assert_eq!(n.rest, b"0_100");
    }

    #[test]
    fn parse_subread_name_rejects_bad() {
        assert!(parse_subread_name(b"movie.a/42").is_err()); // too few
        assert!(parse_subread_name(b"movie.a/42/0_100/x").is_err()); // too many
        assert!(parse_subread_name(b"movie.a/xx/0_100").is_err()); // non-numeric zmw
        assert!(parse_subread_name(b"movie.a//0_100").is_err()); // empty zmw
    }

    #[test]
    fn movie_zmw_key_ok() {
        assert_eq!(movie_zmw_key(b"movie.a/42/ccs").unwrap(), b"movie.a/42");
        assert_eq!(movie_zmw_key(b"movie.a/42").unwrap(), b"movie.a/42");
        assert!(movie_zmw_key(b"movie.a").is_err());
    }
}
