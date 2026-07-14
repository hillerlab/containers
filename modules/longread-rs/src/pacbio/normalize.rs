//! `longread rg` — canonicalize PBSIM3 subread BAMs as one synthetic PacBio movie.
//!
//! CCS chunking is defined for a single movie, but PBSIM3 restarts ZMW numbering for every
//! simulated movie and emits the placeholder read group `ID:ffffffff`. This subcommand:
//!
//! 1. assigns every original `(movie, zmw)` pair a deterministic global ZMW;
//! 2. rewrites each record's QNAME and `zm` tag to that global ZMW;
//! 3. rewrites all records to one specification-compliant `SUBREAD` read group;
//! 4. emits a `zmw_map.tsv` preserving the original identity.
//!
//! It replaces a `samtools view | awk | sort` pipeline that decompressed every BAM ~5 times.
//! Here the work is exactly two parallel passes (scan, then rewrite) with a serial, deterministic
//! allocation step in between, so output is byte-identical regardless of thread count.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};

use noodles_bam as bam;
use noodles_sam as sam;
use rayon::prelude::*;
use sam::alignment::io::Write as _;
use sam::alignment::record::data::field::Tag;
use sam::alignment::record_buf::data::field::Value;
use sam::alignment::RecordBuf;
use sam::header::record::value::map::read_group;

use crate::error::{Error, Result};
use crate::pacbio::{
    build_pool, create_writer, open_reader, parse_subread_name, read_group_id, sanitize_movie,
};

/// PacBio `zm` (ZMW hole number) tag.
const ZM_TAG: Tag = Tag::new(b'z', b'm');

/// Parameters for `longread rg`.
#[derive(Debug, Clone)]
pub struct NormalizeParams {
    /// Input subread BAMs (each a distinct PBSIM3 movie).
    pub bams: Vec<PathBuf>,
    /// Raw synthetic movie name (e.g. `movie.<id>`); sanitized internally.
    pub movie: String,
    /// Directory for the `*.normalized.bam` outputs.
    pub outdir: PathBuf,
    /// Output path for the ZMW map TSV.
    pub zmw_map: PathBuf,
    /// Worker count (0 = all cores), bounded by `max_open_files`.
    pub threads: usize,
    /// Upper bound on simultaneously open file descriptors.
    pub max_open_files: usize,
}

/// Summary statistics for a normalization run.
#[derive(Debug, Clone)]
pub struct NormalizeStats {
    /// Number of input BAMs.
    pub input_bams: usize,
    /// Number of distinct source movies observed.
    pub movies: usize,
    /// Total records processed.
    pub records: u64,
    /// Allocated global ZMW capacity.
    pub zmw_capacity: i64,
    /// Read-group ID assigned to every record.
    pub rg_id: String,
    /// Paths of the written `*.normalized.bam` files (input order).
    pub outputs: Vec<PathBuf>,
}

/// Per-file result of the scan pass.
#[derive(Default)]
struct FileScan {
    /// Per-movie `(min_zmw, max_zmw)`.
    movie_bounds: HashMap<Vec<u8>, (i64, i64)>,
    /// Observed `(movie, zmw)` pairs (for the ZMW map).
    pairs: HashSet<(Vec<u8>, i64)>,
    /// Every QNAME seen (for global duplicate detection).
    qnames: Vec<Vec<u8>>,
    /// Record count.
    records: u64,
}

/// Deterministic per-movie ZMW allocation: `new_zmw = offset + (zmw - min) + 1`.
struct Allocation {
    /// movie -> `(min_zmw, offset)`.
    table: HashMap<Vec<u8>, (i64, i64)>,
    /// Total global ZMW capacity.
    capacity: i64,
}

impl Allocation {
    fn map_zmw(&self, movie: &[u8], zmw: i64) -> Option<i64> {
        self.table
            .get(movie)
            .map(|(min, offset)| offset + (zmw - min) + 1)
    }
}

/// Result of merging per-file scans and assigning global ZMW offsets.
struct Merged {
    /// Deterministic per-movie ZMW allocation.
    allocation: Allocation,
    /// Observed `(movie, zmw)` pairs, sorted by `(movie, zmw)`.
    pairs: Vec<(Vec<u8>, i64)>,
    /// Number of distinct source movies.
    movies: usize,
    /// Total records scanned.
    records: u64,
}

/// Run `longread rg`.
pub fn run(params: &NormalizeParams) -> Result<NormalizeStats> {
    if params.bams.is_empty() {
        return Err(Error::pacbio("no input BAMs provided"));
    }
    let synthetic_movie = sanitize_movie(&params.movie);
    let rg_id = read_group_id(&synthetic_movie);

    // One pool sized for the heavier pass (rewrite opens a reader + a writer per task), reused for
    // both passes so open-FD peak never exceeds `max_open_files`.
    let pool = build_pool(params.threads, params.max_open_files, 2)?;

    // Pass 1 — parallel scan. Also validates each header's read group up front (fail fast).
    let scans: Vec<FileScan> = pool.install(|| {
        params
            .bams
            .par_iter()
            .map(|path| scan_file(path, &synthetic_movie, &rg_id))
            .collect::<Result<Vec<_>>>()
    })?;

    // Serial, deterministic merge + allocation.
    let merged = merge_and_allocate(scans)?;

    write_zmw_map(
        &params.zmw_map,
        &merged.pairs,
        &merged.allocation,
        &synthetic_movie,
    )?;

    // Pass 2 — parallel rewrite.
    let outputs: Vec<PathBuf> = pool.install(|| {
        params
            .bams
            .par_iter()
            .map(|path| {
                rewrite_file(
                    path,
                    &params.outdir,
                    &synthetic_movie,
                    &rg_id,
                    &merged.allocation,
                )
            })
            .collect::<Result<Vec<_>>>()
    })?;

    Ok(NormalizeStats {
        input_bams: params.bams.len(),
        movies: merged.movies,
        records: merged.records,
        zmw_capacity: merged.allocation.capacity,
        rg_id,
        outputs,
    })
}

/// Scan one BAM: validate its read group, then accumulate per-movie ZMW bounds, observed pairs,
/// and QNAMEs. Only the QNAME of each record is decoded.
fn scan_file(path: &Path, synthetic_movie: &str, rg_id: &str) -> Result<FileScan> {
    let (mut reader, header) = open_reader(path)?;
    // Reuse the header builder purely to validate the read group early.
    normalized_header(&header, synthetic_movie, rg_id, path)?;

    let mut scan = FileScan::default();
    let mut record = bam::Record::default();
    loop {
        let n = reader
            .read_record(&mut record)
            .map_err(|e| Error::pacbio(format!("{}: reading record: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        let name = record
            .name()
            .ok_or_else(|| Error::pacbio(format!("{}: record without QNAME", path.display())))?;
        let bytes: &[u8] = name.as_ref();
        let parsed = parse_subread_name(bytes)?;

        scan.movie_bounds
            .entry(parsed.movie.to_vec())
            .and_modify(|(min, max)| {
                if parsed.zmw < *min {
                    *min = parsed.zmw;
                }
                if parsed.zmw > *max {
                    *max = parsed.zmw;
                }
            })
            .or_insert((parsed.zmw, parsed.zmw));
        scan.pairs.insert((parsed.movie.to_vec(), parsed.zmw));
        scan.qnames.push(bytes.to_vec());
        scan.records += 1;
    }
    Ok(scan)
}

/// Merge per-file scans and assign deterministic global ZMW offsets by sorted movie name.
fn merge_and_allocate(scans: Vec<FileScan>) -> Result<Merged> {
    let mut bounds: HashMap<Vec<u8>, (i64, i64)> = HashMap::new();
    let mut pairs: HashSet<(Vec<u8>, i64)> = HashSet::new();
    let mut seen_qnames: HashSet<Vec<u8>> = HashSet::new();
    let mut records = 0u64;

    for scan in scans {
        for (movie, (min, max)) in scan.movie_bounds {
            bounds
                .entry(movie)
                .and_modify(|(lo, hi)| {
                    if min < *lo {
                        *lo = min;
                    }
                    if max > *hi {
                        *hi = max;
                    }
                })
                .or_insert((min, max));
        }
        pairs.extend(scan.pairs);
        for qname in scan.qnames {
            if seen_qnames.contains(&qname) {
                return Err(Error::pacbio(format!(
                    "duplicate PBSIM3 subread QNAME before merge: {}",
                    String::from_utf8_lossy(&qname)
                )));
            }
            seen_qnames.insert(qname);
        }
        records += scan.records;
    }

    // Assign a non-overlapping ZMW range to each movie in sorted order.
    let mut movies: Vec<Vec<u8>> = bounds.keys().cloned().collect();
    movies.sort();
    let mut table: HashMap<Vec<u8>, (i64, i64)> = HashMap::with_capacity(movies.len());
    let mut offset = 0i64;
    for movie in &movies {
        let (min, max) = bounds[movie];
        table.insert(movie.clone(), (min, offset));
        offset += max - min + 1;
    }
    let capacity = offset;
    if !(1..=2_147_483_647).contains(&capacity) {
        return Err(Error::pacbio(format!(
            "invalid global ZMW range: {capacity} (expected 1..=2147483647)"
        )));
    }

    let mut pairs_vec: Vec<(Vec<u8>, i64)> = pairs.into_iter().collect();
    pairs_vec.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    Ok(Merged {
        allocation: Allocation { table, capacity },
        pairs: pairs_vec,
        movies: movies.len(),
        records,
    })
}

/// Write the `zmw_map.tsv` mapping every original `(movie, zmw)` to the synthetic identity.
fn write_zmw_map(
    path: &Path,
    pairs: &[(Vec<u8>, i64)],
    allocation: &Allocation,
    synthetic_movie: &str,
) -> Result<()> {
    let mut w = BufWriter::new(
        File::create(path)
            .map_err(|e| Error::pacbio(format!("cannot create {}: {e}", path.display())))?,
    );
    writeln!(w, "original_movie\toriginal_zmw\tmovie\tzmw")?;
    for (movie, zmw) in pairs {
        let new_zmw = allocation
            .map_zmw(movie, *zmw)
            .expect("every observed movie has an allocation");
        writeln!(
            w,
            "{}\t{}\t{}\t{}",
            String::from_utf8_lossy(movie),
            zmw,
            synthetic_movie,
            new_zmw
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Rewrite one BAM to the synthetic movie, returning the output path.
fn rewrite_file(
    path: &Path,
    outdir: &Path,
    synthetic_movie: &str,
    rg_id: &str,
    allocation: &Allocation,
) -> Result<PathBuf> {
    let (mut reader, in_header) = open_reader(path)?;
    let out_header = normalized_header(&in_header, synthetic_movie, rg_id, path)?;

    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Error::pacbio(format!("invalid input path: {}", path.display())))?;
    let stem = file_name.strip_suffix(".bam").unwrap_or(file_name);
    let out_path = outdir.join(format!("{stem}.normalized.bam"));

    let mut writer = create_writer(&out_path)?;
    writer
        .write_header(&out_header)
        .map_err(|e| Error::pacbio(format!("{}: writing header: {e}", out_path.display())))?;

    let rg_value = Value::String(rg_id.as_bytes().to_vec().into());

    loop {
        // A fresh RecordBuf each iteration keeps decode/mutate/encode independent of any field
        // carried over from the previous record.
        let mut record = RecordBuf::default();
        let n = reader
            .read_record_buf(&in_header, &mut record)
            .map_err(|e| Error::pacbio(format!("{}: reading record: {e}", path.display())))?;
        if n == 0 {
            break;
        }

        let (movie, zmw, rest) = {
            let name = record.name().ok_or_else(|| {
                Error::pacbio(format!("{}: record without QNAME", path.display()))
            })?;
            let bytes: &[u8] = name.as_ref();
            let parsed = parse_subread_name(bytes)?;
            (parsed.movie.to_vec(), parsed.zmw, parsed.rest.to_vec())
        };

        let new_zmw = allocation.map_zmw(&movie, zmw).ok_or_else(|| {
            Error::pacbio(format!(
                "{}: no global ZMW for source movie {}",
                path.display(),
                String::from_utf8_lossy(&movie)
            ))
        })?;

        if record.data().get(&Tag::READ_GROUP).is_none() || record.data().get(&ZM_TAG).is_none() {
            return Err(Error::pacbio(format!(
                "{}: record lacks RG or zm tag (movie {} zmw {})",
                path.display(),
                String::from_utf8_lossy(&movie),
                zmw
            )));
        }

        let mut new_name = Vec::with_capacity(synthetic_movie.len() + rest.len() + 22);
        new_name.extend_from_slice(synthetic_movie.as_bytes());
        new_name.push(b'/');
        new_name.extend_from_slice(new_zmw.to_string().as_bytes());
        new_name.push(b'/');
        new_name.extend_from_slice(&rest);
        *record.name_mut() = Some(new_name.into());

        let new_zmw_i32 = i32::try_from(new_zmw)
            .map_err(|_| Error::pacbio(format!("global ZMW {new_zmw} exceeds i32 range")))?;
        record.data_mut().insert(Tag::READ_GROUP, rg_value.clone());
        record.data_mut().insert(ZM_TAG, Value::Int32(new_zmw_i32));

        writer
            .write_alignment_record(&out_header, &record)
            .map_err(|e| Error::pacbio(format!("{}: writing record: {e}", out_path.display())))?;
    }

    writer
        .try_finish()
        .map_err(|e| Error::pacbio(format!("{}: finishing BAM: {e}", out_path.display())))?;
    Ok(out_path)
}

/// Build the normalized header: collapse to a single `SUBREAD` read group keyed by `rg_id` with
/// `PU` set to the synthetic movie, preserving all other header lines. Errors if there is no
/// `@RG` or the first one is not `READTYPE=SUBREAD`.
fn normalized_header(
    input: &sam::Header,
    synthetic_movie: &str,
    rg_id: &str,
    path: &Path,
) -> Result<sam::Header> {
    let mut header = input.clone();

    let mut rg_map = header
        .read_groups()
        .first()
        .map(|(_, map)| map.clone())
        .ok_or_else(|| Error::pacbio(format!("{}: BAM header has no @RG line", path.display())))?;

    let is_subread = rg_map
        .other_fields()
        .get(&read_group::tag::DESCRIPTION)
        .and_then(|ds| std::str::from_utf8(ds).ok())
        .map(|ds| {
            ds.split(';')
                .filter_map(|field| field.strip_prefix("READTYPE="))
                .any(|value| value == "SUBREAD")
        })
        .unwrap_or(false);
    if !is_subread {
        return Err(Error::pacbio(format!(
            "{}: input @RG is not READTYPE=SUBREAD",
            path.display()
        )));
    }

    rg_map.other_fields_mut().insert(
        read_group::tag::PLATFORM_UNIT,
        synthetic_movie.as_bytes().to_vec().into(),
    );

    let read_groups = header.read_groups_mut();
    read_groups.clear();
    read_groups.insert(rg_id.as_bytes().to_vec().into(), rg_map);

    Ok(header)
}
