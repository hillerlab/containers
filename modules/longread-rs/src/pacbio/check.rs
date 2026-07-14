//! `longread check` — fail before merge if CCS chunks overlap in PacBio's movie/ZMW key space.
//!
//! A CCS movie/ZMW may occur in exactly one chunk. This subcommand scans every chunk BAM in
//! parallel, collects each read's `movie/zmw` key, and errors if any key appears in more than one
//! record across the whole collection. On success it writes a marker file. It replaces a
//! `samtools view | awk | sort | uniq -d` pipeline.

use std::collections::{BTreeSet, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use noodles_bam as bam;
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::pacbio::{build_pool, movie_zmw_key, open_reader};

/// How many duplicate keys to include in an error message (mirrors the old `head -n 20`).
const MAX_REPORTED_DUPLICATES: usize = 20;

/// Parameters for `longread check`.
#[derive(Debug, Clone)]
pub struct CheckParams {
    /// Input CCS chunk BAMs.
    pub bams: Vec<PathBuf>,
    /// Marker file to create on success.
    pub out: PathBuf,
    /// Worker count (0 = all cores), bounded by `max_open_files`.
    pub threads: usize,
    /// Upper bound on simultaneously open file descriptors.
    pub max_open_files: usize,
}

/// Summary statistics for a check run.
#[derive(Debug, Clone)]
pub struct CheckStats {
    /// Number of input BAMs.
    pub input_bams: usize,
    /// Total records (movie/ZMW keys) scanned.
    pub records: u64,
    /// Number of distinct movie/ZMW keys.
    pub distinct_keys: usize,
}

/// Run `longread check`.
pub fn run(params: &CheckParams) -> Result<CheckStats> {
    if params.bams.is_empty() {
        return Err(Error::pacbio("no input BAMs provided"));
    }

    let pool = build_pool(params.threads, params.max_open_files, 1)?;
    let per_file: Vec<Vec<Vec<u8>>> = pool.install(|| {
        params
            .bams
            .par_iter()
            .map(|path| scan_keys(path))
            .collect::<Result<Vec<_>>>()
    })?;

    // Deterministic global duplicate detection over movie/ZMW keys.
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut duplicates: BTreeSet<Vec<u8>> = BTreeSet::new();
    let mut records = 0u64;
    for keys in per_file {
        for key in keys {
            records += 1;
            if seen.contains(&key) {
                duplicates.insert(key);
            } else {
                seen.insert(key);
            }
        }
    }

    if !duplicates.is_empty() {
        let shown: Vec<String> = duplicates
            .iter()
            .take(MAX_REPORTED_DUPLICATES)
            .map(|key| String::from_utf8_lossy(key).into_owned())
            .collect();
        return Err(Error::pacbio(format!(
            "CCS chunks overlap; {} duplicate movie/ZMW key(s), including:\n{}",
            duplicates.len(),
            shown.join("\n")
        )));
    }

    File::create(&params.out)
        .map_err(|e| Error::pacbio(format!("cannot create {}: {e}", params.out.display())))?;

    Ok(CheckStats {
        input_bams: params.bams.len(),
        records,
        distinct_keys: seen.len(),
    })
}

/// Scan one CCS BAM and return the `movie/zmw` key of every record.
fn scan_keys(path: &Path) -> Result<Vec<Vec<u8>>> {
    let (mut reader, _header) = open_reader(path)?;
    let mut keys = Vec::new();
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
        keys.push(movie_zmw_key(bytes)?);
    }
    Ok(keys)
}
