//! `longread rg` — canonicalize PBSIM3 subread BAMs as one synthetic PacBio movie.
//!
//! CCS chunking is defined for a single movie, but PBSIM3 restarts ZMW numbering for every
//! simulated movie and emits the placeholder read group `ID:ffffffff`. This subcommand:
//!
//! 1. assigns every original ZMW a deterministic global ZMW;
//! 2. rewrites each record's QNAME and `zm` tag to that global ZMW;
//! 3. rewrites all records to one specification-compliant `SUBREAD` read group;
//! 4. emits a `zmw_map.tsv` preserving the original identity.
//!
//! **ZMW allocation is keyed per source BAM, not per movie name.** Each PBSIM3 output file has its
//! own restarted ZMW space; a molecule's subreads never span files. In `trans` mode there is one
//! movie per file, so this is identical to keying on the movie. In `wgs` mode PBSIM3 appends the
//! (unpadded) reference index to `--id-prefix`, so movie names *collide* across files
//! (e.g. `id-prefix=movie.x.13, ref=1` and `id-prefix=movie.x.1, ref=31` both yield `movie.x.131`),
//! each restarting ZMW at 1 — so the movie name is not a safe key, but the file is.
//!
//! It replaces a `samtools view | awk | sort` pipeline that decompressed every BAM ~5 times. Here
//! the work is exactly two parallel passes (scan, then rewrite) with a serial, deterministic
//! allocation step in between, so output is byte-identical regardless of thread count.

use std::collections::HashMap;
use std::collections::HashSet;
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
    /// Input subread BAMs (each with its own restarted ZMW space).
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
    /// Number of non-empty source BAMs that received a ZMW range.
    pub source_files: usize,
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
struct FileScan {
    /// Source BAM path (the allocation key).
    path: PathBuf,
    /// Movie name observed in the file (provenance); empty if the file had no records.
    movie: Vec<u8>,
    /// Minimum ZMW in the file (`i64::MAX` if empty).
    min_zmw: i64,
    /// Maximum ZMW in the file (`i64::MIN` if empty).
    max_zmw: i64,
    /// Distinct observed ZMWs, ascending.
    zmws: Vec<i64>,
    /// Record count.
    records: u64,
}

/// One `zmw_map.tsv` row.
struct MapRow {
    source_bam: String,
    original_movie: Vec<u8>,
    original_zmw: i64,
    new_zmw: i64,
}

/// Deterministic per-file ZMW allocation: `new_zmw = offset + (zmw - min) + 1`.
struct Allocation {
    /// source BAM path -> `(min_zmw, offset)`.
    table: HashMap<PathBuf, (i64, i64)>,
    /// Total global ZMW capacity.
    capacity: i64,
}

impl Allocation {
    fn map_zmw(&self, path: &Path, zmw: i64) -> Option<i64> {
        self.table
            .get(path)
            .map(|(min, offset)| offset + (zmw - min) + 1)
    }
}

/// Result of merging per-file scans and assigning global ZMW offsets.
struct Merged {
    /// Deterministic per-file ZMW allocation.
    allocation: Allocation,
    /// `zmw_map.tsv` rows, ordered by `(source_bam, zmw)`.
    map_rows: Vec<MapRow>,
    /// Number of non-empty source files.
    source_files: usize,
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

    write_zmw_map(&params.zmw_map, &merged.map_rows, &synthetic_movie)?;

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
        source_files: merged.source_files,
        records: merged.records,
        zmw_capacity: merged.allocation.capacity,
        rg_id,
        outputs,
    })
}

/// Scan one BAM: validate its read group, then accumulate this file's ZMW bounds and observed
/// ZMWs. QNAMEs must be unique *within* the file (a real integrity check); duplicates *across*
/// files are expected in `wgs` mode and are disambiguated by per-file allocation.
fn scan_file(path: &Path, synthetic_movie: &str, rg_id: &str) -> Result<FileScan> {
    let (mut reader, header) = open_reader(path)?;
    // Reuse the header builder purely to validate the read group early.
    normalized_header(&header, synthetic_movie, rg_id, path)?;

    let mut movie: Vec<u8> = Vec::new();
    let mut min_zmw = i64::MAX;
    let mut max_zmw = i64::MIN;
    let mut zmws: HashSet<i64> = HashSet::new();
    let mut qnames: HashSet<Vec<u8>> = HashSet::new();
    let mut records = 0u64;

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

        if records == 0 {
            movie = parsed.movie.to_vec();
        }
        if parsed.zmw < min_zmw {
            min_zmw = parsed.zmw;
        }
        if parsed.zmw > max_zmw {
            max_zmw = parsed.zmw;
        }
        zmws.insert(parsed.zmw);
        if !qnames.insert(bytes.to_vec()) {
            return Err(Error::pacbio(format!(
                "{}: duplicate QNAME within file: {}",
                path.display(),
                String::from_utf8_lossy(bytes)
            )));
        }
        records += 1;
    }

    let mut zmws: Vec<i64> = zmws.into_iter().collect();
    zmws.sort_unstable();
    Ok(FileScan {
        path: path.to_path_buf(),
        movie,
        min_zmw,
        max_zmw,
        zmws,
        records,
    })
}

/// Merge per-file scans and assign each non-empty file a disjoint global ZMW range, in a
/// deterministic order (by source path).
fn merge_and_allocate(scans: Vec<FileScan>) -> Result<Merged> {
    let mut files: Vec<FileScan> = scans.into_iter().filter(|s| s.records > 0).collect();
    files.sort_by(|a, b| a.path.cmp(&b.path));

    let mut table: HashMap<PathBuf, (i64, i64)> = HashMap::with_capacity(files.len());
    let mut map_rows: Vec<MapRow> = Vec::new();
    let mut offset = 0i64;
    let mut records = 0u64;

    for file in &files {
        table.insert(file.path.clone(), (file.min_zmw, offset));
        let source_bam = file
            .path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        for &zmw in &file.zmws {
            map_rows.push(MapRow {
                source_bam: source_bam.clone(),
                original_movie: file.movie.clone(),
                original_zmw: zmw,
                new_zmw: offset + (zmw - file.min_zmw) + 1,
            });
        }
        offset += file.max_zmw - file.min_zmw + 1;
        records += file.records;
    }

    let capacity = offset;
    if !(1..=2_147_483_647).contains(&capacity) {
        return Err(Error::pacbio(format!(
            "invalid global ZMW range: {capacity} (expected 1..=2147483647)"
        )));
    }

    Ok(Merged {
        allocation: Allocation { table, capacity },
        map_rows,
        source_files: files.len(),
        records,
    })
}

/// Write the `zmw_map.tsv` mapping every original `(source_bam, movie, zmw)` to the synthetic
/// identity.
fn write_zmw_map(path: &Path, rows: &[MapRow], synthetic_movie: &str) -> Result<()> {
    let mut w = BufWriter::new(
        File::create(path)
            .map_err(|e| Error::pacbio(format!("cannot create {}: {e}", path.display())))?,
    );
    writeln!(w, "source_bam\toriginal_movie\toriginal_zmw\tmovie\tzmw")?;
    for row in rows {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}",
            row.source_bam,
            String::from_utf8_lossy(&row.original_movie),
            row.original_zmw,
            synthetic_movie,
            row.new_zmw
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

        let (zmw, rest) = {
            let name = record.name().ok_or_else(|| {
                Error::pacbio(format!("{}: record without QNAME", path.display()))
            })?;
            let bytes: &[u8] = name.as_ref();
            let parsed = parse_subread_name(bytes)?;
            (parsed.zmw, parsed.rest.to_vec())
        };

        let new_zmw = allocation.map_zmw(path, zmw).ok_or_else(|| {
            Error::pacbio(format!(
                "{}: source file has no allocated ZMW range",
                path.display()
            ))
        })?;

        if record.data().get(&Tag::READ_GROUP).is_none() || record.data().get(&ZM_TAG).is_none() {
            return Err(Error::pacbio(format!(
                "{}: record lacks RG or zm tag (zmw {zmw})",
                path.display(),
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
