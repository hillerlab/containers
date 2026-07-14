//! Low-level input/output: BED reading via `genepred`, mapping/chrom-size parsing, BED writing.
//!
//! Rule checks (uniqueness, mapping consistency, bounds, …) live in [`crate::validate`]; this
//! module only performs structural parsing and hard-error reporting.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use genepred::{Bed12, GenePred, Reader, Writer};

use crate::error::{Error, Result};
use crate::model::Transcript;

/// A raw transcript→gene mapping entry with its source line number.
#[derive(Debug, Clone)]
pub struct MappingEntry {
    /// Transcript identifier.
    pub transcript: String,
    /// Gene identifier.
    pub gene: String,
    /// 1-based source line number.
    pub line: usize,
}

/// Read every BED12 record as a `genepred::GenePred`. Hard error on I/O or structural parse
/// failure. Coordinates in returned records are absolute.
pub fn read_bed_records<P: AsRef<Path>>(path: P) -> Result<Vec<GenePred>> {
    let path = path.as_ref();
    let reader = Reader::<Bed12>::from_path(path)
        .map_err(|e| Error::BedRead(format!("{}: {e}", path.display())))?;
    let mut out = Vec::new();
    for record in reader {
        let record = record.map_err(|e| Error::BedRead(format!("{}: {e}", path.display())))?;
        out.push(record);
    }
    Ok(out)
}

/// Whether a mapping file's first line looks like a header rather than data.
fn looks_like_mapping_header(f0: &str, f1: &str) -> bool {
    let a = f0.to_ascii_lowercase();
    let b = f1.to_ascii_lowercase();
    a.contains("transcript") && b.contains("gene")
}

/// Parse a transcript→gene TSV (`TRANSCRIPT_ID\tGENE_ID`). A single leading header line is
/// tolerated only if it clearly matches the column names; otherwise every line is data.
pub fn read_transcript_gene<P: AsRef<Path>>(path: P) -> Result<Vec<MappingEntry>> {
    let path = path.as_ref();
    let display = path.display().to_string();
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let lineno = idx + 1;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut fields = trimmed.split('\t');
        let f0 = fields.next().unwrap_or("");
        let f1 = fields.next();
        let extra = fields.next();
        let (transcript, gene) = match f1 {
            Some(g) => (f0, g),
            None => {
                return Err(Error::parse(
                    &display,
                    lineno,
                    "expected two tab-separated columns (TRANSCRIPT_ID<TAB>GENE_ID)",
                ));
            }
        };
        if extra.is_some() {
            return Err(Error::parse(
                &display,
                lineno,
                "expected exactly two tab-separated columns",
            ));
        }
        if idx == 0 && looks_like_mapping_header(transcript, gene) {
            continue; // header row
        }
        if transcript.is_empty() || gene.is_empty() {
            return Err(Error::parse(
                &display,
                lineno,
                "empty transcript or gene id",
            ));
        }
        entries.push(MappingEntry {
            transcript: transcript.to_string(),
            gene: gene.to_string(),
            line: lineno,
        });
    }
    Ok(entries)
}

/// Parse a two-column `chrom.sizes` file (`CHROM\tSIZE`). Header lines are not expected.
pub fn read_chrom_sizes<P: AsRef<Path>>(path: P) -> Result<HashMap<String, u64>> {
    let path = path.as_ref();
    let display = path.display().to_string();
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut sizes = HashMap::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let lineno = idx + 1;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut fields = trimmed.split('\t');
        let chrom = fields.next().unwrap_or("");
        let size = fields.next();
        let size = match size {
            Some(s) => s.parse::<u64>().map_err(|_| {
                Error::parse(&display, lineno, format!("invalid chromosome size '{s}'"))
            })?,
            None => {
                return Err(Error::parse(
                    &display,
                    lineno,
                    "expected two tab-separated columns (CHROM<TAB>SIZE)",
                ))
            }
        };
        if chrom.is_empty() {
            return Err(Error::parse(&display, lineno, "empty chromosome name"));
        }
        if sizes.insert(chrom.to_string(), size).is_some() {
            return Err(Error::parse(
                &display,
                lineno,
                format!("duplicate chromosome '{chrom}'"),
            ));
        }
    }
    Ok(sizes)
}

/// Write transcripts to a BED12 file via `genepred`'s writer, in the order given.
pub fn write_bed<P: AsRef<Path>>(path: P, transcripts: &[Transcript]) -> Result<()> {
    let records: Vec<GenePred> = transcripts.iter().map(Transcript::to_genepred).collect();
    Writer::<Bed12>::to_path(path.as_ref(), &records)
        .map_err(|e| Error::BedWrite(format!("{}: {e}", path.as_ref().display())))
}

/// Write a headerless transcript→gene mapping TSV.
pub fn write_transcript_gene<P: AsRef<Path>>(path: P, transcripts: &[Transcript]) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    for t in transcripts {
        writeln!(w, "{}\t{}", t.id, t.gene_id)?;
    }
    w.flush()?;
    Ok(())
}
