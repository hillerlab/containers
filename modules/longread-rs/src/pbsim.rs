//! `longread pbsim` — build the PBSIM3 transcript-mode input file (spec §5.2).
//!
//! Combines extracted transcript sequences (`xloci --as-tsv`) with the per-isoform molecule
//! counts from `prepare` into PBSIM3's four-column transcript dataset:
//!
//! ```text
//! TRANSCRIPT_ID    SENSE_COUNT    ANTISENSE_COUNT    SEQUENCE
//! ```
//!
//! Zero-count transcripts are omitted (they would produce no reads). Fusion sequences are
//! optionally validated against the concatenation of their component isoform sequences.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Parameters for `longread pbsim`.
#[derive(Debug, Clone)]
pub struct PbsimParams {
    /// `xloci --as-tsv` output: `TRANSCRIPT_ID\tSEQUENCE`.
    pub sequences: PathBuf,
    /// `${prefix}.isoform_depth.tsv` from `prepare`.
    pub isoform_depth: PathBuf,
    /// Optional `${prefix}.manifest.tsv` used to validate fusion sequences.
    pub manifest: Option<PathBuf>,
    /// Output PBSIM3 transcript file.
    pub output: PathBuf,
}

/// Statistics from a `pbsim` run.
#[derive(Debug, Clone)]
pub struct PbsimStats {
    /// Transcripts written (expressed).
    pub written: usize,
    /// Zero-count transcripts omitted.
    pub omitted: usize,
    /// Fusion sequences validated against their components.
    pub fusions_validated: usize,
}

/// Reject ids/sequences containing tab or newline (would corrupt the TSV).
fn ensure_no_control(field: &str, what: &str, path: &str, line: usize) -> Result<()> {
    if field.contains('\t') || field.contains('\n') || field.contains('\r') {
        return Err(Error::parse(
            path,
            line,
            format!("{what} contains a tab or newline"),
        ));
    }
    Ok(())
}

/// Read `TRANSCRIPT_ID\tSEQUENCE`, rejecting duplicates and empty sequences.
fn read_sequences(path: &Path) -> Result<HashMap<String, String>> {
    let display = path.display().to_string();
    let reader = BufReader::new(std::fs::File::open(path)?);
    let mut map = HashMap::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let lineno = idx + 1;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            continue;
        }
        let mut it = trimmed.splitn(2, '\t');
        let id = it.next().unwrap_or("");
        let seq = it
            .next()
            .ok_or_else(|| Error::parse(&display, lineno, "expected TRANSCRIPT_ID<TAB>SEQUENCE"))?;
        ensure_no_control(id, "transcript id", &display, lineno)?;
        ensure_no_control(seq, "sequence", &display, lineno)?;
        if id.is_empty() {
            return Err(Error::parse(&display, lineno, "empty transcript id"));
        }
        if seq.is_empty() {
            return Err(Error::parse(
                &display,
                lineno,
                format!("empty sequence for '{id}'"),
            ));
        }
        if map.insert(id.to_string(), seq.to_string()).is_some() {
            return Err(Error::parse(
                &display,
                lineno,
                format!("duplicate sequence entry for transcript '{id}'"),
            ));
        }
    }
    Ok(map)
}

/// One expressed transcript destined for PBSIM3.
struct DepthRow {
    id: String,
    sense: u64,
    antisense: u64,
}

/// Read `isoform_depth.tsv` (skips the header), returning rows with a positive read count.
fn read_isoform_depth(path: &Path) -> Result<(Vec<DepthRow>, usize)> {
    let display = path.display().to_string();
    let reader = BufReader::new(std::fs::File::open(path)?);
    let mut rows = Vec::new();
    let mut omitted = 0usize;
    let mut seen: HashMap<String, usize> = HashMap::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let lineno = idx + 1;
        if idx == 0 {
            // header line
            continue;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            continue;
        }
        let f: Vec<&str> = trimmed.split('\t').collect();
        if f.len() < 4 {
            return Err(Error::parse(
                &display,
                lineno,
                "expected TRANSCRIPT_ID GENE_ID ISOFORM_DEPTH ANTISENSE_DEPTH ...",
            ));
        }
        let id = f[0].to_string();
        let sense: u64 = f[2].parse().map_err(|_| {
            Error::parse(
                &display,
                lineno,
                format!("invalid ISOFORM_DEPTH '{}'", f[2]),
            )
        })?;
        let antisense: u64 = f[3].parse().map_err(|_| {
            Error::parse(
                &display,
                lineno,
                format!("invalid ANTISENSE_DEPTH '{}'", f[3]),
            )
        })?;
        if let Some(prev) = seen.insert(id.clone(), lineno) {
            return Err(Error::parse(
                &display,
                lineno,
                format!("duplicate transcript '{id}' (also line {prev})"),
            ));
        }
        if sense + antisense == 0 {
            omitted += 1;
            continue;
        }
        rows.push(DepthRow {
            id,
            sense,
            antisense,
        });
    }
    Ok((rows, omitted))
}

/// Validate that each fusion's sequence equals the concatenation of its partner isoforms'
/// sequences (transcript orientation: 5′ then 3′). Returns the number validated.
fn validate_fusions(manifest: &Path, sequences: &HashMap<String, String>) -> Result<usize> {
    let display = manifest.display().to_string();
    let reader = BufReader::new(std::fs::File::open(manifest)?);
    let mut validated = 0usize;
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        if idx == 0 {
            continue; // header
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 11 {
            continue;
        }
        let is_fusion = f[6] == "true";
        if !is_fusion {
            continue;
        }
        let id = f[0];
        let pt5 = f[8];
        let pt3 = f[10];
        let (Some(fusion_seq), Some(s5), Some(s3)) =
            (sequences.get(id), sequences.get(pt5), sequences.get(pt3))
        else {
            // A partner or the fusion itself is not in the sequence set; skip silently only if
            // the fusion sequence is entirely absent (it may be zero-count and un-extracted).
            continue;
        };
        let expected_len = s5.len() + s3.len();
        if fusion_seq.len() != expected_len {
            return Err(Error::parse(
                &display,
                idx + 1,
                format!(
                    "fusion '{id}' sequence length {} != partner sum {} ({}+{}); xloci extraction mismatch",
                    fusion_seq.len(),
                    expected_len,
                    s5.len(),
                    s3.len()
                ),
            ));
        }
        let concat = format!("{s5}{s3}");
        if *fusion_seq != concat {
            return Err(Error::parse(
                &display,
                idx + 1,
                format!(
                    "fusion '{id}' sequence does not equal 5′ partner '{pt5}' + 3′ partner '{pt3}'"
                ),
            ));
        }
        validated += 1;
    }
    Ok(validated)
}

/// Run `longread pbsim`.
pub fn run(params: &PbsimParams) -> Result<PbsimStats> {
    let sequences = read_sequences(&params.sequences)?;
    let (rows, omitted) = read_isoform_depth(&params.isoform_depth)?;

    let fusions_validated = match &params.manifest {
        Some(m) => validate_fusions(m, &sequences)?,
        None => 0,
    };

    if let Some(parent) = params.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut w = BufWriter::new(std::fs::File::create(&params.output)?);
    let mut written = 0usize;
    for row in &rows {
        let seq = sequences.get(&row.id).ok_or_else(|| {
            Error::config(format!(
                "expressed transcript '{}' has no sequence in {}",
                row.id,
                params.sequences.display()
            ))
        })?;
        writeln!(w, "{}\t{}\t{}\t{}", row.id, row.sense, row.antisense, seq)?;
        written += 1;
    }
    w.flush()?;

    Ok(PbsimStats {
        written,
        omitted,
        fusions_validated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::File::create(&p)
            .unwrap()
            .write_all(content.as_bytes())
            .unwrap();
        p
    }

    const ISO_HEADER: &str =
        "TRANSCRIPT_ID\tGENE_ID\tISOFORM_DEPTH\tANTISENSE_DEPTH\tRELATIVE_WEIGHT\n";

    #[test]
    fn builds_input_and_omits_zero_count() {
        let dir = TempDir::new().unwrap();
        let seqs = write(dir.path(), "seq.tsv", "t1\tACGT\nt2\tGGGG\nt3\tTTTT\n");
        let iso = write(
            dir.path(),
            "iso.tsv",
            &format!("{ISO_HEADER}t1\tg1\t5\t0\t0.5\nt2\tg1\t0\t0\t0.0\nt3\tg2\t2\t0\t1.0\n"),
        );
        let out = dir.path().join("pbsim.tsv");
        let stats = run(&PbsimParams {
            sequences: seqs,
            isoform_depth: iso,
            manifest: None,
            output: out.clone(),
        })
        .unwrap();
        assert_eq!(stats.written, 2);
        assert_eq!(stats.omitted, 1);
        let text = std::fs::read_to_string(&out).unwrap();
        assert!(text.contains("t1\t5\t0\tACGT"));
        assert!(text.contains("t3\t2\t0\tTTTT"));
        assert!(!text.contains("t2"), "zero-count t2 must be omitted");
    }

    #[test]
    fn errors_on_missing_expressed_sequence() {
        let dir = TempDir::new().unwrap();
        let seqs = write(dir.path(), "seq.tsv", "t1\tACGT\n");
        let iso = write(
            dir.path(),
            "iso.tsv",
            &format!("{ISO_HEADER}t2\tg1\t5\t0\t1.0\n"),
        );
        let err = run(&PbsimParams {
            sequences: seqs,
            isoform_depth: iso,
            manifest: None,
            output: dir.path().join("o.tsv"),
        })
        .unwrap_err();
        assert!(err.to_string().contains("no sequence"), "got: {err}");
    }

    #[test]
    fn rejects_duplicate_and_empty_sequences() {
        let dir = TempDir::new().unwrap();
        let dup = write(dir.path(), "dup.tsv", "t1\tAC\nt1\tGT\n");
        let iso = write(
            dir.path(),
            "iso.tsv",
            &format!("{ISO_HEADER}t1\tg1\t1\t0\t1.0\n"),
        );
        assert!(run(&PbsimParams {
            sequences: dup,
            isoform_depth: iso.clone(),
            manifest: None,
            output: dir.path().join("o.tsv"),
        })
        .is_err());

        let empty = write(dir.path(), "empty.tsv", "t1\t\n");
        assert!(run(&PbsimParams {
            sequences: empty,
            isoform_depth: iso,
            manifest: None,
            output: dir.path().join("o2.tsv"),
        })
        .is_err());
    }

    #[test]
    fn validates_fusion_against_components() {
        let dir = TempDir::new().unwrap();
        // fusion sequence = 5' partner + 3' partner.
        let seqs = write(dir.path(), "seq.tsv", "p5\tAAAA\np3\tCCCC\nfus\tAAAACCCC\n");
        let iso = write(
            dir.path(),
            "iso.tsv",
            &format!("{ISO_HEADER}fus\tFUSION_GENE::g5::g3\t3\t0\t1.0\n"),
        );
        // manifest: columns id,gene,parent,event,is_orig,is_gen,is_fusion,pg5,pt5,pg3,pt3,...
        let man = write(
            dir.path(),
            "man.tsv",
            "transcript_id\tgene_id\tparent_transcript_id\tevent_type\tis_original\tis_generated\tis_fusion\tpartner_gene_5p\tpartner_transcript_5p\tpartner_gene_3p\tpartner_transcript_3p\tchromosome\tstrand\texon_count\ttranscript_length\tseed\n\
             fus\tFUSION_GENE::g5::g3\t.\tfusion\tfalse\tfalse\ttrue\tg5\tp5\tg3\tp3\tchr1\t+\t2\t8\t1\n",
        );
        let stats = run(&PbsimParams {
            sequences: seqs,
            isoform_depth: iso,
            manifest: Some(man),
            output: dir.path().join("o.tsv"),
        })
        .unwrap();
        assert_eq!(stats.fusions_validated, 1);
    }

    #[test]
    fn rejects_fusion_sequence_mismatch() {
        let dir = TempDir::new().unwrap();
        let seqs = write(dir.path(), "seq.tsv", "p5\tAAAA\np3\tCCCC\nfus\tAAAAGGGG\n");
        let iso = write(
            dir.path(),
            "iso.tsv",
            &format!("{ISO_HEADER}fus\tFG\t3\t0\t1.0\n"),
        );
        let man = write(
            dir.path(),
            "man.tsv",
            "h\nfus\tFG\t.\tfusion\tfalse\tfalse\ttrue\tg5\tp5\tg3\tp3\tchr1\t+\t2\t8\t1\n",
        );
        let err = run(&PbsimParams {
            sequences: seqs,
            isoform_depth: iso,
            manifest: Some(man),
            output: dir.path().join("o.tsv"),
        })
        .unwrap_err();
        assert!(err.to_string().contains("does not equal"), "got: {err}");
    }
}
