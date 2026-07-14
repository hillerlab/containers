//! `longread split` — balanced chunking of the PBSIM3 transcript file (spec §5.4).
//!
//! Splits the four-column transcript dataset into `N` chunks using greedy largest-first bin
//! packing on estimated work (`transcript_length × sense_count × pass_count`), so PBSIM3 tasks
//! finish in roughly equal wall-clock time. Each chunk gets a unique prefix to keep movie/ZMW
//! identifiers distinct at merge time. Tie-breaks are deterministic (work desc, then id).

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Parameters for `longread split`.
#[derive(Debug, Clone)]
pub struct SplitParams {
    /// Input PBSIM3 four-column transcript file.
    pub transcript: PathBuf,
    /// Number of chunks requested.
    pub chunks: usize,
    /// Number of sequencing passes (for the work estimate).
    pub pass_count: u64,
    /// Output directory.
    pub outdir: PathBuf,
    /// Output prefix stem.
    pub prefix: String,
}

/// Statistics from a `split` run.
#[derive(Debug, Clone)]
pub struct SplitStats {
    /// Number of non-empty chunks written.
    pub chunks_written: usize,
    /// Total transcripts distributed.
    pub transcripts: usize,
    /// Path to the `chunks.tsv` manifest.
    pub chunks_manifest: PathBuf,
}

/// A single PBSIM3 transcript record (verbatim four columns).
struct Record {
    id: String,
    sense: u64,
    antisense: u64,
    seq: String,
}

impl Record {
    /// Estimated work = transcript_length × sense_count × pass_count (spec §5.4).
    fn work(&self, pass_count: u64) -> u128 {
        (self.seq.len() as u128) * (self.sense as u128) * (pass_count.max(1) as u128)
    }
}

/// Parse the four-column transcript file.
fn read_transcripts(path: &Path) -> Result<Vec<Record>> {
    let display = path.display().to_string();
    let reader = BufReader::new(std::fs::File::open(path)?);
    let mut out = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let lineno = idx + 1;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            continue;
        }
        let f: Vec<&str> = trimmed.split('\t').collect();
        if f.len() != 4 {
            return Err(Error::parse(
                &display,
                lineno,
                format!("expected 4 tab-separated columns, got {}", f.len()),
            ));
        }
        let sense: u64 = f[1].parse().map_err(|_| {
            Error::parse(&display, lineno, format!("invalid sense count '{}'", f[1]))
        })?;
        let antisense: u64 = f[2].parse().map_err(|_| {
            Error::parse(
                &display,
                lineno,
                format!("invalid antisense count '{}'", f[2]),
            )
        })?;
        out.push(Record {
            id: f[0].to_string(),
            sense,
            antisense,
            seq: f[3].to_string(),
        });
    }
    Ok(out)
}

/// Run `longread split`.
pub fn run(params: &SplitParams) -> Result<SplitStats> {
    if params.chunks == 0 {
        return Err(Error::config("--pbsim-chunks must be >= 1"));
    }
    let records = read_transcripts(&params.transcript)?;
    if records.is_empty() {
        return Err(Error::config(format!(
            "no transcripts to split in {}",
            params.transcript.display()
        )));
    }

    // Order indices by work descending, then transcript id ascending (deterministic).
    let mut order: Vec<usize> = (0..records.len()).collect();
    order.sort_by(|&a, &b| {
        records[b]
            .work(params.pass_count)
            .cmp(&records[a].work(params.pass_count))
            .then_with(|| records[a].id.cmp(&records[b].id))
    });

    // Greedy largest-first bin packing into min(chunks, n) bins.
    let n_bins = params.chunks.min(records.len());
    let mut loads = vec![0u128; n_bins];
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); n_bins];
    for &idx in &order {
        // Lightest bin; ties broken by lowest bin index.
        let mut best = 0usize;
        for b in 1..n_bins {
            if loads[b] < loads[best] {
                best = b;
            }
        }
        loads[best] += records[idx].work(params.pass_count);
        bins[best].push(idx);
    }

    std::fs::create_dir_all(&params.outdir)?;

    // Keep only non-empty bins, renumbered sequentially from 1.
    let non_empty: Vec<usize> = (0..n_bins).filter(|&b| !bins[b].is_empty()).collect();
    let chunks_manifest = params.outdir.join("chunks.tsv");
    let mut manifest = BufWriter::new(std::fs::File::create(&chunks_manifest)?);
    writeln!(
        manifest,
        "CHUNK\tCHUNK_PREFIX\tTRANSCRIPT_FILE\tN_TRANSCRIPTS\tTOTAL_WORK"
    )?;

    let mut transcripts = 0usize;
    for (new_idx, &bin) in non_empty.iter().enumerate() {
        let chunk_no = new_idx + 1;
        let chunk_prefix = format!("{}_chunk_{:04}", params.prefix, chunk_no);
        let file_name = format!("chunk_{chunk_no:04}.transcript.tsv");
        let chunk_path = params.outdir.join(&file_name);

        // Emit records in the deterministic global order for reproducibility.
        let mut members = bins[bin].clone();
        members.sort_by(|&a, &b| {
            records[b]
                .work(params.pass_count)
                .cmp(&records[a].work(params.pass_count))
                .then_with(|| records[a].id.cmp(&records[b].id))
        });

        let mut w = BufWriter::new(std::fs::File::create(&chunk_path)?);
        for &m in &members {
            let r = &records[m];
            writeln!(w, "{}\t{}\t{}\t{}", r.id, r.sense, r.antisense, r.seq)?;
            transcripts += 1;
        }
        w.flush()?;

        writeln!(
            manifest,
            "{}\t{}\t{}\t{}\t{}",
            chunk_no,
            chunk_prefix,
            file_name,
            members.len(),
            loads[bin],
        )?;
    }
    manifest.flush()?;

    Ok(SplitStats {
        chunks_written: non_empty.len(),
        transcripts,
        chunks_manifest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::File::create(&p)
            .unwrap()
            .write_all(content.as_bytes())
            .unwrap();
        p
    }

    #[test]
    fn distributes_all_transcripts_and_conserves_them() {
        let dir = TempDir::new().unwrap();
        // 5 transcripts, varied work.
        let content = "\
t1\t100\t0\tAAAAAAAAAA
t2\t1\t0\tACGT
t3\t50\t0\tACGTACGTAC
t4\t20\t0\tTTTT
t5\t5\t0\tGGGGGGGG
";
        let tf = write_file(dir.path(), "in.tsv", content);
        let params = SplitParams {
            transcript: tf,
            chunks: 3,
            pass_count: 2,
            outdir: dir.path().join("out"),
            prefix: "sim".into(),
        };
        let stats = run(&params).unwrap();
        assert_eq!(stats.transcripts, 5);
        assert!(stats.chunks_written <= 3 && stats.chunks_written >= 1);

        // Every input transcript appears exactly once across all chunk files.
        let mut ids = Vec::new();
        for entry in std::fs::read_dir(dir.path().join("out")).unwrap() {
            let p = entry.unwrap().path();
            if p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("chunk_")
            {
                for line in std::fs::read_to_string(&p).unwrap().lines() {
                    ids.push(line.split('\t').next().unwrap().to_string());
                }
            }
        }
        ids.sort();
        assert_eq!(ids, vec!["t1", "t2", "t3", "t4", "t5"]);
    }

    #[test]
    fn more_chunks_than_transcripts_caps_bins() {
        let dir = TempDir::new().unwrap();
        let tf = write_file(dir.path(), "in.tsv", "a\t1\t0\tAC\nb\t1\t0\tGT\n");
        let params = SplitParams {
            transcript: tf,
            chunks: 10,
            pass_count: 1,
            outdir: dir.path().join("out"),
            prefix: "sim".into(),
        };
        let stats = run(&params).unwrap();
        assert_eq!(stats.chunks_written, 2, "cannot exceed transcript count");
    }

    #[test]
    fn single_chunk_holds_everything() {
        let dir = TempDir::new().unwrap();
        let tf = write_file(
            dir.path(),
            "in.tsv",
            "a\t1\t0\tAC\nb\t2\t0\tGT\nc\t3\t0\tTT\n",
        );
        let params = SplitParams {
            transcript: tf,
            chunks: 1,
            pass_count: 1,
            outdir: dir.path().join("out"),
            prefix: "sim".into(),
        };
        let stats = run(&params).unwrap();
        assert_eq!(stats.chunks_written, 1);
        assert_eq!(stats.transcripts, 3);
    }
}
