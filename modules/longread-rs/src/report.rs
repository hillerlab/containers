//! Output writers for the manifest, depth tables, and `stats.json` (spec §4.11).

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use serde::Serialize;

use crate::error::Result;
use crate::expression::{GeneDepth, IsoformDepth};
use crate::model::{Provenance, Transcript};

/// The manifest header row.
const MANIFEST_HEADER: &str = "transcript_id\tgene_id\tparent_transcript_id\tevent_type\tis_original\tis_generated\tis_fusion\tpartner_gene_5p\tpartner_transcript_5p\tpartner_gene_3p\tpartner_transcript_3p\tchromosome\tstrand\texon_count\ttranscript_length\tseed";

/// Write `${prefix}.manifest.tsv`.
pub fn write_manifest<P: AsRef<Path>>(path: P, transcripts: &[Transcript]) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "{MANIFEST_HEADER}")?;
    for t in transcripts {
        let (is_original, is_generated, is_fusion) = match t.provenance {
            Provenance::Original => (true, false, false),
            Provenance::Generated(_) => (false, true, false),
            Provenance::Fusion => (false, false, true),
        };
        let event_type = match t.provenance {
            Provenance::Original => ".".to_string(),
            Provenance::Generated(e) => e.label().to_string(),
            Provenance::Fusion => "fusion".to_string(),
        };
        let parent = t.parent_id.as_deref().unwrap_or(".");
        let (pg5, pt5, pg3, pt3) = match &t.fusion {
            Some(f) => (
                f.gene_5p.as_str(),
                f.transcript_5p.as_str(),
                f.gene_3p.as_str(),
                f.transcript_3p.as_str(),
            ),
            None => (".", ".", ".", "."),
        };
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            t.id,
            t.gene_id,
            parent,
            event_type,
            is_original,
            is_generated,
            is_fusion,
            pg5,
            pt5,
            pg3,
            pt3,
            t.chrom,
            t.strand.as_char(),
            t.exon_count(),
            t.transcript_length(),
            t.seed,
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Write `${prefix}.gene_depth.tsv`.
pub fn write_gene_depth<P: AsRef<Path>>(path: P, depths: &[GeneDepth]) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "GENE_ID\tGENE_DEPTH\tRAW_WEIGHT\tIS_FUSION_GENE")?;
    for d in depths {
        writeln!(
            w,
            "{}\t{}\t{:.10}\t{}",
            d.gene_id, d.depth, d.raw_weight, d.is_fusion
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Write `${prefix}.isoform_depth.tsv`.
pub fn write_isoform_depth<P: AsRef<Path>>(path: P, depths: &[IsoformDepth]) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        w,
        "TRANSCRIPT_ID\tGENE_ID\tISOFORM_DEPTH\tANTISENSE_DEPTH\tRELATIVE_WEIGHT"
    )?;
    for d in depths {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{:.10}",
            d.transcript_id, d.gene_id, d.depth, d.antisense_depth, d.relative_weight
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Machine-readable run statistics (`stats.json`, spec §4.11).
#[derive(Debug, Clone, Serialize)]
pub struct Stats {
    /// Number of input transcripts.
    pub input_transcripts: usize,
    /// Number of input genes.
    pub input_genes: usize,
    /// Number of output transcripts.
    pub output_transcripts: usize,
    /// Number of ordinary (non-fusion) genes in the output.
    pub ordinary_genes: usize,
    /// Number of fusion pseudo-genes in the output.
    pub fusion_pseudo_genes: usize,
    /// Generated events by type (label → count).
    pub generated_events_by_type: BTreeMap<String, u64>,
    /// Requested (selected) events by type (label → count).
    pub requested_events_by_type: BTreeMap<String, u64>,
    /// Number of event applications that failed.
    pub failed_event_attempts: u64,
    /// Number of duplicate structures rejected.
    pub duplicate_structures_rejected: u64,
    /// Genes with no eligible parent for a given event type (label → count).
    pub ineligible_genes_by_event_type: BTreeMap<String, u64>,
    /// Number of requested fusions.
    pub requested_fusions: u32,
    /// Number of successful fusions.
    pub successful_fusions: usize,
    /// Fusion failures by reason.
    pub fusion_failures_by_reason: BTreeMap<String, u64>,
    /// Total requested molecules.
    pub total_requested_molecules: u64,
    /// Total allocated gene molecules.
    pub total_allocated_gene_molecules: u64,
    /// Total allocated isoform molecules.
    pub total_allocated_isoform_molecules: u64,
    /// Number of genes with zero molecules.
    pub zero_count_genes: usize,
    /// Number of isoforms with zero molecules.
    pub zero_count_isoforms: usize,
}

/// Write `${prefix}.stats.json` (pretty-printed for readability).
pub fn write_stats<P: AsRef<Path>>(path: P, stats: &Stats) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    let json = serde_json::to_string_pretty(stats)?;
    w.write_all(json.as_bytes())?;
    w.write_all(b"\n")?;
    w.flush()?;
    Ok(())
}
