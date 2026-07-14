//! Input validation (spec §4.3) and construction of the validated gene model.
//!
//! All problems are accumulated and reported together, so a malformed input yields one
//! comprehensive error rather than failing on the first issue. Nothing is silently skipped
//! (execution rule #15).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use genepred::GenePred;

use crate::error::{Error, Result};
use crate::io;
use crate::model::{Provenance, Strand, Transcript};

/// A validated gene: all of its original isoforms share a chromosome and strand.
#[derive(Debug, Clone)]
pub struct GeneModel {
    /// Gene identifier.
    pub gene_id: String,
    /// Chromosome.
    pub chrom: String,
    /// Strand shared by all isoforms.
    pub strand: Strand,
    /// Original isoforms (canonical order).
    pub transcripts: Vec<Transcript>,
}

impl GeneModel {
    /// Gene span `[min exon start, max exon end]` over the original isoforms.
    pub fn span(&self) -> (u64, u64) {
        let start = self
            .transcripts
            .iter()
            .map(|t| t.chrom_start())
            .min()
            .unwrap_or(0);
        let end = self
            .transcripts
            .iter()
            .map(|t| t.chrom_end())
            .max()
            .unwrap_or(0);
        (start, end)
    }
}

/// The fully validated input ready for simulation.
#[derive(Debug, Clone)]
pub struct ValidatedInput {
    /// Genes in deterministic (gene_id) order.
    pub genes: Vec<GeneModel>,
    /// Optional chromosome sizes (present iff `--chrom-sizes` was supplied).
    pub chrom_sizes: Option<HashMap<String, u64>>,
    /// Total number of input transcripts.
    pub input_transcript_count: usize,
}

/// Extract absolute exon intervals from a parsed BED12 record, in file order.
fn extract_exons(g: &GenePred) -> Vec<(u64, u64)> {
    match (g.block_starts(), g.block_ends()) {
        (Some(starts), Some(ends)) => starts.iter().copied().zip(ends.iter().copied()).collect(),
        _ => vec![(g.start(), g.end())],
    }
}

/// Load, validate, and assemble the input (spec §4.3).
pub fn load_and_validate<P: AsRef<Path>>(
    bed_path: P,
    mapping_path: P,
    chrom_sizes_path: Option<P>,
    global_seed: u64,
    min_transcript_length: u64,
) -> Result<ValidatedInput> {
    let records = io::read_bed_records(&bed_path)?;
    let mapping = io::read_transcript_gene(&mapping_path)?;
    let chrom_sizes = match &chrom_sizes_path {
        Some(p) => Some(io::read_chrom_sizes(p)?),
        None => None,
    };

    let mut errors: Vec<String> = Vec::new();
    let mut warns: Vec<String> = Vec::new();
    let mut must_ignore = HashSet::new();

    // Rule 2: unique transcript name across BED.
    let mut name_counts: HashMap<String, usize> = HashMap::new();
    let mut record_names: Vec<Option<String>> = Vec::with_capacity(records.len());
    for g in &records {
        let name = g
            .name()
            .and_then(|n| std::str::from_utf8(n).ok())
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty());
        if let Some(n) = &name {
            *name_counts.entry(n.clone()).or_insert(0) += 1;
        }
        record_names.push(name);
    }

    for (idx, g) in records.iter().enumerate() {
        let lineno = idx + 1;
        match &record_names[idx] {
            None => errors.push(format!("BED record {lineno}: missing transcript name")),
            Some(n) if name_counts.get(n).copied().unwrap_or(0) > 1 => {
                errors.push(format!(
                    "ERROR: BED record {lineno}: duplicate transcript name '{n}'"
                ));
            }
            _ => {}
        }

        // Rule 3: strand must be + or -.
        let strand = g.strand().and_then(Strand::from_genepred);
        if strand.is_none() {
            errors.push(format!(
                "ERROR: BED record {lineno} ('{}'): strand must be '+' or '-'",
                record_names[idx].as_deref().unwrap_or("?")
            ));
        }

        // Rule 4: sorted, positive, non-overlapping blocks.
        let exons = extract_exons(g);
        let probe = Transcript {
            id: record_names[idx].clone().unwrap_or_default(),
            gene_id: String::new(),
            chrom: String::from_utf8_lossy(g.chrom()).into_owned(),
            strand: strand.unwrap_or(Strand::Plus),
            exons: exons.clone(),
            thick_start: g.thick_start().unwrap_or_else(|| g.start()),
            thick_end: g.thick_end().unwrap_or_else(|| g.end()),
            provenance: Provenance::Original,
            parent_id: None,
            fusion: None,
            coding_defined: false,
            seed: global_seed,
        };

        if let Err(e) = probe.validate_blocks() {
            errors.push(format!(
                "ERROR: BED record {lineno} ('{}'): {e}",
                record_names[idx].as_deref().unwrap_or("?")
            ));
        }

        // Rule 4.1: Ensure transcript exonic lenght > min_transcript_length.
        // If transcript does not comply, exclude it from the validate input and
        // report as warning
        let exonic_length = g.exonic_length();
        if exonic_length < min_transcript_length {
            warns.push(format!(
                "BED record {lineno} ('{}'): exonic length ({exonic_length}) is less than minimum length ({min_transcript_length})",
                record_names[idx].as_deref().unwrap_or("?")
            ));

            must_ignore.insert(idx);
        }

        // Rule 5: within chromosome bounds (only when sizes are supplied).
        if let Some(sizes) = &chrom_sizes {
            let chrom = String::from_utf8_lossy(g.chrom()).into_owned();
            match sizes.get(&chrom) {
                None => errors.push(format!(
                    "ERROR: BED record {lineno} ('{}'): chromosome '{chrom}' absent from chrom.sizes",
                    record_names[idx].as_deref().unwrap_or("?")
                )),
                Some(&size) => {
                    if !exons.is_empty() {
                        let max_end = exons.iter().map(|e| e.1).max().unwrap_or(0);
                        if max_end > size {
                            errors.push(format!(
                                "ERROR: BED record {lineno} ('{}'): end {max_end} exceeds chromosome '{chrom}' size {size}",
                                record_names[idx].as_deref().unwrap_or("?")
                            ));
                        }
                    }
                }
            }
        }
    }

    // Mapping consistency (rules 6, 7, 8).
    let mut mapping_by_tx: BTreeMap<String, Vec<(String, usize)>> = BTreeMap::new();
    for m in &mapping {
        mapping_by_tx
            .entry(m.transcript.clone())
            .or_default()
            .push((m.gene.clone(), m.line));
    }

    // Rule 7: a transcript mapped more than once (to any gene) is rejected.
    for (tx, genes) in &mapping_by_tx {
        if genes.len() > 1 {
            let detail: Vec<String> = genes.iter().map(|(g, l)| format!("{g}@line {l}")).collect();
            errors.push(format!(
                "ERROR: transcript '{tx}' has multiple mapping entries: {}",
                detail.join(", ")
            ));
        }
    }

    let bed_name_set: HashSet<&str> = record_names.iter().filter_map(|n| n.as_deref()).collect();

    // Rule 6: every BED transcript must be mapped.
    for name in &bed_name_set {
        if !mapping_by_tx.contains_key(*name) {
            errors.push(format!(
                "ERROR: BED transcript '{name}' has no transcript→gene mapping"
            ));
        }
    }

    // Resolve a single gene per transcript (only meaningful for singletons).
    let tx_gene: HashMap<String, String> = mapping_by_tx
        .iter()
        .filter(|(_, genes)| genes.len() == 1)
        .map(|(tx, genes)| (tx.clone(), genes[0].0.clone()))
        .collect();

    // Rule 9: all transcripts of a gene share chromosome and strand.
    // Group valid (named, mapped, +/- strand) records by gene.
    let mut gene_members: BTreeMap<String, Vec<(String, Strand)>> = BTreeMap::new();
    for (idx, g) in records.iter().enumerate() {
        let Some(name) = &record_names[idx] else {
            continue;
        };
        let Some(strand) = g.strand().and_then(Strand::from_genepred) else {
            continue;
        };
        let Some(gene) = tx_gene.get(name) else {
            continue;
        };
        let chrom = String::from_utf8_lossy(g.chrom()).into_owned();
        gene_members
            .entry(gene.clone())
            .or_default()
            .push((format!("{chrom}\t{}", strand.as_char()), strand));
    }

    for (gene, members) in &gene_members {
        let distinct: HashSet<&String> = members.iter().map(|(k, _)| k).collect();
        if distinct.len() > 1 {
            let mut labels: Vec<String> = distinct.iter().map(|k| k.replace('\t', ":")).collect();
            labels.sort();
            errors.push(format!(
                "ERROR: gene '{gene}' spans multiple chromosome/strand combinations: {}",
                labels.join(", ")
            ));
        }
    }

    // Rule 10: duplicate genomic transcript structures among originals -> WARNING not ERROR
    {
        let mut seen: HashSet<(String, char, Vec<u64>, Vec<u64>)> = HashSet::new();
        for (idx, g) in records.iter().enumerate() {
            let Some(strand) = g.strand().and_then(Strand::from_genepred) else {
                continue;
            };
            let exons = extract_exons(g);
            if exons.is_empty() {
                continue;
            }
            let chrom = String::from_utf8_lossy(g.chrom()).into_owned();
            let starts: Vec<u64> = exons.iter().map(|e| e.0).collect();
            let ends: Vec<u64> = exons.iter().map(|e| e.1).collect();
            let key = (chrom, strand.as_char(), starts, ends);
            if !seen.insert(key) {
                // WARN: duplicate genomic transcript structures are not errors
                // we ignore them for now

                let warn = format!(
                    "BED record {} ('{}'): duplicate genomic transcript structure",
                    idx + 1,
                    record_names[idx].as_deref().unwrap_or("?")
                );
                warns.push(warn);
            }
        }
    }

    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(Error::Validation(errors.join("\n")));
    }

    if !warns.is_empty() {
        warns.sort();
        warns.dedup();

        for warn in warns {
            println!("WARN:  {}", warn);
        }
    }

    // All checks passed — assemble the validated model.
    let mut genes_map: BTreeMap<String, GeneModel> = BTreeMap::new();
    for (idx, g) in records.iter().enumerate() {
        if must_ignore.contains(&idx) {
            continue;
        }

        let name = record_names[idx].clone().expect("validated: name present");
        let strand = g
            .strand()
            .and_then(Strand::from_genepred)
            .expect("validated: strand +/-");
        let gene = tx_gene.get(&name).expect("validated: single gene").clone();
        let chrom = String::from_utf8_lossy(g.chrom()).into_owned();
        let exons = extract_exons(g);
        let thick_start = g.thick_start().unwrap_or_else(|| g.start());
        let thick_end = g.thick_end().unwrap_or_else(|| g.end());
        let transcript = Transcript {
            id: name,
            gene_id: gene.clone(),
            chrom: chrom.clone(),
            strand,
            exons,
            thick_start,
            thick_end,
            provenance: Provenance::Original,
            parent_id: None,
            fusion: None,
            coding_defined: thick_start < thick_end,
            seed: global_seed,
        };
        let entry = genes_map.entry(gene.clone()).or_insert_with(|| GeneModel {
            gene_id: gene,
            chrom,
            strand,
            transcripts: Vec::new(),
        });
        entry.transcripts.push(transcript);
    }

    let mut genes: Vec<GeneModel> = genes_map.into_values().collect();
    for gene in &mut genes {
        crate::model::canonical_sort(&mut gene.transcripts);
    }
    genes.sort_by(|a, b| a.gene_id.cmp(&b.gene_id));

    let input_transcript_count = records.len();
    Ok(ValidatedInput {
        genes,
        chrom_sizes,
        input_transcript_count,
    })
}
