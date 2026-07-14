//! Two-gene fusion construction (spec §4.6).
//!
//! A fusion joins one complete original isoform from each of two different genes on the same
//! chromosome and strand, with non-overlapping spans and a strictly positive genomic gap. The
//! candidate space of `(ordered pair, iso5, iso3)` combinations is enumerated deterministically
//! and shuffled with a fusion-order seed; fusion `i` takes candidate `i`, which guarantees a
//! unique combination without rejection sampling.

use std::collections::{BTreeMap, HashMap, HashSet};

use rand::seq::SliceRandom;

use crate::model::{FusionInfo, Provenance, Strand, StructuralKey, Transcript};
use crate::rng::{rng_from_seed, stable_seed, stable_seed_ordinal};
use crate::validate::GeneModel;

/// Parameters controlling fusion generation.
#[derive(Debug, Clone, Copy)]
pub struct FusionParams {
    /// Global seed.
    pub global_seed: u64,
    /// Requested number of unique fusion transcripts.
    pub fusion_count: u32,
    /// Minimum genomic intron between the two partner isoforms.
    pub min_fusion_intron: u64,
    /// Maximum genomic distance between partner gene spans (`None` = unlimited).
    pub max_fusion_distance: Option<u64>,
    /// Minimum transcript (exonic) length for the fused transcript.
    pub min_transcript_length: u64,
}

/// The result of fusion generation.
#[derive(Debug, Default, Clone)]
pub struct FusionResult {
    /// Successfully generated fusion transcripts.
    pub fusions: Vec<Transcript>,
    /// Number of fusions requested.
    pub requested: u32,
    /// Number of eligible candidate combinations available.
    pub eligible_candidates: usize,
    /// Failures by reason (e.g. `"min_length"`, `"duplicate_structure"`).
    pub failures_by_reason: BTreeMap<String, u64>,
}

impl FusionResult {
    fn note_failure(&mut self, reason: &str) {
        *self
            .failures_by_reason
            .entry(reason.to_string())
            .or_insert(0) += 1;
    }
}

/// A resolved candidate: gene indices and isoform indices within those genes.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    g5: usize,
    g3: usize,
    iso5: usize,
    iso3: usize,
}

/// Enumerate eligible `(ordered pair, iso5, iso3)` candidates in deterministic order.
fn enumerate_candidates(genes: &[GeneModel], params: &FusionParams) -> Vec<Candidate> {
    // Group gene indices by (chrom, strand).
    let mut groups: BTreeMap<(String, char), Vec<usize>> = BTreeMap::new();
    for (idx, g) in genes.iter().enumerate() {
        // A gene with no original isoforms cannot be a fusion partner.
        if g.transcripts.is_empty() {
            continue;
        }
        groups
            .entry((g.chrom.clone(), g.strand.as_char()))
            .or_default()
            .push(idx);
    }

    let mut candidates = Vec::new();
    for indices in groups.values() {
        // Sort within group by span start, then gene id, for a stable enumeration.
        let mut sorted = indices.clone();
        sorted.sort_by(|&a, &b| {
            genes[a]
                .span()
                .0
                .cmp(&genes[b].span().0)
                .then(genes[a].gene_id.cmp(&genes[b].gene_id))
        });

        for ai in 0..sorted.len() {
            for bi in (ai + 1)..sorted.len() {
                let lower = sorted[ai];
                let higher = sorted[bi];
                let (_ls, le) = genes[lower].span();
                let (hs, _he) = genes[higher].span();
                // Non-overlapping spans (half-open): lower must end at/before higher starts.
                if le > hs {
                    continue;
                }
                let span_gap = hs - le;
                if let Some(md) = params.max_fusion_distance {
                    if span_gap > md {
                        continue;
                    }
                }
                // 5′/3′ assignment by strand (shared across the group).
                let (g5, g3) = match genes[lower].strand {
                    Strand::Plus => (lower, higher),
                    Strand::Minus => (higher, lower),
                };
                for iso5 in 0..genes[g5].transcripts.len() {
                    for iso3 in 0..genes[g3].transcripts.len() {
                        // Genomic gap between the two selected isoforms.
                        let t5 = &genes[g5].transcripts[iso5];
                        let t3 = &genes[g3].transcripts[iso3];
                        let (lo, hi) = if t5.chrom_start() <= t3.chrom_start() {
                            (t5, t3)
                        } else {
                            (t3, t5)
                        };
                        if hi.chrom_start() < lo.chrom_end() {
                            continue; // isoform spans overlap
                        }
                        let iso_gap = hi.chrom_start() - lo.chrom_end();
                        if iso_gap < params.min_fusion_intron.max(1) {
                            continue;
                        }
                        candidates.push(Candidate { g5, g3, iso5, iso3 });
                    }
                }
            }
        }
    }
    candidates
}

/// Assemble the combined, sorted exon list for a fusion candidate.
fn fusion_exons(genes: &[GeneModel], cand: &Candidate) -> Vec<(u64, u64)> {
    let t5 = &genes[cand.g5].transcripts[cand.iso5];
    let t3 = &genes[cand.g3].transcripts[cand.iso3];
    let mut exons: Vec<(u64, u64)> = Vec::with_capacity(t5.exons.len() + t3.exons.len());
    exons.extend_from_slice(&t5.exons);
    exons.extend_from_slice(&t3.exons);
    exons.sort_by_key(|e| e.0);
    exons
}

/// Generate fusion transcripts (spec §4.6).
pub fn generate_fusions(genes: &[GeneModel], params: &FusionParams) -> FusionResult {
    let mut result = FusionResult {
        requested: params.fusion_count,
        ..Default::default()
    };
    if params.fusion_count == 0 {
        return result;
    }

    let mut candidates = enumerate_candidates(genes, params);
    result.eligible_candidates = candidates.len();
    if candidates.is_empty() {
        return result;
    }

    // Deterministic shuffle.
    let order_seed = stable_seed(params.global_seed, "fusion-order", b"");
    let mut order_rng = rng_from_seed(order_seed);
    candidates.shuffle(&mut order_rng);

    let mut seen: HashSet<StructuralKey> = HashSet::new();
    let mut ordinal_by_pair: HashMap<(String, String), u32> = HashMap::new();

    for cand in &candidates {
        if result.fusions.len() as u32 >= params.fusion_count {
            break;
        }
        let g5 = &genes[cand.g5];
        let g3 = &genes[cand.g3];
        let exons = fusion_exons(genes, cand);

        // Structural safety checks (independent of id/ordinal).
        let exonic_len: u64 = exons.iter().map(|(s, e)| e - s).sum();
        let starts: Vec<u64> = exons.iter().map(|e| e.0).collect();
        let ends: Vec<u64> = exons.iter().map(|e| e.1).collect();
        // Non-overlapping / positive introns.
        let blocks_valid =
            exons.windows(2).all(|w| w[0].1 < w[1].0) && exons.iter().all(|(s, e)| e > s);
        if !blocks_valid {
            result.note_failure("invalid_blocks");
            continue;
        }
        if exonic_len < params.min_transcript_length {
            result.note_failure("min_length");
            continue;
        }
        let key: StructuralKey = (g5.chrom.clone(), g5.strand, starts, ends);
        if !seen.insert(key) {
            result.note_failure("duplicate_structure");
            continue;
        }

        // Accepted: assign a per-pseudo-gene ordinal and build the transcript.
        let ord = {
            let e = ordinal_by_pair
                .entry((g5.gene_id.clone(), g3.gene_id.clone()))
                .or_insert(0);
            *e += 1;
            *e
        };
        let seed = stable_seed_ordinal(params.global_seed, "fusion", result.fusions.len() as u64);
        let chrom_start = exons.first().map(|e| e.0).unwrap_or(0);
        let t5 = &g5.transcripts[cand.iso5];
        let t3 = &g3.transcripts[cand.iso3];
        result.fusions.push(Transcript {
            id: format!("FUSION::{}::{}::{}", g5.gene_id, g3.gene_id, ord),
            gene_id: format!("FUSION_GENE::{}::{}", g5.gene_id, g3.gene_id),
            chrom: g5.chrom.clone(),
            strand: g5.strand,
            exons,
            thick_start: chrom_start,
            thick_end: chrom_start,
            provenance: Provenance::Fusion,
            parent_id: None,
            fusion: Some(FusionInfo {
                gene_5p: g5.gene_id.clone(),
                transcript_5p: t5.id.clone(),
                gene_3p: g3.gene_id.clone(),
                transcript_3p: t3.id.clone(),
            }),
            coding_defined: false,
            seed,
        });
    }

    result
}
