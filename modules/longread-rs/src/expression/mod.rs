//! Expression assignment: gene-level GMM depths and within-gene isoform allocation
//! (spec §4.9, §4.10). Guarantees the exact-conservation invariants:
//!
//! ```text
//! sum(gene_depth)                     == total_molecules
//! sum(isoform_depth for gene G)       == gene_depth[G]
//! ```

pub mod alloc;
pub mod gmm;
pub mod zipf;

use std::collections::BTreeMap;

use crate::model::{Provenance, Transcript};

use gmm::GeneGmm;

/// Per-gene depth record.
#[derive(Debug, Clone)]
pub struct GeneDepth {
    /// Gene identifier.
    pub gene_id: String,
    /// Allocated integer molecule count.
    pub depth: u64,
    /// Raw (pre-fusion-scale) GMM weight.
    pub raw_weight: f64,
    /// Whether this is a fusion pseudo-gene.
    pub is_fusion: bool,
}

/// Per-isoform depth record.
#[derive(Debug, Clone)]
pub struct IsoformDepth {
    /// Transcript identifier.
    pub transcript_id: String,
    /// Owning gene identifier.
    pub gene_id: String,
    /// Allocated sense molecule count.
    pub depth: u64,
    /// Antisense molecule count (0 in v1).
    pub antisense_depth: u64,
    /// Normalized within-gene relative weight.
    pub relative_weight: f64,
}

/// Parameters controlling expression assignment.
#[derive(Debug, Clone, Copy)]
pub struct ExpressionParams {
    /// Global seed.
    pub global_seed: u64,
    /// Total molecule budget.
    pub total_molecules: u64,
    /// Zipf coefficient.
    pub alpha: f64,
    /// Multiplier applied to fusion pseudo-gene raw weights.
    pub fusion_expression_scale: f64,
}

/// The result of expression assignment.
#[derive(Debug, Clone)]
pub struct ExpressionResult {
    /// Gene depths, sorted by gene id.
    pub gene_depths: Vec<GeneDepth>,
    /// Isoform depths, sorted by (gene id, transcript id).
    pub isoform_depths: Vec<IsoformDepth>,
    /// Number of genes allocated zero molecules.
    pub zero_count_genes: usize,
    /// Number of isoforms allocated zero molecules.
    pub zero_count_isoforms: usize,
    /// Total molecules allocated at gene level (should equal `total_molecules`).
    pub total_allocated_gene: u64,
    /// Total molecules allocated at isoform level (should equal `total_allocated_gene`).
    pub total_allocated_isoform: u64,
}

/// Assign expression across the final transcript set.
pub fn assign(transcripts: &[Transcript], params: &ExpressionParams) -> ExpressionResult {
    // Group transcripts by gene, preserving a deterministic (id-sorted) order.
    let mut gene_isoforms: BTreeMap<String, Vec<&Transcript>> = BTreeMap::new();
    let mut gene_is_fusion: BTreeMap<String, bool> = BTreeMap::new();
    for t in transcripts {
        gene_isoforms.entry(t.gene_id.clone()).or_default().push(t);
        let is_fusion = matches!(t.provenance, Provenance::Fusion);
        let entry = gene_is_fusion.entry(t.gene_id.clone()).or_insert(false);
        *entry = *entry || is_fusion;
    }

    let gene_ids: Vec<String> = gene_isoforms.keys().cloned().collect();
    let gmm = GeneGmm::default();

    // Gene-level raw and adjusted weights.
    let mut raw_weights = Vec::with_capacity(gene_ids.len());
    let mut adjusted_weights = Vec::with_capacity(gene_ids.len());
    for gid in &gene_ids {
        let raw = gmm.raw_weight(params.global_seed, gid);
        raw_weights.push(raw);
        let adj = if gene_is_fusion[gid] {
            raw * params.fusion_expression_scale
        } else {
            raw
        };
        adjusted_weights.push(adj);
    }

    let gene_counts =
        alloc::largest_remainder(&adjusted_weights, params.total_molecules, &gene_ids);

    let mut gene_depths = Vec::with_capacity(gene_ids.len());
    let mut isoform_depths = Vec::new();
    let mut zero_count_genes = 0usize;
    let mut zero_count_isoforms = 0usize;
    let mut total_allocated_gene = 0u64;
    let mut total_allocated_isoform = 0u64;

    for (gi, gid) in gene_ids.iter().enumerate() {
        let gene_count = gene_counts[gi];
        total_allocated_gene += gene_count;
        if gene_count == 0 {
            zero_count_genes += 1;
        }
        gene_depths.push(GeneDepth {
            gene_id: gid.clone(),
            depth: gene_count,
            raw_weight: raw_weights[gi],
            is_fusion: gene_is_fusion[gid],
        });

        // Isoforms sorted by transcript id for a deterministic rank assignment.
        let mut isoforms = gene_isoforms[gid].clone();
        isoforms.sort_by(|a, b| a.id.cmp(&b.id));
        let ids: Vec<String> = isoforms.iter().map(|t| t.id.clone()).collect();
        let rel = zipf::isoform_weights(isoforms.len(), params.alpha, params.global_seed, gid);
        let counts = alloc::largest_remainder(&rel, gene_count, &ids);

        for (k, t) in isoforms.iter().enumerate() {
            let depth = counts[k];
            total_allocated_isoform += depth;
            if depth == 0 {
                zero_count_isoforms += 1;
            }
            isoform_depths.push(IsoformDepth {
                transcript_id: t.id.clone(),
                gene_id: gid.clone(),
                depth,
                antisense_depth: 0,
                relative_weight: rel.get(k).copied().unwrap_or(0.0),
            });
        }
    }

    ExpressionResult {
        gene_depths,
        isoform_depths,
        zero_count_genes,
        zero_count_isoforms,
        total_allocated_gene,
        total_allocated_isoform,
    }
}
