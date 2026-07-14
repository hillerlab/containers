//! The `prepare` command orchestration (spec §4.2): validate → generate isoforms → build
//! fusions → assign expression → write outputs, with a global deterministic dedup pass.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use rayon::ThreadPoolBuilder;

use crate::error::{Error, Result};
use crate::events;
use crate::expression::{self, ExpressionParams};
use crate::fusion::{self, FusionParams};
use crate::generate::{self, EventWeights, GenerationParams, NewIsoformsMode};
use crate::model::{canonical_sort, EventType, Provenance, StructuralKey, Transcript};
use crate::report::{self, Stats};
use crate::{io, validate};

/// All parameters for a `prepare` run.
#[derive(Debug, Clone)]
pub struct PrepareParams {
    /// Input BED12 path.
    pub bed: PathBuf,
    /// Input transcript→gene TSV path.
    pub transcript_gene: PathBuf,
    /// Optional chrom.sizes path.
    pub chrom_sizes: Option<PathBuf>,
    /// Output prefix (may include directories).
    pub output_prefix: String,
    /// Global RNG seed.
    pub seed: u64,
    /// Thread count (0 = all available).
    pub threads: usize,
    /// New-isoform count mode.
    pub new_isoforms_mode: NewIsoformsMode,
    /// Event selection weights.
    pub event_weights: EventWeights,
    /// Max attempts per requested isoform.
    pub max_event_attempts: u32,
    /// Minimum transcript (exonic) length.
    pub min_transcript_length: u64,
    /// Requested fusion count.
    pub fusion_count: u32,
    /// Minimum fusion intron.
    pub min_fusion_intron: u64,
    /// Maximum fusion distance (`None` = unlimited).
    pub max_fusion_distance: Option<u64>,
    /// Fusion expression scale.
    pub fusion_expression_scale: f64,
    /// Total molecule budget.
    pub total_molecules: u64,
    /// Zipf coefficient.
    pub alpha: f64,
    /// Fail if the requested event/fusion counts are not met exactly.
    pub require_exact_event_count: bool,
}

/// Paths of the files written by a `prepare` run.
#[derive(Debug, Clone)]
pub struct PrepareOutputs {
    /// `${prefix}.isoforms.bed`
    pub isoforms_bed: PathBuf,
    /// `${prefix}.transcript_gene.tsv`
    pub transcript_gene: PathBuf,
    /// `${prefix}.gene_depth.tsv`
    pub gene_depth: PathBuf,
    /// `${prefix}.isoform_depth.tsv`
    pub isoform_depth: PathBuf,
    /// `${prefix}.manifest.tsv`
    pub manifest: PathBuf,
    /// `${prefix}.stats.json`
    pub stats: PathBuf,
}

impl PrepareOutputs {
    fn from_prefix(prefix: &str) -> Self {
        let p = |suffix: &str| PathBuf::from(format!("{prefix}.{suffix}"));
        PrepareOutputs {
            isoforms_bed: p("isoforms.bed"),
            transcript_gene: p("transcript_gene.tsv"),
            gene_depth: p("gene_depth.tsv"),
            isoform_depth: p("isoform_depth.tsv"),
            manifest: p("manifest.tsv"),
            stats: p("stats.json"),
        }
    }
}

/// Run `prepare`, returning the statistics and the list of output paths.
pub fn run(params: &PrepareParams) -> Result<(Stats, PrepareOutputs)> {
    // 1. Validate and assemble the input model.
    let input = validate::load_and_validate(
        &params.bed,
        &params.transcript_gene,
        params.chrom_sizes.as_ref(),
        params.seed,
        params.min_transcript_length,
    )?;

    // 2. Generate synthetic isoforms per gene (deterministic in thread count).
    let gp = GenerationParams {
        global_seed: params.seed,
        mode: params.new_isoforms_mode,
        weights: params.event_weights,
        max_event_attempts: params.max_event_attempts,
        min_transcript_length: params.min_transcript_length,
    };
    let threads = if params.threads == 0 {
        rayon::current_num_threads().max(1)
    } else {
        params.threads
    };
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| Error::config(format!("failed to build thread pool: {e}")))?;
    let outcomes = pool.install(|| generate::generate_all(&input.genes, &gp));

    // 3. Build fusions.
    let fp = FusionParams {
        global_seed: params.seed,
        fusion_count: params.fusion_count,
        min_fusion_intron: params.min_fusion_intron,
        max_fusion_distance: params.max_fusion_distance,
        min_transcript_length: params.min_transcript_length,
    };
    let fusion_result = fusion::generate_fusions(&input.genes, &fp);

    // 4. Assemble the final transcript set with a global deterministic dedup pass.
    let mut seen: HashSet<StructuralKey> = HashSet::new();
    let mut final_transcripts: Vec<Transcript> = Vec::new();
    let mut duplicate_structures_rejected: u64 = 0;

    // Originals are always kept (validation guarantees they are unique).
    for gene in &input.genes {
        for t in &gene.transcripts {
            seen.insert(t.structural_key());
            final_transcripts.push(t.clone());
        }
    }
    // Local dedup rejections recorded during generation.
    for o in &outcomes {
        duplicate_structures_rejected += o.local_duplicates;
    }
    // Generated transcripts, canonical order, dropping cross-gene structural collisions.
    let mut generated: Vec<Transcript> =
        outcomes.iter().flat_map(|o| o.generated.clone()).collect();
    canonical_sort(&mut generated);
    for t in generated {
        if seen.insert(t.structural_key()) {
            final_transcripts.push(t);
        } else {
            duplicate_structures_rejected += 1;
        }
    }
    // Fusions, canonical order, dropping any structural collisions.
    let mut fusions = fusion_result.fusions.clone();
    canonical_sort(&mut fusions);
    for t in fusions {
        if seen.insert(t.structural_key()) {
            final_transcripts.push(t);
        } else {
            duplicate_structures_rejected += 1;
        }
    }

    canonical_sort(&mut final_transcripts);

    // Guard: transcript IDs must be globally unique.
    check_unique_ids(&final_transcripts)?;

    // 5. Enforce exact-count requests if asked.
    if params.require_exact_event_count {
        let requested_isoforms: u64 = outcomes.iter().map(|o| o.requested_isoforms as u64).sum();
        let generated_isoforms: u64 = final_transcripts
            .iter()
            .filter(|t| matches!(t.provenance, Provenance::Generated(_)))
            .count() as u64;
        if generated_isoforms < requested_isoforms {
            return Err(Error::config(format!(
                "--require-exact-event-count: requested {requested_isoforms} synthetic isoforms but only {generated_isoforms} were generated"
            )));
        }
        let successful_fusions = final_transcripts
            .iter()
            .filter(|t| matches!(t.provenance, Provenance::Fusion))
            .count() as u32;
        if successful_fusions < params.fusion_count {
            return Err(Error::config(format!(
                "--require-exact-event-count: requested {} fusions but only {successful_fusions} were generated",
                params.fusion_count
            )));
        }
    }

    // 6. Assign expression.
    let ep = ExpressionParams {
        global_seed: params.seed,
        total_molecules: params.total_molecules,
        alpha: params.alpha,
        fusion_expression_scale: params.fusion_expression_scale,
    };
    let expr = expression::assign(&final_transcripts, &ep);

    // Conservation invariants (spec §2.4) — assert they hold before writing.
    debug_assert_eq!(expr.total_allocated_gene, params.total_molecules);
    debug_assert_eq!(expr.total_allocated_isoform, expr.total_allocated_gene);

    // 7. Build statistics.
    let stats = build_stats(
        &input,
        &outcomes,
        &fusion_result,
        &final_transcripts,
        &expr,
        params,
        duplicate_structures_rejected,
    );

    // 8. Write outputs.
    let outputs = PrepareOutputs::from_prefix(&params.output_prefix);
    if let Some(parent) = outputs.isoforms_bed.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    io::write_bed(&outputs.isoforms_bed, &final_transcripts)?;
    io::write_transcript_gene(&outputs.transcript_gene, &final_transcripts)?;
    report::write_gene_depth(&outputs.gene_depth, &expr.gene_depths)?;
    report::write_isoform_depth(&outputs.isoform_depth, &expr.isoform_depths)?;
    report::write_manifest(&outputs.manifest, &final_transcripts)?;
    report::write_stats(&outputs.stats, &stats)?;

    Ok((stats, outputs))
}

/// Ensure transcript IDs are globally unique.
fn check_unique_ids(transcripts: &[Transcript]) -> Result<()> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut dups: Vec<String> = Vec::new();
    for t in transcripts {
        if !seen.insert(t.id.as_str()) {
            dups.push(t.id.clone());
        }
    }
    if dups.is_empty() {
        Ok(())
    } else {
        dups.sort();
        dups.dedup();
        Err(Error::config(format!(
            "duplicate transcript identifiers in output (generated/fusion ids collided with input names): {}",
            dups.join(", ")
        )))
    }
}

/// Compute, for each event type, how many genes have no eligible original isoform.
fn ineligible_genes_by_type(input: &validate::ValidatedInput) -> BTreeMap<String, u64> {
    let mut map: BTreeMap<String, u64> = BTreeMap::new();
    for &e in &EventType::ALL {
        let count = input
            .genes
            .iter()
            .filter(|g| {
                !g.transcripts
                    .iter()
                    .any(|t| events::eligible(&t.exons, t.strand, e))
            })
            .count() as u64;
        map.insert(e.label().to_string(), count);
    }
    map
}

/// Build the run statistics.
#[allow(clippy::too_many_arguments)]
fn build_stats(
    input: &validate::ValidatedInput,
    outcomes: &[generate::GeneOutcome],
    fusion_result: &fusion::FusionResult,
    final_transcripts: &[Transcript],
    expr: &expression::ExpressionResult,
    params: &PrepareParams,
    duplicate_structures_rejected: u64,
) -> Stats {
    // Requested (selected) events by type.
    let mut requested_events: BTreeMap<String, u64> = BTreeMap::new();
    for &e in &EventType::ALL {
        requested_events.insert(e.label().to_string(), 0);
    }
    let mut failed_event_attempts = 0u64;
    for o in outcomes {
        failed_event_attempts += o.failed_attempts;
        for (i, &e) in EventType::ALL.iter().enumerate() {
            *requested_events.get_mut(e.label()).unwrap() += o.selections[i];
        }
    }

    // Generated events by type (from the final surviving set).
    let mut generated_events: BTreeMap<String, u64> = BTreeMap::new();
    for &e in &EventType::ALL {
        generated_events.insert(e.label().to_string(), 0);
    }
    let mut ordinary_gene_ids: HashSet<&str> = HashSet::new();
    let mut fusion_gene_ids: HashSet<&str> = HashSet::new();
    for t in final_transcripts {
        match t.provenance {
            Provenance::Generated(e) => {
                *generated_events.get_mut(e.label()).unwrap() += 1;
            }
            Provenance::Fusion => {}
            Provenance::Original => {}
        }
        if matches!(t.provenance, Provenance::Fusion) {
            fusion_gene_ids.insert(t.gene_id.as_str());
        } else {
            ordinary_gene_ids.insert(t.gene_id.as_str());
        }
    }

    let successful_fusions = final_transcripts
        .iter()
        .filter(|t| matches!(t.provenance, Provenance::Fusion))
        .count();

    // Fusion failures include structural cross-drops handled in the merge pass, but those are
    // folded into `duplicate_structures_rejected`; here we report per-reason construction failures.
    let fusion_failures = fusion_result.failures_by_reason.clone();
    let _ = fusion_result.eligible_candidates; // available for future reporting

    Stats {
        input_transcripts: input.input_transcript_count,
        input_genes: input.genes.len(),
        output_transcripts: final_transcripts.len(),
        ordinary_genes: ordinary_gene_ids.len(),
        fusion_pseudo_genes: fusion_gene_ids.len(),
        generated_events_by_type: generated_events,
        requested_events_by_type: requested_events,
        failed_event_attempts,
        duplicate_structures_rejected,
        ineligible_genes_by_event_type: ineligible_genes_by_type(input),
        requested_fusions: fusion_result.requested,
        successful_fusions,
        fusion_failures_by_reason: fusion_failures,
        total_requested_molecules: params.total_molecules,
        total_allocated_gene_molecules: expr.total_allocated_gene,
        total_allocated_isoform_molecules: expr.total_allocated_isoform,
        zero_count_genes: expr.zero_count_genes,
        zero_count_isoforms: expr.zero_count_isoforms,
    }
}
