//! Per-gene synthetic isoform generation (spec §4.4, §4.7, §4.8).
//!
//! Each gene is generated independently from a seed derived solely from `(global_seed, gene_id)`,
//! so results are identical regardless of thread count. Only original isoforms are used as event
//! parents; generated transcripts never become parents (no recursion).

use std::collections::{HashMap, HashSet};

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Poisson};
use rayon::prelude::*;

use crate::events;
use crate::model::{EventType, Provenance, Strand, StructuralKey, Transcript};
use crate::rng::{rng_from_seed, stable_seed};
use crate::validate::GeneModel;

/// How many new isoforms to request per gene.
#[derive(Debug, Clone, Copy)]
pub enum NewIsoformsMode {
    /// A fixed count for every gene.
    Fixed(u32),
    /// Sample from `Poisson(mean)` and cap at `max`.
    PoissonCapped {
        /// Poisson mean.
        mean: f64,
        /// Hard cap.
        max: u32,
    },
}

/// Per-event selection weights, indexed by `EventType::ALL`.
#[derive(Debug, Clone, Copy)]
pub struct EventWeights([f64; 5]);

impl EventWeights {
    /// All weights equal to 1.0.
    pub fn uniform() -> Self {
        EventWeights([1.0; 5])
    }

    /// Build from a per-type array (order of `EventType::ALL`).
    pub fn from_array(w: [f64; 5]) -> Self {
        EventWeights(w)
    }

    /// Weight for a given event type.
    pub fn weight(&self, e: EventType) -> f64 {
        let idx = EventType::ALL.iter().position(|&x| x == e).unwrap();
        self.0[idx]
    }

    /// Whether any weight is positive.
    pub fn any_positive(&self) -> bool {
        self.0.iter().any(|&w| w > 0.0)
    }
}

/// Parameters controlling generation.
#[derive(Debug, Clone, Copy)]
pub struct GenerationParams {
    /// Global seed.
    pub global_seed: u64,
    /// New-isoform count mode.
    pub mode: NewIsoformsMode,
    /// Event selection weights.
    pub weights: EventWeights,
    /// Maximum attempts per requested isoform.
    pub max_event_attempts: u32,
    /// Minimum transcript (exonic) length.
    pub min_transcript_length: u64,
}

/// The outcome of generating for a single gene.
#[derive(Debug, Default, Clone)]
pub struct GeneOutcome {
    /// Newly generated isoforms.
    pub generated: Vec<Transcript>,
    /// Number of new isoforms requested for this gene (fixed count or Poisson draw).
    pub requested_isoforms: u32,
    /// Per-event selection counts (attempts that chose each event), indexed by `EventType::ALL`.
    pub selections: [u64; 5],
    /// Number of attempts whose event application returned `None`.
    pub failed_attempts: u64,
    /// Number of locally rejected duplicate structures.
    pub local_duplicates: u64,
    /// Whether the gene had no eligible event at all.
    pub no_eligible_event: bool,
}

/// Index of an event type in `EventType::ALL`.
fn event_index(e: EventType) -> usize {
    EventType::ALL.iter().position(|&x| x == e).unwrap()
}

/// Sample the requested number of new isoforms for a gene.
fn sample_new_count(mode: NewIsoformsMode, rng: &mut ChaCha8Rng) -> u32 {
    match mode {
        NewIsoformsMode::Fixed(n) => n,
        NewIsoformsMode::PoissonCapped { mean, max } => {
            if mean <= 0.0 {
                return 0;
            }
            // Poisson::new only fails for non-positive lambda, handled above.
            let p = Poisson::new(mean).expect("positive lambda");
            let sample = p.sample(rng);
            (sample.round() as u64).min(max as u64) as u32
        }
    }
}

/// Choose an eligible, positively-weighted event for `parent` using cumulative weighting.
fn choose_event(
    exons: &[(u64, u64)],
    strand: Strand,
    weights: &EventWeights,
    rng: &mut ChaCha8Rng,
) -> Option<EventType> {
    let mut choices: Vec<(EventType, f64)> = Vec::with_capacity(5);
    for &e in &EventType::ALL {
        let w = weights.weight(e);
        if w > 0.0 && events::eligible(exons, strand, e) {
            choices.push((e, w));
        }
    }
    if choices.is_empty() {
        return None;
    }
    let total: f64 = choices.iter().map(|(_, w)| w).sum();
    let mut pick = rng.gen::<f64>() * total;
    for (e, w) in &choices {
        if pick < *w {
            return Some(*e);
        }
        pick -= *w;
    }
    // Floating-point fall-through: return the last choice.
    Some(choices.last().unwrap().0)
}

/// Generate synthetic isoforms for one gene.
pub fn generate_for_gene(gene: &GeneModel, params: &GenerationParams) -> GeneOutcome {
    let mut outcome = GeneOutcome::default();

    if !params.weights.any_positive() {
        return outcome;
    }

    // Parents that have at least one eligible, positively-weighted event.
    let eligible_parents: Vec<&Transcript> = gene
        .transcripts
        .iter()
        .filter(|p| {
            EventType::ALL
                .iter()
                .any(|&e| params.weights.weight(e) > 0.0 && events::eligible(&p.exons, p.strand, e))
        })
        .collect();

    if eligible_parents.is_empty() {
        outcome.no_eligible_event = true;
        return outcome;
    }

    let seed = stable_seed(params.global_seed, "events", gene.gene_id.as_bytes());
    let mut rng = rng_from_seed(seed);

    let n_new = sample_new_count(params.mode, &mut rng);
    outcome.requested_isoforms = n_new;

    // Local structural set: this gene's originals plus accepted generated structures.
    let mut seen: HashSet<StructuralKey> = gene
        .transcripts
        .iter()
        .map(|t| t.structural_key())
        .collect();
    // Per-(parent id, event) ID counter for deterministic, unique ids.
    let mut id_counter: HashMap<(String, EventType), u32> = HashMap::new();

    for _ in 0..n_new {
        for _attempt in 0..params.max_event_attempts {
            let parent = eligible_parents[rng.gen_range(0..eligible_parents.len())];
            let Some(event) = choose_event(&parent.exons, parent.strand, &params.weights, &mut rng)
            else {
                // Should not happen (parent was pre-filtered), but guard anyway.
                continue;
            };
            outcome.selections[event_index(event)] += 1;

            let Some(new_exons) = events::apply(
                &parent.exons,
                parent.strand,
                event,
                params.min_transcript_length,
                &mut rng,
            ) else {
                outcome.failed_attempts += 1;
                continue;
            };

            let starts: Vec<u64> = new_exons.iter().map(|e| e.0).collect();
            let ends: Vec<u64> = new_exons.iter().map(|e| e.1).collect();
            let key: StructuralKey = (gene.chrom.clone(), gene.strand, starts, ends);
            if seen.contains(&key) {
                outcome.local_duplicates += 1;
                continue;
            }
            seen.insert(key);

            let counter = id_counter.entry((parent.id.clone(), event)).or_insert(0);
            *counter += 1;
            let id = format!("{}-{}-{}", parent.id, event.code(), counter);

            let chrom_start = new_exons.first().map(|e| e.0).unwrap_or(0);
            let transcript = Transcript {
                id,
                gene_id: gene.gene_id.clone(),
                chrom: gene.chrom.clone(),
                strand: gene.strand,
                exons: new_exons,
                // Generated transcripts do not carry an inferred coding frame (see decisions.md).
                thick_start: chrom_start,
                thick_end: chrom_start,
                provenance: Provenance::Generated(event),
                parent_id: Some(parent.id.clone()),
                fusion: None,
                coding_defined: false,
                seed,
            };
            outcome.generated.push(transcript);
            break;
        }
    }

    outcome
}

/// Generate synthetic isoforms across all genes in parallel (deterministic in thread count).
pub fn generate_all(genes: &[GeneModel], params: &GenerationParams) -> Vec<GeneOutcome> {
    genes
        .par_iter()
        .map(|gene| generate_for_gene(gene, params))
        .collect()
}
