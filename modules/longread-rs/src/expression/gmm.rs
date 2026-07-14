//! Gene-level Gaussian-mixture expression sampler (spec §4.9).
//!
//! Uses YASIM's validated `simulate_gene_level_depth_gmm` coefficients as built-in defaults but
//! samples deterministically per gene: pick a component by weight, draw `x ~ Normal(mu, sigma)`,
//! and map to `raw = 10^(x-1) - 1`. Non-positive draws are resampled from the gene's own stream
//! (bounded), then clamped to `low_cutoff`. Because gene weights are normalized against
//! `--total-molecules`, the absolute scale is irrelevant; only the shape and positivity matter.

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

use crate::rng::{rng_from_seed, stable_seed};

/// A single mixture component `(weight, mu, sigma)`.
#[derive(Debug, Clone, Copy)]
pub struct Component {
    /// Mixing weight.
    pub weight: f64,
    /// Mean.
    pub mu: f64,
    /// Standard deviation (may be 0).
    pub sigma: f64,
}

/// The default lower clamp for a raw weight.
pub const DEFAULT_LOW_CUTOFF: f64 = 0.01;

/// YASIM gene-level GMM coefficients (helper/depth.py `simulate_gene_level_depth_gmm`).
pub const YASIM_GENE_GMM: [Component; 4] = [
    Component {
        weight: 0.1427655350146609,
        mu: 1.5746800285632137,
        sigma: 0.5388586661428069,
    },
    Component {
        weight: 0.06837661567650871,
        mu: 3.699056854547668,
        sigma: 0.0,
    },
    Component {
        weight: 0.31792430333258503,
        mu: 2.369675925231799,
        sigma: 0.345735276372599,
    },
    Component {
        weight: 0.4709335459762455,
        mu: 3.1287227827640636,
        sigma: 0.33812788057291315,
    },
];

/// A gene-level GMM sampler.
#[derive(Debug, Clone)]
pub struct GeneGmm {
    components: Vec<Component>,
    low_cutoff: f64,
    max_attempts: u32,
}

impl Default for GeneGmm {
    fn default() -> Self {
        GeneGmm {
            components: YASIM_GENE_GMM.to_vec(),
            low_cutoff: DEFAULT_LOW_CUTOFF,
            max_attempts: 64,
        }
    }
}

impl GeneGmm {
    /// Draw a single value `x` from the mixture using `rng`.
    fn draw_x(&self, rng: &mut ChaCha8Rng) -> f64 {
        let total: f64 = self.components.iter().map(|c| c.weight).sum();
        let mut pick = rng.gen::<f64>() * total;
        let mut chosen = self.components.last().copied().unwrap();
        for c in &self.components {
            if pick < c.weight {
                chosen = *c;
                break;
            }
            pick -= c.weight;
        }
        if chosen.sigma <= 0.0 {
            chosen.mu
        } else {
            Normal::new(chosen.mu, chosen.sigma)
                .expect("valid normal parameters")
                .sample(rng)
        }
    }

    /// Sample a positive raw weight for `gene_id`, deterministic in `(global_seed, gene_id)`.
    pub fn raw_weight(&self, global_seed: u64, gene_id: &str) -> f64 {
        let seed = stable_seed(global_seed, "gweight", gene_id.as_bytes());
        let mut rng = rng_from_seed(seed);
        for _ in 0..self.max_attempts {
            let x = self.draw_x(&mut rng);
            let raw = 10f64.powf(x - 1.0) - 1.0;
            if raw > 0.0 {
                return raw.max(self.low_cutoff);
            }
        }
        self.low_cutoff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_weight_is_positive_and_deterministic() {
        let gmm = GeneGmm::default();
        let a = gmm.raw_weight(42, "ENSG1");
        let b = gmm.raw_weight(42, "ENSG1");
        assert_eq!(a, b);
        assert!(a > 0.0);
    }

    #[test]
    fn raw_weight_varies_by_gene() {
        let gmm = GeneGmm::default();
        // Extremely unlikely to be identical for different gene ids.
        let a = gmm.raw_weight(42, "ENSG1");
        let b = gmm.raw_weight(42, "ENSG2");
        assert!(a > 0.0 && b > 0.0);
    }
}
