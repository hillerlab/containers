//! Within-gene isoform weighting (spec §4.10).
//!
//! Each of a gene's `k` isoforms is assigned a distinct rank `r ∈ 1..=k` by shuffling with the
//! gene's isoform seed; the relative weight of rank `r` is `r^(-alpha)`. Weights are normalized
//! within the gene so they sum to 1 (the constant `(alpha-1)` factor in YASIM cancels here).

use rand::seq::SliceRandom;

use crate::rng::{rng_from_seed, stable_seed};

/// Relative within-gene weights for `k` isoforms, aligned to the caller's isoform order.
///
/// The result sums to 1 (for `k >= 1`); rank assignment is deterministic in
/// `(global_seed, gene_id)`.
pub fn isoform_weights(k: usize, alpha: f64, global_seed: u64, gene_id: &str) -> Vec<f64> {
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        return vec![1.0];
    }
    let seed = stable_seed(global_seed, "isoform", gene_id.as_bytes());
    let mut rng = rng_from_seed(seed);
    // perm[i] + 1 is the rank assigned to isoform i (in the caller's order).
    let mut perm: Vec<usize> = (0..k).collect();
    perm.shuffle(&mut rng);

    let raw: Vec<f64> = perm
        .iter()
        .map(|&p| ((p + 1) as f64).powf(-alpha))
        .collect();
    let sum: f64 = raw.iter().sum();
    if sum <= 0.0 {
        return vec![1.0 / k as f64; k];
    }
    raw.iter().map(|w| w / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_isoform_gets_all_weight() {
        assert_eq!(isoform_weights(1, 4.0, 1, "g"), vec![1.0]);
    }

    #[test]
    fn weights_sum_to_one() {
        let w = isoform_weights(5, 4.0, 7, "gene");
        let s: f64 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "sum was {s}");
        assert_eq!(w.len(), 5);
    }

    #[test]
    fn deterministic() {
        assert_eq!(
            isoform_weights(6, 4.0, 3, "gene"),
            isoform_weights(6, 4.0, 3, "gene")
        );
    }
}
