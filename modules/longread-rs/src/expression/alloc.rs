//! Deterministic largest-remainder (Hamilton) integer allocation.
//!
//! Given non-negative real weights and an integer `total`, distribute `total` integer units
//! across the items so that the result sums *exactly* to `total`, favouring the largest
//! fractional remainders and breaking ties deterministically by a caller-supplied key.

/// Allocate `total` integer units across `weights`, tie-breaking by `keys[i]` (ascending).
///
/// Invariant: `sum(result) == total`, and `result.len() == weights.len()`.
pub fn largest_remainder<K: Ord>(weights: &[f64], total: u64, keys: &[K]) -> Vec<u64> {
    let n = weights.len();
    assert_eq!(n, keys.len(), "weights and keys must have equal length");
    if n == 0 {
        return Vec::new();
    }
    if total == 0 {
        return vec![0; n];
    }

    let sum: f64 = weights
        .iter()
        .copied()
        .filter(|w| w.is_finite() && *w > 0.0)
        .sum();
    if sum <= 0.0 {
        // No positive weight: fall back to an even, deterministic distribution.
        return even_distribution(total, keys);
    }

    let total_f = total as f64;
    let mut floors = vec![0u64; n];
    let mut remainders: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut allocated: u64 = 0;
    for (i, &w) in weights.iter().enumerate() {
        let ideal = if w.is_finite() && w > 0.0 {
            w / sum * total_f
        } else {
            0.0
        };
        let floor = ideal.floor();
        let f = floor.max(0.0) as u64;
        floors[i] = f;
        allocated += f;
        remainders.push((i, ideal - floor));
    }

    // `allocated <= total` because sum of ideals == total and floors <= ideals.
    let mut leftover = total.saturating_sub(allocated);

    // Order by remainder desc, then key asc, then index asc.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        remainders[b]
            .1
            .partial_cmp(&remainders[a].1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| keys[a].cmp(&keys[b]))
            .then_with(|| a.cmp(&b))
    });

    let mut i = 0;
    while leftover > 0 && !order.is_empty() {
        floors[order[i % order.len()]] += 1;
        leftover -= 1;
        i += 1;
    }
    floors
}

/// Distribute `total` as evenly as possible, giving the remainder to the smallest keys.
fn even_distribution<K: Ord>(total: u64, keys: &[K]) -> Vec<u64> {
    let n = keys.len() as u64;
    let base = total / n;
    let mut rem = total % n;
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|&a, &b| keys[a].cmp(&keys[b]).then_with(|| a.cmp(&b)));
    let mut out = vec![base; keys.len()];
    for &idx in &order {
        if rem == 0 {
            break;
        }
        out[idx] += 1;
        rem -= 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_to_total() {
        let w = [1.0, 2.0, 3.0, 4.0];
        let keys = ["a", "b", "c", "d"];
        let out = largest_remainder(&w, 100, &keys);
        assert_eq!(out.iter().sum::<u64>(), 100);
    }

    #[test]
    fn exact_when_divisible() {
        let w = [1.0, 1.0, 1.0, 1.0];
        let keys = [0, 1, 2, 3];
        let out = largest_remainder(&w, 8, &keys);
        assert_eq!(out, vec![2, 2, 2, 2]);
    }

    #[test]
    fn tie_break_by_key() {
        // Equal weights, total not divisible: remainder goes to smallest keys.
        let w = [1.0, 1.0, 1.0];
        let keys = ["z", "a", "m"];
        let out = largest_remainder(&w, 4, &keys);
        assert_eq!(out.iter().sum::<u64>(), 4);
        // "a" (smallest key) should receive the extra unit.
        let idx_a = 1;
        assert_eq!(out[idx_a], 2);
    }

    #[test]
    fn zero_total_all_zero() {
        let w = [1.0, 2.0];
        let keys = [0, 1];
        assert_eq!(largest_remainder(&w, 0, &keys), vec![0, 0]);
    }

    #[test]
    fn zero_weights_even_fallback() {
        let w = [0.0, 0.0, 0.0];
        let keys = [0, 1, 2];
        let out = largest_remainder(&w, 5, &keys);
        assert_eq!(out.iter().sum::<u64>(), 5);
    }
}
