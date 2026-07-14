//! Deterministic, portable seeding utilities (spec §4.8).
//!
//! We deliberately avoid Rust's standard-library hashers: `HashMap`'s default hasher is
//! randomized per process, and even `DefaultHasher::new()` (fixed keys) is not guaranteed
//! stable across toolchain versions. Instead we use a small, fully specified mix of FNV-1a
//! and SplitMix64 so that the same inputs always yield the same seed on any platform.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// One round of SplitMix64 finalization over `x`.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Fold a byte slice into a running FNV-1a hash.
#[inline]
fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// A byte separator mixed between hash components so that `("ab", "c")` and `("a", "bc")`
/// do not collide.
const SEP: &[u8] = b"\x1f";

/// Derive a stable `u64` seed from the global seed, a namespace, and a key.
///
/// Deterministic and portable across platforms and thread counts.
pub fn stable_seed(global_seed: u64, namespace: &str, key: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    h = fnv1a(h, &global_seed.to_le_bytes());
    h = fnv1a(h, SEP);
    h = fnv1a(h, namespace.as_bytes());
    h = fnv1a(h, SEP);
    h = fnv1a(h, key);
    splitmix64(h)
}

/// Derive a stable `u64` seed keyed by an integer ordinal (used for fusions).
pub fn stable_seed_ordinal(global_seed: u64, namespace: &str, ordinal: u64) -> u64 {
    stable_seed(global_seed, namespace, &ordinal.to_le_bytes())
}

/// Construct a deterministic ChaCha8 generator from a `u64` seed.
#[inline]
pub fn rng_from_seed(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_seed_is_deterministic() {
        let a = stable_seed(42, "events", b"ENSG1");
        let b = stable_seed(42, "events", b"ENSG1");
        assert_eq!(a, b);
    }

    #[test]
    fn stable_seed_varies_by_input() {
        let base = stable_seed(42, "events", b"ENSG1");
        assert_ne!(base, stable_seed(43, "events", b"ENSG1"));
        assert_ne!(base, stable_seed(42, "gweight", b"ENSG1"));
        assert_ne!(base, stable_seed(42, "events", b"ENSG2"));
    }

    #[test]
    fn separator_prevents_boundary_collision() {
        // ("ab","c") vs ("a","bc") must differ thanks to the separator.
        assert_ne!(stable_seed(0, "ab", b"c"), stable_seed(0, "a", b"bc"),);
    }
}
