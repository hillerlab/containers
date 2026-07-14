//! Transcript event primitives (spec §4.5).
//!
//! Every function operates on a parent's exon list (sorted, non-overlapping, half-open,
//! absolute genomic intervals) and returns a **new** exon list for the generated isoform, or
//! `None` when the parent is ineligible or a random attempt did not yield a valid structure.
//! Donor/acceptor sites are defined in transcript orientation and handled per strand.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::model::{EventType, Strand};

/// Which genomic boundary of an intron to move.
#[derive(Clone, Copy)]
enum Boundary {
    /// The left (lower-coordinate) boundary = `exon[j].end`.
    Left,
    /// The right (higher-coordinate) boundary = `exon[j+1].start`.
    Right,
}

/// Whether any intron is long enough (`>= 2`) to host an interior splice site.
fn has_movable_intron(exons: &[(u64, u64)]) -> bool {
    exons.windows(2).any(|w| w[1].0 > w[0].1 + 1)
}

/// Is `parent` eligible for `event`?
pub fn eligible(exons: &[(u64, u64)], strand: Strand, event: EventType) -> bool {
    match event {
        EventType::ExonSkipping => exons.len() >= 3,
        EventType::IntronRetention => exons.len() >= 2,
        EventType::AlternativeDonor | EventType::AlternativeAcceptor => {
            exons.len() >= 2 && has_movable_intron(exons)
        }
        EventType::FivePrimeTruncation => !truncation_candidates(exons, strand).is_empty(),
    }
}

/// Apply `event` to `exons`, returning a new exon list or `None`.
pub fn apply(
    exons: &[(u64, u64)],
    strand: Strand,
    event: EventType,
    min_len: u64,
    rng: &mut ChaCha8Rng,
) -> Option<Vec<(u64, u64)>> {
    let result = match event {
        EventType::ExonSkipping => exon_skipping(exons, rng),
        EventType::IntronRetention => intron_retention(exons, rng),
        EventType::AlternativeDonor => alternative_donor(exons, strand, rng),
        EventType::AlternativeAcceptor => alternative_acceptor(exons, strand, rng),
        EventType::FivePrimeTruncation => five_prime_truncation(exons, strand, rng),
    }?;
    // Enforce the minimum transcript length and never emit a structural copy of the parent.
    let exonic_len: u64 = result.iter().map(|(s, e)| e - s).sum();
    if exonic_len < min_len {
        return None;
    }
    if result.as_slice() == exons {
        return None;
    }
    Some(result)
}

/// Exon skipping: remove one internal exon. Requires `>= 3` exons; output has `>= 2`.
fn exon_skipping(exons: &[(u64, u64)], rng: &mut ChaCha8Rng) -> Option<Vec<(u64, u64)>> {
    let n = exons.len();
    if n < 3 {
        return None;
    }
    let i = rng.gen_range(1..=n - 2); // internal exon
    let mut out = Vec::with_capacity(n - 1);
    out.extend_from_slice(&exons[..i]);
    out.extend_from_slice(&exons[i + 1..]);
    Some(out)
}

/// Intron retention: merge the two exons flanking one intron. Requires `>= 2` exons.
fn intron_retention(exons: &[(u64, u64)], rng: &mut ChaCha8Rng) -> Option<Vec<(u64, u64)>> {
    let n = exons.len();
    if n < 2 {
        return None;
    }
    let j = rng.gen_range(0..n - 1); // intron between exon j and j+1
    let merged = (exons[j].0, exons[j + 1].1);
    let mut out = Vec::with_capacity(n - 1);
    out.extend_from_slice(&exons[..j]);
    out.push(merged);
    out.extend_from_slice(&exons[j + 2..]);
    Some(out)
}

/// Move an intron boundary strictly into the intron interior.
fn move_boundary(
    exons: &[(u64, u64)],
    boundary: Boundary,
    rng: &mut ChaCha8Rng,
) -> Option<Vec<(u64, u64)>> {
    let n = exons.len();
    let candidates: Vec<usize> = (0..n.saturating_sub(1))
        .filter(|&j| exons[j + 1].0 > exons[j].1 + 1)
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let j = candidates[rng.gen_range(0..candidates.len())];
    let intron_start = exons[j].1;
    let intron_end = exons[j + 1].0;
    // interior position strictly inside (intron_start, intron_end)
    let pos = rng.gen_range(intron_start + 1..=intron_end - 1);
    let mut out = exons.to_vec();
    match boundary {
        Boundary::Left => out[j].1 = pos,      // extend exon j rightward
        Boundary::Right => out[j + 1].0 = pos, // extend exon j+1 leftward
    }
    Some(out)
}

/// Alternative donor: move the transcript-5′ intron boundary.
/// `+` strand donor = left boundary; `-` strand donor = right boundary.
fn alternative_donor(
    exons: &[(u64, u64)],
    strand: Strand,
    rng: &mut ChaCha8Rng,
) -> Option<Vec<(u64, u64)>> {
    let boundary = match strand {
        Strand::Plus => Boundary::Left,
        Strand::Minus => Boundary::Right,
    };
    move_boundary(exons, boundary, rng)
}

/// Alternative acceptor: move the transcript-3′ intron boundary.
/// `+` strand acceptor = right boundary; `-` strand acceptor = left boundary.
fn alternative_acceptor(
    exons: &[(u64, u64)],
    strand: Strand,
    rng: &mut ChaCha8Rng,
) -> Option<Vec<(u64, u64)>> {
    let boundary = match strand {
        Strand::Plus => Boundary::Right,
        Strand::Minus => Boundary::Left,
    };
    move_boundary(exons, boundary, rng)
}

/// The set of exon indices eligible for 5′ truncation (spec §4.5), per strand.
///
/// The selected exon may not be the terminal 3′ exon, and the retained half must be positive:
/// - `+`: indices `0..=n-2`, with either `i>0` or the exon long enough to actually shorten.
/// - `-`: indices `1..=n-1`, requiring exon length `>= 2` so `floor(len/2) >= 1`.
fn truncation_candidates(exons: &[(u64, u64)], strand: Strand) -> Vec<usize> {
    let n = exons.len();
    if n < 2 {
        return Vec::new();
    }
    match strand {
        Strand::Plus => (0..=n - 2)
            .filter(|&i| {
                let len = exons[i].1 - exons[i].0;
                i > 0 || len >= 2 // must change the structure
            })
            .collect(),
        Strand::Minus => (1..=n - 1)
            .filter(|&i| (exons[i].1 - exons[i].0) >= 2)
            .collect(),
    }
}

/// Five-prime truncation: keep the downstream half of a selected exon and everything downstream
/// of it in transcript orientation.
fn five_prime_truncation(
    exons: &[(u64, u64)],
    strand: Strand,
    rng: &mut ChaCha8Rng,
) -> Option<Vec<(u64, u64)>> {
    let candidates = truncation_candidates(exons, strand);
    if candidates.is_empty() {
        return None;
    }
    let i = candidates[rng.gen_range(0..candidates.len())];
    let (start, end) = exons[i];
    let mid = start + (end - start) / 2;
    match strand {
        Strand::Plus => {
            // retained part of selected exon = [mid, end); keep downstream (higher) exons.
            let mut out = Vec::with_capacity(exons.len() - i);
            out.push((mid, end));
            out.extend_from_slice(&exons[i + 1..]);
            Some(out)
        }
        Strand::Minus => {
            // retained part of selected exon = [start, mid); keep downstream (lower) exons.
            if mid <= start {
                return None;
            }
            let mut out = Vec::with_capacity(i + 1);
            out.extend_from_slice(&exons[..i]);
            out.push((start, mid));
            Some(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::rng_from_seed;

    fn assert_valid(exons: &[(u64, u64)]) {
        assert!(!exons.is_empty());
        for w in exons.windows(2) {
            assert!(
                w[0].1 < w[1].0,
                "blocks must be sorted and separated: {exons:?}"
            );
        }
        for &(s, e) in exons {
            assert!(e > s, "positive length: {exons:?}");
        }
    }

    #[test]
    fn exon_skip_removes_internal_exon() {
        let exons = vec![(0, 10), (20, 30), (40, 50), (60, 70)];
        let mut rng = rng_from_seed(1);
        let out = exon_skipping(&exons, &mut rng).unwrap();
        assert_eq!(out.len(), 3);
        assert_valid(&out);
        assert_eq!(out.first().unwrap(), &(0, 10));
        assert_eq!(out.last().unwrap(), &(60, 70));
    }

    #[test]
    fn intron_retention_merges() {
        let exons = vec![(0, 10), (20, 30), (40, 50)];
        let mut rng = rng_from_seed(2);
        let out = intron_retention(&exons, &mut rng).unwrap();
        assert_eq!(out.len(), 2);
        assert_valid(&out);
    }

    #[test]
    fn donor_plus_extends_left_boundary() {
        let exons = vec![(0, 10), (20, 30)];
        let mut rng = rng_from_seed(3);
        let out = alternative_donor(&exons, Strand::Plus, &mut rng).unwrap();
        assert_valid(&out);
        assert_eq!(out[0].0, 0);
        assert!(
            out[0].1 > 10 && out[0].1 < 20,
            "donor moved into intron: {out:?}"
        );
        assert_eq!(out[1], (20, 30));
    }

    #[test]
    fn donor_minus_extends_right_boundary() {
        let exons = vec![(0, 10), (20, 30)];
        let mut rng = rng_from_seed(4);
        let out = alternative_donor(&exons, Strand::Minus, &mut rng).unwrap();
        assert_valid(&out);
        assert_eq!(out[0], (0, 10));
        assert!(
            out[1].0 > 10 && out[1].0 < 20,
            "donor moved into intron: {out:?}"
        );
        assert_eq!(out[1].1, 30);
    }

    #[test]
    fn acceptor_is_opposite_boundary_of_donor() {
        let exons = vec![(0, 10), (20, 30)];
        let mut rd = rng_from_seed(7);
        let mut ra = rng_from_seed(7);
        let donor = alternative_donor(&exons, Strand::Plus, &mut rd).unwrap();
        let acceptor = alternative_acceptor(&exons, Strand::Plus, &mut ra).unwrap();
        // + donor moves left boundary (exon 0 end); + acceptor moves right boundary (exon 1 start)
        assert_eq!(donor[1], (20, 30));
        assert_eq!(acceptor[0], (0, 10));
    }

    #[test]
    fn truncation_plus_keeps_downstream() {
        let exons = vec![(0, 10), (20, 30), (40, 50)];
        // Force selecting exon 0 by using candidates directly.
        let out = {
            let mut rng = rng_from_seed(11);
            five_prime_truncation(&exons, Strand::Plus, &mut rng).unwrap()
        };
        assert_valid(&out);
        // First exon must be a right-half of some exon; last exon unchanged.
        assert_eq!(out.last().unwrap(), &(40, 50));
    }

    #[test]
    fn truncation_minus_keeps_downstream() {
        let exons = vec![(0, 10), (20, 30), (40, 50)];
        let mut rng = rng_from_seed(13);
        let out = five_prime_truncation(&exons, Strand::Minus, &mut rng).unwrap();
        assert_valid(&out);
        // For minus strand, first (lowest) exon is retained; last exon is a truncated exon.
        assert_eq!(out.first().unwrap(), &(0, 10));
    }
}
