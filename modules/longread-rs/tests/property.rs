//! Randomized property tests over the event primitives (spec §7).
//!
//! We generate many random *valid* exon structures and assert that every event output preserves
//! block ordering, positive block lengths, non-overlap, transcript bounds, strand-correct
//! donor/acceptor movement, valid truncation, and determinism.

use longread::events::{apply, eligible};
use longread::model::{EventType, Strand};
use longread::rng::rng_from_seed;
use rand::Rng;

/// Generate a random valid exon structure with `n` exons (sorted, positive, separated).
fn random_exons(rng: &mut rand_chacha::ChaCha8Rng, n: usize) -> Vec<(u64, u64)> {
    let mut pos: u64 = rng.gen_range(0..1000);
    let mut exons = Vec::with_capacity(n);
    for _ in 0..n {
        let exon_len = rng.gen_range(1..=200u64);
        let start = pos;
        let end = start + exon_len;
        exons.push((start, end));
        let intron_len = rng.gen_range(1..=300u64);
        pos = end + intron_len;
    }
    exons
}

/// Assert a structure is sorted, positive-length, and non-overlapping (positive introns).
fn assert_valid_blocks(exons: &[(u64, u64)]) {
    assert!(!exons.is_empty(), "no blocks");
    for &(s, e) in exons {
        assert!(e > s, "non-positive block {exons:?}");
    }
    for w in exons.windows(2) {
        assert!(w[0].1 < w[1].0, "unsorted/overlapping blocks {exons:?}");
    }
}

fn exonic_len(exons: &[(u64, u64)]) -> u64 {
    exons.iter().map(|(s, e)| e - s).sum()
}

#[test]
fn events_preserve_structural_invariants() {
    let mut meta = rng_from_seed(0xC0FFEE);
    for iter in 0..4000u64 {
        let n = meta.gen_range(1..=7usize);
        let strand = if meta.gen::<bool>() {
            Strand::Plus
        } else {
            Strand::Minus
        };
        let parent = random_exons(&mut meta, n);
        let parent_min = parent.first().unwrap().0;
        let parent_max = parent.last().unwrap().1;

        for &event in &EventType::ALL {
            if !eligible(&parent, strand, event) {
                continue;
            }
            let mut rng = rng_from_seed(iter.wrapping_mul(0x9E3779B1).wrapping_add(event as u64));
            let Some(out) = apply(&parent, strand, event, 0, &mut rng) else {
                continue;
            };
            assert_valid_blocks(&out);
            // Bounds: events never leave the parent's genomic span.
            assert!(out.first().unwrap().0 >= parent_min, "left bound {out:?}");
            assert!(out.last().unwrap().1 <= parent_max, "right bound {out:?}");
            // Never a structural copy of the parent.
            assert_ne!(
                out.as_slice(),
                parent.as_slice(),
                "copy of parent {event:?}"
            );

            match event {
                EventType::ExonSkipping => {
                    assert_eq!(out.len(), parent.len() - 1);
                    assert!(out.len() >= 2);
                }
                EventType::IntronRetention => {
                    assert_eq!(out.len(), parent.len() - 1);
                }
                EventType::AlternativeDonor | EventType::AlternativeAcceptor => {
                    // Exon count preserved; spliced length strictly increases (extends into intron).
                    assert_eq!(out.len(), parent.len());
                    assert!(
                        exonic_len(&out) > exonic_len(&parent),
                        "{event:?} did not extend: {parent:?} -> {out:?}"
                    );
                    assert_donor_acceptor_direction(&parent, &out, strand, event);
                }
                EventType::FivePrimeTruncation => {
                    assert!(out.len() >= 2, "truncation kept < 2 exons: {out:?}");
                    assert!(
                        exonic_len(&out) < exonic_len(&parent),
                        "truncation did not shorten: {parent:?} -> {out:?}"
                    );
                }
            }
        }
    }
}

/// Verify the single changed boundary matches the strand-specific donor/acceptor rule.
fn assert_donor_acceptor_direction(
    parent: &[(u64, u64)],
    out: &[(u64, u64)],
    strand: Strand,
    event: EventType,
) {
    assert_eq!(parent.len(), out.len());
    let mut start_changed = false;
    let mut end_changed = false;
    let mut changes = 0;
    for (p, o) in parent.iter().zip(out.iter()) {
        if p.0 != o.0 {
            start_changed = true;
            changes += 1;
            assert!(
                o.0 < p.0,
                "a moved start must decrease (extend into intron)"
            );
        }
        if p.1 != o.1 {
            end_changed = true;
            changes += 1;
            assert!(o.1 > p.1, "a moved end must increase (extend into intron)");
        }
    }
    assert_eq!(
        changes, 1,
        "exactly one boundary must move: {parent:?} -> {out:?}"
    );
    // + donor / - acceptor move a left boundary (an exon END increases).
    // + acceptor / - donor move a right boundary (an exon START decreases).
    let expect_end = matches!(
        (strand, event),
        (Strand::Plus, EventType::AlternativeDonor)
            | (Strand::Minus, EventType::AlternativeAcceptor)
    );
    if expect_end {
        assert!(
            end_changed && !start_changed,
            "expected an END to move for {strand:?} {event:?}"
        );
    } else {
        assert!(
            start_changed && !end_changed,
            "expected a START to move for {strand:?} {event:?}"
        );
    }
}

#[test]
fn events_are_deterministic() {
    let mut meta = rng_from_seed(7);
    for iter in 0..1000u64 {
        let n = meta.gen_range(2..=6usize);
        let strand = if meta.gen::<bool>() {
            Strand::Plus
        } else {
            Strand::Minus
        };
        let parent = random_exons(&mut meta, n);
        for &event in &EventType::ALL {
            if !eligible(&parent, strand, event) {
                continue;
            }
            let mut a = rng_from_seed(iter + 1);
            let mut b = rng_from_seed(iter + 1);
            let ra = apply(&parent, strand, event, 0, &mut a);
            let rb = apply(&parent, strand, event, 0, &mut b);
            assert_eq!(ra, rb, "non-deterministic {event:?} on {parent:?}");
        }
    }
}

#[test]
fn eligibility_matches_exon_counts() {
    // Exon skipping needs >= 3, intron retention needs >= 2.
    let one = vec![(0u64, 10u64)];
    let two = vec![(0, 10), (20, 30)];
    let three = vec![(0, 10), (20, 30), (40, 50)];
    assert!(!eligible(&one, Strand::Plus, EventType::ExonSkipping));
    assert!(!eligible(&two, Strand::Plus, EventType::ExonSkipping));
    assert!(eligible(&three, Strand::Plus, EventType::ExonSkipping));
    assert!(!eligible(&one, Strand::Plus, EventType::IntronRetention));
    assert!(eligible(&two, Strand::Plus, EventType::IntronRetention));
}
