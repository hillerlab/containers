// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
// Distributed under the terms of the Apache License, Version 2.0.

use dashmap::{DashMap, DashSet};
use log::{debug, info, warn};

use std::{
    cmp::Ordering,
    collections::HashMap,
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
};

use crate::spliceai::{
    parse_spliceai_record, ParsedSpliceAiRecord, SpliceSite, Strand, StrandSpliceMap,
};

/// Bin edges for SpliceAI score calibration (15 edges = 14 bins).
pub const SCORE_BIN_EDGES: [f32; 15] = [
    0.0, 1e-3, 1e-2, 1e-1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999, 1.0,
];

/// Map from dinucleotide to calibration bins.
pub type CalibrationMap = HashMap<Vec<u8>, Vec<CalibrationBin>>;
/// Null profile: dinucleotide -> probability of being annotated.
pub type NullProfile = HashMap<Vec<u8>, f64>;

/// A bin for calibrating SpliceAI scores by dinucleotide.
///
/// # Fields
/// - `lower`: Lower score bound (inclusive)
/// - `upper`: Upper score bound (exclusive)
/// - `n_total`: Total sites in this bin
/// - `n_real`: Annotated sites in this bin
/// - `empirical_probability`: n_real / n_total
#[derive(Debug, Clone)]
pub struct CalibrationBin {
    pub lower: f32,
    pub upper: f32,
    pub n_total: usize,
    pub n_real: usize,
    pub empirical_probability: f64,
}

/// Internal counter for binning sites.
#[derive(Debug, Clone, Copy, Default)]
struct BinCounts {
    n_total: usize,
    n_real: usize,
}

/// A derived splice score record for output.
///
/// # Fields
/// - `chr`: Chromosome
/// - `coordinate`: Position
/// - `strand`: Strand (+/-)
/// - `dinucleotide`: Splice site dinucleotide
/// - `splice_site`: Donor or Acceptor
/// - `derived_score`: Calibrated score
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedScoreRecord {
    pub chr: Vec<u8>,
    pub coordinate: usize,
    pub strand: Strand,
    pub dinucleotide: Vec<u8>,
    pub splice_site: SpliceSite,
    pub derived_score: i32,
}

/// Converts derived score record to tab-separated bytes.
impl DerivedScoreRecord {
    fn as_tsv_bytes(&self) -> Vec<u8> {
        format!(
            "{}\t{}\t{}\t{}\t{}\n",
            String::from_utf8_lossy(&self.chr),
            self.coordinate,
            self.strand,
            self.splice_site,
            // String::from_utf8_lossy(&self.dinucleotide),
            self.derived_score
        )
        .into_bytes()
    }
}

/// Builds a null profile: probability of each dinucleotide being annotated.
///
/// # Arguments
/// * `label` - "donor" or "acceptor" for logging
/// * `annotated_counts` - Counts of annotated dinucleotides
/// * `dinucleotide_count` - Total genomic dinucleotide counts
///
/// # Returns
/// Map from dinucleotide to probability (annotated/total)
///
/// # Example
/// ```rust,ignore
/// let profile = build_null_profile("donor", &donors, &genome_counts);
/// ```
pub fn build_null_profile(
    label: &str,
    annotated_counts: &DashMap<Vec<u8>, usize>,
    dinucleotide_count: &HashMap<Vec<u8>, usize>,
) -> NullProfile {
    info!("Building {} null profile...", label);
    let mut profile = HashMap::new();

    dinucleotide_count.iter().for_each(|(dinucleotide, total)| {
        let annotated = annotated_counts
            .get(dinucleotide)
            .map(|count| *count.value())
            .unwrap_or(0);
        let probability = if *total == 0 {
            warn!(
                "{} null profile has zero genomic count for motif {}",
                label,
                String::from_utf8_lossy(dinucleotide)
            );
            0.0
        } else {
            annotated as f64 / *total as f64
        };

        info!(
            "{} null {} -> {} ({}/{})",
            label,
            String::from_utf8_lossy(dinucleotide),
            probability,
            annotated,
            total,
        );
        profile.insert(dinucleotide.clone(), probability);
    });

    annotated_counts.iter().for_each(|entry| {
        if !dinucleotide_count.contains_key(entry.key()) {
            warn!(
                "{} annotation contains motif {} absent from genome counts",
                label,
                String::from_utf8_lossy(entry.key())
            );
            profile.entry(entry.key().clone()).or_insert(0.0);
        }
    });

    profile
}

/// Creates calibration bins from SpliceAI scores and annotations.
///
/// Bins SpliceAI scores by dinucleotide and whether they overlap annotated sites.
///
/// # Arguments
/// * `spliceai_scores` - Raw SpliceAI scores from BigWig
/// * `annotated_ss` - Set of annotated splice site coordinates
/// * `annotated_counts` - Annotated dinucleotide counts
/// * `dinucleotide_count` - Genomic dinucleotide counts
///
/// # Returns
/// Map from dinucleotide to vector of CalibrationBins
///
/// # Example
/// ```rust,ignore
/// let bins = make_bins(&scores, &annotated, &donor_counts, &genome_counts);
/// ```
pub fn make_bins(
    spliceai_scores: &StrandSpliceMap,
    annotated_ss: &DashSet<Vec<u8>>,
    annotated_counts: &DashMap<Vec<u8>, usize>,
    dinucleotide_count: &HashMap<Vec<u8>, usize>,
) -> CalibrationMap {
    let mut bins: HashMap<Vec<u8>, Vec<BinCounts>> = HashMap::new();

    dinucleotide_count.keys().for_each(|dinucleotide| {
        bins.insert(dinucleotide.clone(), empty_bin_counts());
    });
    annotated_counts.iter().for_each(|entry| {
        bins.entry(entry.key().clone())
            .or_insert_with(empty_bin_counts);
    });

    spliceai_scores.iter().for_each(|entry| {
        let (chr, splice_sites) = entry.pair();
        debug!(
            "Binning {} scored splice sites from chromosome {}",
            splice_sites.len(),
            chr
        );

        splice_sites.iter().for_each(|ss| {
            let Some(parsed) = parse_spliceai_record(ss.as_slice()) else {
                warn!(
                    "Skipping malformed SpliceAI record while binning: {}",
                    String::from_utf8_lossy(ss.as_slice())
                );
                return;
            };

            let bin_index = find_bin_index(parsed.score);
            let motif_bins = bins
                .entry(parsed.dinucleotide.clone())
                .or_insert_with(empty_bin_counts);
            motif_bins[bin_index].n_total += 1;

            if annotated_ss.contains(parsed.coordinate_key.as_slice()) {
                motif_bins[bin_index].n_real += 1;
            }
        });
    });

    bins.into_iter()
        .map(|(dinucleotide, mut counts)| {
            let recovered_total = counts.iter().map(|bin| bin.n_total).sum::<usize>();
            let recovered_real = counts.iter().map(|bin| bin.n_real).sum::<usize>();
            let expected_total = dinucleotide_count.get(&dinucleotide).copied().unwrap_or(0);
            let expected_real = annotated_counts
                .get(&dinucleotide)
                .map(|count| *count.value())
                .unwrap_or(0);

            if recovered_total > expected_total {
                warn!(
                    "Recovered {} scores for motif {} but genome count is {}. Clamping zero bin to 0.",
                    recovered_total,
                    String::from_utf8_lossy(&dinucleotide),
                    expected_total
                );
            }
            if recovered_real > expected_real {
                warn!(
                    "Recovered {} annotated scores for motif {} but annotation count is {}. Clamping zero bin to 0.",
                    recovered_real,
                    String::from_utf8_lossy(&dinucleotide),
                    expected_real
                );
            }

            let zero_total = expected_total.saturating_sub(recovered_total);
            let zero_real = expected_real.saturating_sub(recovered_real).min(zero_total);

            counts[0].n_total += zero_total;
            counts[0].n_real += zero_real;

            let calibration_bins = counts
                .into_iter()
                .zip(SCORE_BIN_EDGES.windows(2))
                .map(|(count, window)| CalibrationBin {
                    lower: window[0],
                    upper: window[1],
                    n_total: count.n_total,
                    n_real: count.n_real,
                    empirical_probability: if count.n_total == 0 {
                        0.0
                    } else {
                        count.n_real as f64 / count.n_total as f64
                    },
                })
                .collect::<Vec<_>>();

            (dinucleotide, calibration_bins)
        })
        .collect()
}

/// Derives calibrated splice scores using null profile and calibration bins.
///
/// Computes likelihood ratio: calibrated_probability / null_probability,
/// converts to log2 scale, rounds and clamps to floor/ceiling.
///
/// # Arguments
/// * `spliceai_donor_scores` - Donor SpliceAI scores
/// * `spliceai_acceptor_scores` - Acceptor SpliceAI scores
/// * `donor_bins` - Donor calibration bins
/// * `acceptor_bins` - Acceptor calibration bins
/// * `donor_profile` - Donor null profile
/// * `acceptor_profile` - Acceptor null profile
/// * `floor` - Minimum output score
/// * `ceiling` - Maximum output score
/// * `prefix` - Output file prefix
///
/// # Returns
/// Number of records written
///
/// # Example
/// ```rust,ignore
/// let count = derive_spliceai_score(&donor_scores, &acceptor_scores, &donor_bins, &acceptor_bins, &donor_profile, &acceptor_profile, -4, 13, "output")?;
/// ```
pub fn derive_spliceai_score(
    spliceai_donor_scores: &StrandSpliceMap,
    spliceai_acceptor_scores: &StrandSpliceMap,
    donor_bins: &CalibrationMap,
    acceptor_bins: &CalibrationMap,
    donor_profile: &NullProfile,
    acceptor_profile: &NullProfile,
    floor: i32,
    ceiling: i32,
    prefix: &str,
    output_dir: &Path,
) -> io::Result<usize> {
    info!(
        "Deriving splice scores with floor={} and ceiling={}",
        floor, ceiling
    );

    let mut records = derive_records(
        "donor",
        spliceai_donor_scores,
        SpliceSite::Donor,
        donor_bins,
        donor_profile,
        floor,
        ceiling,
    );
    let donor_count = records.len();

    let mut acceptor_records = derive_records(
        "acceptor",
        spliceai_acceptor_scores,
        SpliceSite::Acceptor,
        acceptor_bins,
        acceptor_profile,
        floor,
        ceiling,
    );
    let acceptor_count = acceptor_records.len();
    records.append(&mut acceptor_records);
    records.sort_by(sort_derived_records);

    let output_path = output_dir.join(format!("{}.derived.tsv", prefix));
    write_derived_scores(&output_path, &records)?;

    info!(
        "Wrote {} donor and {} acceptor derived scores to {}",
        donor_count,
        acceptor_count,
        output_path.display()
    );

    Ok(records.len())
}

/// Derives records for a specific splice site type.
///
/// # Arguments
/// * `label` - "donor" or "acceptor" for logging
/// * `spliceai_scores` - SpliceAI scores for the strand
/// * `expected_splice_site` - Expected splice site type (donor/acceptor)
/// * `bins` - Calibration bins map
/// * `null_profile` - Null profile map
/// * `floor` - Minimum output score
/// * `ceiling` - Maximum output score
///
/// # Returns
/// Vector of derived score records
fn derive_records(
    label: &str,
    spliceai_scores: &StrandSpliceMap,
    expected_splice_site: SpliceSite,
    bins: &CalibrationMap,
    null_profile: &NullProfile,
    floor: i32,
    ceiling: i32,
) -> Vec<DerivedScoreRecord> {
    let mut records = Vec::new();

    spliceai_scores.iter().for_each(|entry| {
        let (chr, splice_sites) = entry.pair();
        debug!(
            "Deriving {} scores from chromosome {} ({} records)",
            label,
            chr,
            splice_sites.len()
        );

        splice_sites.iter().for_each(|ss| {
            let Some(parsed) = parse_spliceai_record(ss.as_slice()) else {
                warn!(
                    "Skipping malformed SpliceAI record while deriving: {}",
                    String::from_utf8_lossy(ss.as_slice())
                );
                return;
            };

            if parsed.splice_site != expected_splice_site {
                warn!(
                    "Skipping {} record found in {} collection: {}",
                    match parsed.splice_site {
                        SpliceSite::Donor => "donor",
                        SpliceSite::Acceptor => "acceptor",
                    },
                    label,
                    String::from_utf8_lossy(ss.as_slice())
                );
                return;
            }

            let calibrated_probability =
                lookup_calibrated_probability(parsed.score, &parsed.dinucleotide, bins);
            let null_probability = null_profile
                .get(&parsed.dinucleotide)
                .copied()
                .unwrap_or_else(|| {
                    warn!(
                        "Missing null profile for motif {} in {} collection; treating as 0",
                        String::from_utf8_lossy(&parsed.dinucleotide),
                        label
                    );
                    0.0
                });

            let derived_score = compute_derived_score(
                label,
                &parsed,
                calibrated_probability,
                null_probability,
                floor,
                ceiling,
            );

            // INFO: adding +1 on acceptor(+) and donor(-) to match minisplice coords
            match parsed.strand {
                Strand::Forward => match parsed.splice_site {
                    SpliceSite::Donor => {
                        records.push(DerivedScoreRecord {
                            chr: parsed.chr,
                            coordinate: parsed.position,
                            strand: parsed.strand,
                            dinucleotide: parsed.dinucleotide,
                            splice_site: parsed.splice_site,
                            derived_score,
                        });
                    }
                    SpliceSite::Acceptor => {
                        records.push(DerivedScoreRecord {
                            chr: parsed.chr,
                            coordinate: parsed.position + 1,
                            strand: parsed.strand,
                            dinucleotide: parsed.dinucleotide,
                            splice_site: parsed.splice_site,
                            derived_score,
                        });
                    }
                },
                Strand::Reverse => match parsed.splice_site {
                    SpliceSite::Donor => {
                        records.push(DerivedScoreRecord {
                            chr: parsed.chr,
                            coordinate: parsed.position + 1,
                            strand: parsed.strand,
                            dinucleotide: parsed.dinucleotide,
                            splice_site: parsed.splice_site,
                            derived_score,
                        });
                    }
                    SpliceSite::Acceptor => {
                        records.push(DerivedScoreRecord {
                            chr: parsed.chr,
                            coordinate: parsed.position,
                            strand: parsed.strand,
                            dinucleotide: parsed.dinucleotide,
                            splice_site: parsed.splice_site,
                            derived_score,
                        });
                    }
                },
            }
        });
    });

    info!("Derived {} {} scores", records.len(), label);
    records
}

/// Creates empty bin counts for all score bins.
///
/// # Returns
/// Vector of BinCounts initialized to zero for each score bin
fn empty_bin_counts() -> Vec<BinCounts> {
    SCORE_BIN_EDGES
        .windows(2)
        .map(|_| BinCounts::default())
        .collect()
}

/// Looks up the calibrated probability for a given raw score and dinucleotide.
///
/// # Arguments
/// * `raw_score` - Raw SpliceAI score (0.0 to 1.0)
/// * `dinucleotide` - Dinucleotide sequence (e.g., b"GT")
/// * `bins` - Calibration bins map from dinucleotide to calibration bins
///
/// # Returns
/// Empirical probability for the bin containing the raw score, or 0.0 if bins not found
fn lookup_calibrated_probability(
    raw_score: f32,
    dinucleotide: &[u8],
    bins: &CalibrationMap,
) -> f64 {
    match bins.get(dinucleotide) {
        Some(motif_bins) => {
            let bin_index = find_bin_index(raw_score);
            let calibration_bin = &motif_bins[bin_index];
            debug!(
                "Mapped raw score {} for motif {} into [{}, {}) with empirical probability {}",
                raw_score,
                String::from_utf8_lossy(dinucleotide),
                calibration_bin.lower,
                calibration_bin.upper,
                calibration_bin.empirical_probability
            );
            calibration_bin.empirical_probability
        }
        None => {
            warn!(
                "Missing calibration bins for motif {}; treating calibrated probability as 0",
                String::from_utf8_lossy(dinucleotide)
            );
            0.0
        }
    }
}

/// Computes the derived splice score from calibrated and null probabilities.
///
/// Calculates likelihood ratio: calibrated_probability / null_probability,
/// converts to log2 scale, rounds and clamps to floor/ceiling.
///
/// # Arguments
/// * `label` - "donor" or "acceptor" for logging
/// * `parsed` - Parsed SpliceAI record
/// * `calibrated_probability` - Empirical probability from calibration bins
/// * `null_probability` - Background probability from null profile
/// * `floor` - Minimum output score
/// * `ceiling` - Maximum output score
///
/// # Returns
/// Derived score clamped to [floor, ceiling] range
fn compute_derived_score(
    label: &str,
    parsed: &ParsedSpliceAiRecord,
    calibrated_probability: f64,
    null_probability: f64,
    floor: i32,
    ceiling: i32,
) -> i32 {
    let site_label = format!(
        "{}:{}({}) {}",
        String::from_utf8_lossy(&parsed.chr),
        parsed.position,
        parsed.strand,
        String::from_utf8_lossy(&parsed.dinucleotide)
    );

    if calibrated_probability == 0.0 && null_probability > 0.0 {
        warn!(
            "{} site {} has calibrated probability 0 and null probability {}; clamping to floor {}",
            label, site_label, null_probability, floor
        );
        return floor;
    }

    if calibrated_probability > 0.0 && null_probability == 0.0 {
        warn!(
            "{} site {} has calibrated probability {} and null probability 0; clamping to ceiling {}",
            label, site_label, calibrated_probability, ceiling
        );
        return ceiling;
    }

    if calibrated_probability == 0.0 && null_probability == 0.0 {
        warn!(
            "{} site {} has calibrated and null probabilities both equal to 0; clamping to floor {}",
            label, site_label, floor
        );
        return floor;
    }

    let ratio = calibrated_probability / null_probability;
    if !ratio.is_finite() || ratio <= 0.0 {
        warn!(
            "{} site {} produced an invalid likelihood ratio {}; clamping to floor {}",
            label, site_label, ratio, floor
        );
        return floor;
    }

    let rounded_score = (2.0 * ratio.log2()).round() as i32;
    let clamped_score = rounded_score.clamp(floor, ceiling);

    if clamped_score != rounded_score {
        debug!(
            "Clamped {} site {} from {} to {}",
            label, site_label, rounded_score, clamped_score
        );
    }

    clamped_score
}

/// Writes derived score records to a TSV file.
///
/// # Arguments
/// * `path` - Output file path
/// * `records` - Slice of derived score records to write
///
/// # Returns
/// Result indicating success or I/O error
fn write_derived_scores(path: &Path, records: &[DerivedScoreRecord]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);

    records
        .iter()
        .try_for_each(|record| -> io::Result<()> { writer.write_all(&record.as_tsv_bytes()) })?;
    writer.flush()
}

/// Finds the bin index for a given SpliceAI score.
///
/// # Arguments
/// * `score` - Raw SpliceAI score (0.0 to 1.0)
///
/// # Returns
/// Bin index (0 to 13)
///
/// # Example
/// ```rust,ignore
/// let idx = find_bin_index(0.75); // returns index for [0.6, 0.8)
/// ```
pub fn find_bin_index(score: f32) -> usize {
    let clamped_score = score.clamp(0.0, 1.0);

    SCORE_BIN_EDGES
        .windows(2)
        .enumerate()
        .find(|(index, window)| {
            clamped_score >= window[0]
                && (clamped_score < window[1]
                    || (*index == SCORE_BIN_EDGES.len() - 2 && clamped_score <= window[1]))
        })
        .map(|(index, _)| index)
        .unwrap_or(SCORE_BIN_EDGES.len() - 2)
}

/// Compares two derived score records for sorting.
///
/// Sorts by chromosome, then coordinate, then strand, then dinucleotide.
///
/// # Arguments
/// * `left` - First record to compare
/// * `right` - Second record to compare
///
/// # Returns
/// Ordering of the two records
fn sort_derived_records(left: &DerivedScoreRecord, right: &DerivedScoreRecord) -> Ordering {
    left.chr
        .cmp(&right.chr)
        .then(left.coordinate.cmp(&right.coordinate))
        .then(strand_rank(left.strand).cmp(&strand_rank(right.strand)))
        .then(left.dinucleotide.cmp(&right.dinucleotide))
}

/// Converts a strand to a numeric rank for sorting.
///
/// Forward strand gets rank 0, Reverse strand gets rank 1.
///
/// # Arguments
/// * `strand` - The strand to convert
///
/// # Returns
/// u8 representing the strand rank (0 for Forward, 1 for Reverse)
fn strand_rank(strand: Strand) -> u8 {
    match strand {
        Strand::Forward => 0,
        Strand::Reverse => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(value: &str) -> Vec<u8> {
        value.as_bytes().to_vec()
    }

    #[test]
    fn make_bins_adds_subthreshold_sites_to_lowest_bin() {
        let spliceai_scores = DashMap::new();
        let splice_sites = DashSet::new();
        splice_sites.insert(bytes("chr1:10(+)\tD\t0.98\tGT"));
        splice_sites.insert(bytes("chr1:20(+)\tD\t0.2\tGT"));
        spliceai_scores.insert("chr1".to_string(), splice_sites);

        let annotated_ss = DashSet::new();
        annotated_ss.insert(bytes("chr1:10(+)"));
        annotated_ss.insert(bytes("chr1:30(+)"));

        let annotated_counts = DashMap::new();
        annotated_counts.insert(bytes("GT"), 2);

        let dinucleotide_count = HashMap::from([(bytes("GT"), 5)]);

        let bins = make_bins(
            &spliceai_scores,
            &annotated_ss,
            &annotated_counts,
            &dinucleotide_count,
        );
        let gt_bins = bins.get(&bytes("GT")).unwrap();

        assert_eq!(gt_bins[0].n_total, 3);
        assert_eq!(gt_bins[0].n_real, 1);

        let high_bin = &gt_bins[find_bin_index(0.98)];
        assert_eq!(high_bin.n_total, 1);
        assert_eq!(high_bin.n_real, 1);

        let medium_bin = &gt_bins[find_bin_index(0.2)];
        assert_eq!(medium_bin.n_total, 1);
        assert_eq!(medium_bin.n_real, 0);
    }

    #[test]
    fn find_bin_index_includes_one_in_last_bin() {
        assert_eq!(find_bin_index(1.0), SCORE_BIN_EDGES.len() - 2);
    }

    #[test]
    fn compute_derived_score_handles_zero_policies() {
        let parsed = ParsedSpliceAiRecord {
            coordinate_key: bytes("chr1:10(+)"),
            chr: bytes("chr1"),
            position: 10,
            strand: Strand::Forward,
            splice_site: SpliceSite::Donor,
            score: 0.98,
            dinucleotide: bytes("GT"),
        };

        assert_eq!(compute_derived_score("donor", &parsed, 0.0, 0.2, -7, 7), -7);
        assert_eq!(compute_derived_score("donor", &parsed, 0.2, 0.0, -7, 7), 7);
        assert_eq!(compute_derived_score("donor", &parsed, 0.0, 0.0, -7, 7), -7);
    }

    #[test]
    fn compute_derived_score_rounds_and_clamps_final_value() {
        let parsed = ParsedSpliceAiRecord {
            coordinate_key: bytes("chr1:10(+)"),
            chr: bytes("chr1"),
            position: 10,
            strand: Strand::Forward,
            splice_site: SpliceSite::Donor,
            score: 0.98,
            dinucleotide: bytes("GT"),
        };

        assert_eq!(
            compute_derived_score("donor", &parsed, 0.8, 0.2, -10, 10),
            4
        );
        assert_eq!(compute_derived_score("donor", &parsed, 1.0, 0.01, -3, 3), 3);
    }

    #[test]
    fn derived_score_record_serializes_to_requested_format() {
        let record = DerivedScoreRecord {
            chr: bytes("chr1"),
            coordinate: 10,
            strand: Strand::Forward,
            dinucleotide: bytes("GT"),
            splice_site: SpliceSite::Donor,
            derived_score: 4,
        };

        assert_eq!(
            String::from_utf8(record.as_tsv_bytes()).unwrap(),
            "chr1\t10\t+\tD\t4\n"
        );
    }
}
