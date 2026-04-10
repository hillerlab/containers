// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
// Distributed under the terms of the Apache License, Version 2.0.

use bigtools::{utils::reopen::Reopen, BigWigRead};
use dashmap::{DashMap, DashSet};
use log::info;
use rayon::prelude::*;

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

use crate::utils::{build_coordinate_key, reverse_complement_in_place, uppercase_in_place};

/// Type alias: (plus_strand_map, minus_strand_map) for both strands.
pub type SpliceMap = (StrandSpliceMap, StrandSpliceMap);
/// Thread-safe map: chromosome -> set of splice site records.
pub type StrandSpliceMap = DashMap<String, DashSet<Vec<u8>>>;
/// Shared splice scores (unused in current implementation).
pub type SharedSpliceMap = (Option<DashMap<usize, f32>>, Option<DashMap<usize, f32>>);
/// Splice scores: vectors of strand maps for each BigWig type.
pub type SpliceScores = (Vec<StrandSpliceMap>, Vec<StrandSpliceMap>);

/// Fetches and processes splice scores from BigWig files.
///
/// This is a convenience wrapper around `make_splice_map`.
///
/// # Arguments
///
/// * `bigwigs`: Path to the directory containing SpliceAI BigWig files.
/// * `chrs`: Chromosome names to process.
/// * `genome`: Genome sequence keyed by chromosome name.
///
/// # Returns
///
/// * A `Result` containing donor and acceptor `StrandSpliceMap`s.
///
/// # Errors
///
/// Returns `SpliceAiFileError` if the directory does not contain exactly one donor/acceptor
/// file for each plus/minus strand combination.
///
/// # Example
///
/// ```rust,ignore
/// let scores = get_splice_scores(splice_scores, chrs, genome)?;
/// ```
pub fn get_splice_scores<T: AsRef<std::path::Path> + std::fmt::Debug>(
    bigwigs: T,
    chrs: Vec<Vec<u8>>,
    genome: HashMap<Vec<u8>, Vec<u8>>,
) -> Result<(StrandSpliceMap, StrandSpliceMap), SpliceAiFileError> {
    // INFO: DashMap<String, DashSet<Vec<u8>>> -> chr -> [ b'pos -> score' ]
    make_splice_map(bigwigs, chrs, genome)
}

/// Creates `StrandSpliceMap`s for both plus and minus strands by parsing BigWig files.
///
/// This function scans a directory for four SpliceAI BigWig files covering donor/acceptor
/// scores on the plus/minus strands. Filenames are matched case-insensitively and may include
/// arbitrary prefixes or suffixes as long as the basename contains exactly one splice-site token
/// (`donor` or `acceptor`) and exactly one strand token (`plus` or `minus`).
///
/// Supported BigWig extensions are `.bw` and `.bigWig` in any letter casing. Once the four
/// required files are resolved, the function uses `rayon` to parallelize parsing into
/// thread-safe `DashMap`s.
///
/// # Arguments
///
/// * `dir`: The path to the directory containing the SpliceAI BigWig files.
/// * `chrs`: Chromosome names to process.
/// * `genome`: Genome sequence keyed by chromosome name.
///
/// # Returns
///
/// * A `Result` containing donor and acceptor `StrandSpliceMap`s.
///
/// # Errors
///
/// Returns `SpliceAiFileError` if the directory is invalid, if a candidate filename is
/// ambiguous, if a required combination is missing, or if duplicate files match the same
/// donor/acceptor and strand classification.
///
/// # Example
///
/// ```rust,ignore
/// let (donor_scores, acceptor_scores) = make_splice_map(dir, chrs, genome)?;
/// ```
pub fn make_splice_map<T: AsRef<std::path::Path> + std::fmt::Debug>(
    dir: T,
    chrs: Vec<Vec<u8>>,
    genome: HashMap<Vec<u8>, Vec<u8>>,
) -> Result<(StrandSpliceMap, StrandSpliceMap), SpliceAiFileError> {
    let resolved = discover_spliceai_bigwigs(dir.as_ref())?;
    let (plus, minus) = resolved.into_strand_paths();

    info!("Parsing BigWigs...");
    let (plus, minus) = rayon::join(
        || bigwig_to_map(plus, &chrs, Strand::Forward, &genome),
        || bigwig_to_map(minus, &chrs, Strand::Reverse, &genome),
    );

    let [plus_donor, plus_acceptor] = <[_; 2]>::try_from(plus).unwrap();
    let [minus_donor, minus_acceptor] = <[_; 2]>::try_from(minus).unwrap();

    let donor_scores = merge_splice_maps(plus_donor, minus_donor);
    let acceptor_scores = merge_splice_maps(plus_acceptor, minus_acceptor);

    Ok((donor_scores, acceptor_scores))
}

/// Merges two strand-specific splice maps (e.g., plus and minus strands).
///
/// # Arguments
/// * `primary` - First map (e.g., plus strand)
/// * `secondary` - Second map to merge in (e.g., minus strand)
///
/// # Returns
/// Merged map with all entries
///
/// # Example
/// ```rust,ignore
/// let merged = merge_splice_maps(plus_donor, minus_donor);
/// ```
fn merge_splice_maps(primary: StrandSpliceMap, secondary: StrandSpliceMap) -> StrandSpliceMap {
    let merged = primary;

    secondary.into_iter().for_each(|(chr, splice_sites)| {
        if let Some(existing) = merged.get_mut(&chr) {
            splice_sites.into_iter().for_each(|splice_site| {
                existing.insert(splice_site);
            });
        } else {
            merged.insert(chr, splice_sites);
        }
    });

    merged
}

/// DNA strand orientation for splice sites.
///
/// # Variants
/// - Forward: Plus strand (+)
/// - Reverse: Minus strand (-)
///
/// # Example
/// ```rust,ignore
/// use splicing::spliceai::Strand;
///
/// let forward = Strand::Forward;
/// let reverse = Strand::Reverse;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Strand {
    Forward,
    Reverse,
}

impl std::fmt::Display for Strand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strand::Forward => write!(f, "+"),
            Strand::Reverse => write!(f, "-"),
        }
    }
}

impl Strand {
    fn token(self) -> &'static str {
        match self {
            Strand::Forward => "plus",
            Strand::Reverse => "minus",
        }
    }
}

// public enums
/// Splice site type
///
/// This enum is used to store the type of splice site.
///
/// # Example
///
/// ```rust, no_run
/// use splicing::spliceai::SpliceSite;
///
/// let donor = SpliceSite::Donor;
/// let acceptor = SpliceSite::Acceptor;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpliceSite {
    Donor,
    Acceptor,
}

impl std::fmt::Display for SpliceSite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpliceSite::Donor => write!(f, "D"),
            SpliceSite::Acceptor => write!(f, "A"),
        }
    }
}

impl SpliceSite {
    fn token(self) -> &'static str {
        match self {
            SpliceSite::Donor => "donor",
            SpliceSite::Acceptor => "acceptor",
        }
    }
}

/// Classification of a SpliceAI BigWig file by splice site type and strand.
///
/// Combines SpliceSite (donor/acceptor) with Strand (forward/reverse).
///
/// # Fields
/// - `splice_site`: Donor or Acceptor
/// - `strand`: Forward or Reverse
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BigWigClass {
    splice_site: SpliceSite,
    strand: Strand,
}

impl BigWigClass {
    const fn new(splice_site: SpliceSite, strand: Strand) -> Self {
        Self {
            splice_site,
            strand,
        }
    }

    fn label(self) -> &'static str {
        match (self.splice_site, self.strand) {
            (SpliceSite::Donor, Strand::Forward) => "donor plus",
            (SpliceSite::Donor, Strand::Reverse) => "donor minus",
            (SpliceSite::Acceptor, Strand::Forward) => "acceptor plus",
            (SpliceSite::Acceptor, Strand::Reverse) => "acceptor minus",
        }
    }
}

const EXPECTED_BIGWIGS: [BigWigClass; 4] = [
    BigWigClass::new(SpliceSite::Donor, Strand::Forward),
    BigWigClass::new(SpliceSite::Acceptor, Strand::Forward),
    BigWigClass::new(SpliceSite::Donor, Strand::Reverse),
    BigWigClass::new(SpliceSite::Acceptor, Strand::Reverse),
];

/// Resolved paths to the four required SpliceAI BigWig files.
///
/// # Fields
/// - `donor_plus`: Path to donor scores on forward strand
/// - `acceptor_plus`: Path to acceptor scores on forward strand
/// - `donor_minus`: Path to donor scores on reverse strand
/// - `acceptor_minus`: Path to acceptor scores on reverse strand
#[derive(Debug, Clone)]
struct ResolvedBigWigs {
    donor_plus: PathBuf,
    acceptor_plus: PathBuf,
    donor_minus: PathBuf,
    acceptor_minus: PathBuf,
}

impl ResolvedBigWigs {
    fn into_strand_paths(self) -> (Vec<PathBuf>, Vec<PathBuf>) {
        (
            vec![self.donor_plus, self.acceptor_plus],
            vec![self.donor_minus, self.acceptor_minus],
        )
    }
}

/// Errors raised while discovering the four required SpliceAI BigWig files.
#[derive(Debug)]
pub enum SpliceAiFileError {
    InvalidDirectory {
        path: PathBuf,
    },
    ReadDirectory {
        path: PathBuf,
        source: io::Error,
    },
    ReadDirectoryEntry {
        path: PathBuf,
        source: io::Error,
    },
    InvalidFilename {
        path: PathBuf,
        reason: &'static str,
    },
    DuplicateClassification {
        classification: &'static str,
        paths: Vec<PathBuf>,
    },
    MissingClassifications {
        path: PathBuf,
        classifications: Vec<&'static str>,
    },
}

impl std::fmt::Display for SpliceAiFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDirectory { path } => {
                write!(
                    f,
                    "SpliceAI BigWig path '{}' is not a directory",
                    path.display()
                )
            }
            Self::ReadDirectory { path, source } => write!(
                f,
                "failed to read SpliceAI BigWig directory '{}': {}",
                path.display(),
                source
            ),
            Self::ReadDirectoryEntry { path, source } => write!(
                f,
                "failed to read an entry from SpliceAI BigWig directory '{}': {}",
                path.display(),
                source
            ),
            Self::InvalidFilename { path, reason } => write!(
                f,
                "invalid SpliceAI BigWig filename '{}': {}",
                path.display(),
                reason
            ),
            Self::DuplicateClassification {
                classification,
                paths,
            } => write!(
                f,
                "multiple SpliceAI BigWig files matched {}: {}",
                classification,
                display_paths(paths)
            ),
            Self::MissingClassifications {
                path,
                classifications,
            } => write!(
                f,
                "missing required SpliceAI BigWig files in '{}': {}",
                path.display(),
                classifications.join(", ")
            ),
        }
    }
}

impl std::error::Error for SpliceAiFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ReadDirectory { source, .. } => Some(source),
            Self::ReadDirectoryEntry { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Scans a directory and resolves exactly four SpliceAI BigWig files.
///
/// Filenames are matched after removing non-alphanumeric characters and lowercasing the stem, so
/// patterns like `prefixDonorPlusSuffix.bw`, `ACCEPTOR_MINUS.bigWig`, and
/// `run42spliceaiacceptorplusv2.BW` are all supported.
///
/// BigWig files that do not contain any of the required tokens are ignored. Files with partial or
/// ambiguous token matches are rejected with `SpliceAiFileError::InvalidFilename`.
fn discover_spliceai_bigwigs(dir: &Path) -> Result<ResolvedBigWigs, SpliceAiFileError> {
    if !dir.is_dir() {
        return Err(SpliceAiFileError::InvalidDirectory {
            path: dir.to_path_buf(),
        });
    }

    let mut matches: HashMap<BigWigClass, Vec<PathBuf>> = HashMap::new();

    for entry in fs::read_dir(dir).map_err(|source| SpliceAiFileError::ReadDirectory {
        path: dir.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| SpliceAiFileError::ReadDirectoryEntry {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();

        if !path.is_file() || !is_bigwig_path(&path) {
            continue;
        }

        if let Some(classification) = classify_bigwig_path(&path)? {
            matches.entry(classification).or_default().push(path);
        }
    }

    for classification in EXPECTED_BIGWIGS {
        if let Some(paths) = matches.get_mut(&classification) {
            paths.sort();

            if paths.len() > 1 {
                return Err(SpliceAiFileError::DuplicateClassification {
                    classification: classification.label(),
                    paths: paths.clone(),
                });
            }
        }
    }

    let missing = EXPECTED_BIGWIGS
        .iter()
        .copied()
        .filter(|classification| !matches.contains_key(classification))
        .map(BigWigClass::label)
        .collect::<Vec<_>>();

    if !missing.is_empty() {
        return Err(SpliceAiFileError::MissingClassifications {
            path: dir.to_path_buf(),
            classifications: missing,
        });
    }

    Ok(ResolvedBigWigs {
        donor_plus: take_classified_path(
            &mut matches,
            BigWigClass::new(SpliceSite::Donor, Strand::Forward),
        ),
        acceptor_plus: take_classified_path(
            &mut matches,
            BigWigClass::new(SpliceSite::Acceptor, Strand::Forward),
        ),
        donor_minus: take_classified_path(
            &mut matches,
            BigWigClass::new(SpliceSite::Donor, Strand::Reverse),
        ),
        acceptor_minus: take_classified_path(
            &mut matches,
            BigWigClass::new(SpliceSite::Acceptor, Strand::Reverse),
        ),
    })
}

fn take_classified_path(
    matches: &mut HashMap<BigWigClass, Vec<PathBuf>>,
    classification: BigWigClass,
) -> PathBuf {
    matches
        .remove(&classification)
        .and_then(|mut paths| paths.pop())
        .unwrap_or_else(|| {
            panic!(
                "ERROR: missing validated SpliceAI BigWig classification {}",
                classification.label()
            )
        })
}

fn classify_bigwig_path(path: &Path) -> Result<Option<BigWigClass>, SpliceAiFileError> {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| SpliceAiFileError::InvalidFilename {
            path: path.to_path_buf(),
            reason: "filename stem is not valid UTF-8",
        })?;

    classify_bigwig_stem(path, stem)
}

fn classify_bigwig_stem(path: &Path, stem: &str) -> Result<Option<BigWigClass>, SpliceAiFileError> {
    let normalized = normalize_bigwig_stem(stem);
    let has_donor = normalized.contains(SpliceSite::Donor.token());
    let has_acceptor = normalized.contains(SpliceSite::Acceptor.token());
    let has_plus = normalized.contains(Strand::Forward.token());
    let has_minus = normalized.contains(Strand::Reverse.token());

    let site_count = usize::from(has_donor) + usize::from(has_acceptor);
    let strand_count = usize::from(has_plus) + usize::from(has_minus);

    if site_count == 0 && strand_count == 0 {
        return Ok(None);
    }

    if site_count != 1 || strand_count != 1 {
        return Err(SpliceAiFileError::InvalidFilename {
            path: path.to_path_buf(),
            reason: classification_error_reason(site_count, strand_count),
        });
    }

    let splice_site = if has_donor {
        SpliceSite::Donor
    } else {
        SpliceSite::Acceptor
    };
    let strand = if has_plus {
        Strand::Forward
    } else {
        Strand::Reverse
    };

    Ok(Some(BigWigClass::new(splice_site, strand)))
}

fn classification_error_reason(site_count: usize, strand_count: usize) -> &'static str {
    match (site_count, strand_count) {
        (0, 1) | (0, 2) => "missing donor/acceptor token",
        (1, 0) | (2, 0) => "missing plus/minus token",
        (2, 1) => "contains both donor and acceptor tokens",
        (1, 2) => "contains both plus and minus tokens",
        (2, 2) => "contains both donor/acceptor and plus/minus token pairs",
        _ => "must contain exactly one donor/acceptor token and one plus/minus token",
    }
}

fn normalize_bigwig_stem(stem: &str) -> String {
    stem.chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

fn is_bigwig_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("bw") || extension.eq_ignore_ascii_case("bigwig")
        })
}

fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Converts a vector of BigWig files into a vector of thread-safe maps.
///
/// This function is designed to be run in parallel for plus and minus strands. It iterates through
/// the BigWig files, reads the chromosome data, and populates `DashMap`s with scores that
/// meet a certain significance threshold.
///
/// # Arguments
///
/// * `bigwigs`: A `Vec` of paths to the BigWig files (e.g., donor and acceptor).
/// * `chrs`: Chromosome names to be processed.
///
/// # Returns
///
/// * A `Vec<StrandSpliceMap>` where the outer vector corresponds to donor/acceptor sites.
///
/// # Example
///
/// ```rust,ignore
/// let splice_maps = bigwig_to_map(bigwigs, &chrs);
/// ```
fn bigwig_to_map<T: AsRef<std::path::Path> + std::fmt::Debug + Sized + Sync>(
    bigwigs: Vec<T>,
    chrs: &[Vec<u8>],
    strand: Strand,
    genome: &HashMap<Vec<u8>, Vec<u8>>,
) -> Vec<DashMap<String, DashSet<Vec<u8>>>> {
    let total_count = AtomicU32::new(0);
    let rs = Mutex::new(vec![DashMap::new(), DashMap::new()]);

    // [donor, acceptor]
    bigwigs
        .into_par_iter()
        .zip(vec![SpliceSite::Donor, SpliceSite::Acceptor])
        .for_each(|(bigwig, splice_site)| {
            let acc = DashMap::new();

            let bwread = BigWigRead::open_file(bigwig).expect("ERROR: Cannot open BigWig file");
            let chroms: Vec<_> = bwread.chroms().to_vec();
            let splice_site_arc = Arc::new(splice_site);

            chroms.into_par_iter().for_each(|chr| {
                // INFO: per-chromosome map
                let mapper = DashSet::new();
                let local_count = AtomicU32::new(0);

                let mut bwread =
                    BigWigRead::reopen(&bwread).expect("ERROR: Cannot re-open BigWig file");

                if !chrs.contains(&chr.name.as_bytes().to_vec()) {
                    return; // INFO: skip chromosomes not in records
                }

                let name = chr.name.clone();
                let length = chr.length;
                let values = bwread
                    .values(&name, 0, length)
                    .expect("ERROR: Cannot read values from BigWig!");

                values.into_iter().enumerate().for_each(|(i, v)| {
                    if v >= crate::cli::SPLICE_AI_SCORE_RECOVERY_THRESHOLD {
                        let pos = i;
                        let sequence = genome
                            .get(name.as_bytes())
                            .unwrap_or_else(|| panic!("ERROR: cannot chr {} in genome!", name));
                        let Some(dnt) = extract_dinucleotide(sequence, pos, splice_site, strand)
                        else {
                            log::warn!(
                                "Skipping SpliceAI site outside chromosome bounds: {}:{}",
                                name,
                                pos
                            );
                            return;
                        };

                        let line =
                            Minisplice::new(&name, pos, strand, splice_site_arc.clone(), v, dnt);
                        mapper.insert(line.as_bytes());
                        local_count.fetch_add(1, Ordering::Relaxed);
                    }
                });

                acc.insert(name, mapper);
                total_count.fetch_add(local_count.load(Ordering::Relaxed), Ordering::Relaxed);
            });

            let mut guard = rs.lock().expect("ERROR: Cannot lock mutex");
            match splice_site {
                SpliceSite::Donor => guard[0] = acc,
                SpliceSite::Acceptor => guard[1] = acc,
            }
        });

    info!(
        "Parsed and combined {} significant splicing scores from BigWigs!",
        total_count.load(Ordering::Relaxed)
    );

    rs.into_inner()
        .expect("ERROR: Cannot unwrap collection of SpliceAI scores!")
}

/// Extracts the dinucleotide at a splice site from the genome sequence.
///
/// Position depends on splice site type and strand.
///
/// # Arguments
/// * `sequence` - Chromosome sequence
/// * `pos` - Genomic position
/// * `splice_site` - Donor or Acceptor
/// * `strand` - Forward or Reverse
///
/// # Returns
/// Dinucleotide bytes or None if out of bounds
///
/// # Example
/// ```rust,ignore
/// let dnt = extract_dinucleotide(&seq, 100, SpliceSite::Donor, Strand::Forward);
/// ```
fn extract_dinucleotide(
    sequence: &[u8],
    pos: usize,
    splice_site: SpliceSite,
    strand: Strand,
) -> Option<Vec<u8>> {
    let mut dinucleotide = match (splice_site, strand) {
        (SpliceSite::Donor, Strand::Forward) => {
            let end = pos.checked_add(2)?;
            sequence.get(pos..end)?.to_vec()
        }
        (SpliceSite::Donor, Strand::Reverse) => {
            let start = pos.checked_sub(1)?;
            let end = pos.checked_add(1)?;
            sequence.get(start..end)?.to_vec()
        }
        (SpliceSite::Acceptor, Strand::Forward) => {
            let start = pos.checked_sub(1)?;
            let end = pos.checked_add(1)?;
            sequence.get(start..end)?.to_vec()
        }
        (SpliceSite::Acceptor, Strand::Reverse) => {
            let end = pos.checked_add(2)?;
            sequence.get(pos..end)?.to_vec()
        }
    };

    match strand {
        Strand::Forward => {}
        Strand::Reverse => {
            reverse_complement_in_place(&mut dinucleotide);
        }
    }

    uppercase_in_place(&mut dinucleotide);

    Some(dinucleotide)
}

/// Minimal splice record for SpliceAI data.
///
/// Format: chr:pos(strand)\tD/A\tscore\tdinucleotide
///
/// # Fields
/// - `chr`: Chromosome name
/// - `position`: Genomic position
/// - `strand`: Forward or Reverse
/// - `splice_site`: Donor or Acceptor
/// - `score`: SpliceAI score (0-1)
/// - `dinucleotide`: Splice site dinucleotide
#[derive(Debug, Clone, PartialEq)]
pub struct Minisplice {
    pub chr: String,
    pub position: usize,
    pub strand: Strand,
    pub splice_site: Arc<SpliceSite>,
    pub score: f32,
    pub dinucleotide: Vec<u8>,
}

impl Minisplice {
    /// Creates a new Minisplice record.
    ///
    /// # Arguments
    /// * `chr` - Chromosome name
    /// * `position` - Genomic position
    /// * `strand` - Forward or Reverse
    /// * `splice_site` - Donor or Acceptor (as Arc)
    /// * `score` - SpliceAI score
    /// * `dinucleotide` - Splice site dinucleotide
    ///
    /// # Example
    /// ```rust,ignore
    /// let ms = Minisplice::new("chr1", 100, Strand::Forward, Arc::new(SpliceSite::Donor), 0.95, b"GT".to_vec());
    /// ```
    pub fn new(
        chr: &str,
        position: usize,
        strand: Strand,
        splice_site: Arc<SpliceSite>,
        score: f32,
        dinucleotide: Vec<u8>,
    ) -> Self {
        Self {
            chr: chr.to_string(),
            position,
            strand,
            splice_site,
            score,
            dinucleotide,
        }
    }

    /// Converts to tab-separated byte format.
    ///
    /// # Example
    /// ```rust,ignore
    /// let bytes = minisplice.as_bytes();
    /// ```
    pub fn as_bytes(&self) -> Vec<u8> {
        format!(
            "{}:{}({})\t{}\t{}\t{}",
            self.chr,
            self.position,
            self.strand,
            self.splice_site,
            self.score,
            std::str::from_utf8(&self.dinucleotide).unwrap()
        )
        .into_bytes()
    }
}

impl std::fmt::Display for Minisplice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.chr,
            self.position,
            self.strand,
            self.splice_site,
            self.score,
            std::str::from_utf8(&self.dinucleotide).unwrap()
        )
    }
}

/// Parsed SpliceAI record with extracted fields.
///
/// # Fields
/// - `coordinate_key`: Full coordinate string "chr:pos(strand)"
/// - `chr`: Chromosome bytes
/// - `position`: Genomic position
/// - `strand`: Forward or Reverse
/// - `splice_site`: Donor or Acceptor
/// - `score`: SpliceAI score (0-1)
/// - `dinucleotide`: Splice site dinucleotide
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedSpliceAiRecord {
    pub coordinate_key: Vec<u8>,
    pub chr: Vec<u8>,
    pub position: usize,
    pub strand: Strand,
    pub splice_site: SpliceSite,
    pub score: f32,
    pub dinucleotide: Vec<u8>,
}

/// Parses a tab-separated SpliceAI record line.
///
/// # Arguments
/// * `line` - Tab-separated record: coord\tD/A\tscore\tdinucleotide
///
/// # Returns
/// Parsed record or None if invalid
///
/// # Example
/// ```rust,ignore
/// let parsed = parse_spliceai_record(b"chr1:100(+)\tD\t0.95\tGT")?;
/// ```
pub fn parse_spliceai_record(line: &[u8]) -> Option<ParsedSpliceAiRecord> {
    let mut fields = line.split(|byte| *byte == b'\t');
    let coordinate_field = fields.next()?;
    let splice_site_field = fields.next()?;
    let score = std::str::from_utf8(fields.next()?)
        .ok()?
        .parse::<f32>()
        .ok()?;
    let mut dinucleotide = fields.next()?.to_vec();

    if fields.next().is_some() {
        return None;
    }

    uppercase_in_place(&mut dinucleotide);

    let (chr, position, strand) = parse_coordinate_field(coordinate_field)?;
    let splice_site = parse_splice_site(splice_site_field)?;
    let coordinate_key = build_coordinate_key(&chr, position, strand);

    Some(ParsedSpliceAiRecord {
        coordinate_key,
        chr,
        position,
        strand,
        splice_site,
        score,
        dinucleotide,
    })
}

/// Parses coordinate field "chr:position(strand)" into components.
///
/// # Example
/// ```rust,ignore
/// let (chr, pos, strand) = parse_coordinate_field(b"chr1:12345(+)")?;
/// ```
fn parse_coordinate_field(field: &[u8]) -> Option<(Vec<u8>, usize, Strand)> {
    let strand_open = field.iter().rposition(|byte| *byte == b'(')?;
    let strand_close = field.last().copied()?;
    if strand_close != b')' || strand_open == 0 || strand_open + 2 != field.len() - 1 {
        return None;
    }

    let strand = match field.get(strand_open + 1)? {
        b'+' => Strand::Forward,
        b'-' => Strand::Reverse,
        _ => return None,
    };
    let chr_pos = field.get(..strand_open)?;
    let separator = chr_pos.iter().rposition(|byte| *byte == b':')?;
    let chr = chr_pos.get(..separator)?.to_vec();
    let position = std::str::from_utf8(chr_pos.get(separator + 1..)?)
        .ok()?
        .parse::<usize>()
        .ok()?;

    Some((chr, position, strand))
}

/// Parses splice site type field (D = Donor, A = Acceptor).
///
/// # Example
/// ```rust,ignore
/// let site = parse_splice_site(b"D")?; // Some(SpliceSite::Donor)
/// ```
fn parse_splice_site(field: &[u8]) -> Option<SpliceSite> {
    match field {
        b"D" => Some(SpliceSite::Donor),
        b"A" => Some(SpliceSite::Acceptor),
        _ => None,
    }
}

/// Writes the results to files.
///
/// This function takes two vectors of `StrandSpliceMap`s and writes the results to files.
/// It first writes the plus strand scores to a file named `prefix.plus.tsv`, and then
/// writes the minus strand scores to a file named `prefix.minus.tsv`.
///
/// # Arguments
///
/// * `plus_scores`: A `Vec<StrandSpliceMap>` containing the plus strand scores.
/// * `minus_scores`: A `Vec<StrandSpliceMap>` containing the minus strand scores.
/// * `prefix`: A `String` representing the prefix for the output files.
///
/// # Example
///
/// ```rust,ignore
/// write_results(plus_scores, minus_scores, prefix);
/// ```
pub fn write_results(
    plus_scores: Vec<StrandSpliceMap>,
    minus_scores: Vec<StrandSpliceMap>,
    prefix: String,
) {
    let mut plus_file = BufWriter::new(
        File::create(format!("{}.plus.tsv", prefix))
            .unwrap_or_else(|e| panic!("ERROR: Cannot create plus.tsv file -> {e}!")),
    );
    let mut minus_file = BufWriter::new(
        File::create(format!("{}.minus.tsv", prefix))
            .unwrap_or_else(|e| panic!("ERROR: Cannot create minus.tsv file -> {e}!")),
    );

    plus_scores.iter().for_each(|entry| {
        entry.iter().for_each(|val| {
            let (chr, scores) = val.pair();
            info!(
                "Writing {} scores for {} in forward strand...",
                scores.len(),
                chr
            );
            scores.iter().for_each(|score| {
                plus_file
                    .write_all(&score)
                    .unwrap_or_else(|e| panic!("ERROR: Cannot write to plus.tsv file -> {e}!"));
                plus_file
                    .write_all(b"\n")
                    .unwrap_or_else(|e| panic!("ERROR: Cannot write to plus.tsv file -> {e}!"));
            });
        });
    });
    plus_file.flush().unwrap();

    minus_scores.iter().for_each(|entry| {
        entry.iter().for_each(|val| {
            let (chr, scores) = val.pair();
            info!(
                "Writing {} scores for {} in reverse strand...",
                scores.len(),
                chr
            );
            scores.iter().for_each(|score| {
                minus_file
                    .write_all(&score)
                    .unwrap_or_else(|e| panic!("ERROR: Cannot write to minus.tsv file -> {e}!"));
                minus_file
                    .write_all(b"\n")
                    .unwrap_or_else(|e| panic!("ERROR: Cannot write to minus.tsv file -> {e}!"));
            });
        });
    });
    minus_file.flush().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        env, fs,
        fs::File,
        time::{SystemTime, UNIX_EPOCH},
    };

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            let unique = format!(
                "splicing-spliceai-test-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            let path = env::temp_dir().join(unique);
            fs::create_dir_all(&path).unwrap();

            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }

        fn touch(&self, filename: &str) {
            File::create(self.path.join(filename)).unwrap();
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn parse_spliceai_record_extracts_expected_fields() {
        let parsed = parse_spliceai_record(b"chr1:42(-)\tA\t0.75\tag").unwrap();

        assert_eq!(parsed.coordinate_key, b"chr1:42(-)".to_vec());
        assert_eq!(parsed.chr, b"chr1".to_vec());
        assert_eq!(parsed.position, 42);
        assert_eq!(parsed.strand, Strand::Reverse);
        assert_eq!(parsed.splice_site, SpliceSite::Acceptor);
        assert_eq!(parsed.score, 0.75);
        assert_eq!(parsed.dinucleotide, b"AG".to_vec());
    }

    #[test]
    fn classify_bigwig_stem_accepts_affixes_casing_and_extensions() {
        let donor_plus = classify_bigwig_path(Path::new("run42spliceAiDONORPLUSv2.bigWig"))
            .unwrap()
            .unwrap();
        let acceptor_minus = classify_bigwig_path(Path::new("prefix_acceptor-minus_suffix.BW"))
            .unwrap()
            .unwrap();

        assert_eq!(
            donor_plus,
            BigWigClass::new(SpliceSite::Donor, Strand::Forward)
        );
        assert_eq!(
            acceptor_minus,
            BigWigClass::new(SpliceSite::Acceptor, Strand::Reverse)
        );
    }

    #[test]
    fn classify_bigwig_stem_rejects_partial_matches() {
        let error = classify_bigwig_path(Path::new("spliceai_donor_only.bw")).unwrap_err();

        match error {
            SpliceAiFileError::InvalidFilename { reason, .. } => {
                assert_eq!(reason, "missing plus/minus token");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn discover_spliceai_bigwigs_resolves_mixed_extensions_and_affixes() {
        let dir = TempDir::new();
        dir.touch("run42DonorPlusv2.BW");
        dir.touch("prefix_acceptor-plus_suffix.bigWig");
        dir.touch("sampleDONORMINUSextra.bw");
        dir.touch("xxAcceptorMinusyy.bigwig");

        let resolved = discover_spliceai_bigwigs(dir.path()).unwrap();

        assert_eq!(
            resolved.donor_plus.file_name().unwrap(),
            "run42DonorPlusv2.BW"
        );
        assert_eq!(
            resolved.acceptor_plus.file_name().unwrap(),
            "prefix_acceptor-plus_suffix.bigWig"
        );
        assert_eq!(
            resolved.donor_minus.file_name().unwrap(),
            "sampleDONORMINUSextra.bw"
        );
        assert_eq!(
            resolved.acceptor_minus.file_name().unwrap(),
            "xxAcceptorMinusyy.bigwig"
        );
    }

    #[test]
    fn discover_spliceai_bigwigs_rejects_duplicate_classifications() {
        let dir = TempDir::new();
        dir.touch("sample_donor_plus_v1.bw");
        dir.touch("sample_donor_plus_v2.bigwig");
        dir.touch("sample_acceptor_plus.bw");
        dir.touch("sample_donor_minus.bw");
        dir.touch("sample_acceptor_minus.bw");

        let error = discover_spliceai_bigwigs(dir.path()).unwrap_err();

        match error {
            SpliceAiFileError::DuplicateClassification {
                classification,
                paths,
            } => {
                assert_eq!(classification, "donor plus");
                assert_eq!(paths.len(), 2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn discover_spliceai_bigwigs_rejects_missing_classifications() {
        let dir = TempDir::new();
        dir.touch("sample_donor_plus.bw");
        dir.touch("sample_acceptor_plus.bw");
        dir.touch("sample_acceptor_minus.bw");

        let error = discover_spliceai_bigwigs(dir.path()).unwrap_err();

        match error {
            SpliceAiFileError::MissingClassifications {
                classifications, ..
            } => {
                assert_eq!(classifications, vec!["donor minus"]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
