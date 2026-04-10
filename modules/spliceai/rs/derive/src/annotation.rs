// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
// Distributed under the terms of the Apache License, Version 2.0.

use dashmap::{DashMap, DashSet};
use flate2::read::MultiGzDecoder;
use genepred::{bed::BedFormat, reader::ReaderError, Bed12, GenePred, Gff, Gtf, Reader, Strand};
use log::{error, info, warn};
use rayon::prelude::*;

use std::{
    collections::HashMap,
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use crate::utils::{build_coordinate_key, reverse_complement_in_place, uppercase_in_place};

/// Extracts splice site donors and acceptors from genomic annotation.
///
/// Reads BED/GTF/GFF annotation file and extracts donor/acceptor dinucleotides
/// at splice sites, along with their genomic coordinates.
///
/// # Arguments
/// * `genome` - Map of chromosome names to sequences
/// * `regions` - Path to annotation file
///
/// # Returns
/// (donors_map, acceptors_map, donor_coords, acceptor_coords)
///
/// # Example
/// ```rust,ignore
/// let (donors, acceptors, ss_donors, ss_acceptors) = get_ss_from_annotation(&genome, path)?;
/// ```
pub fn get_ss_from_annotation(
    genome: &HashMap<Vec<u8>, Vec<u8>>,
    regions: PathBuf,
) -> (
    DashMap<Vec<u8>, usize>,
    DashMap<Vec<u8>, usize>,
    DashSet<Vec<u8>>,
    DashSet<Vec<u8>>,
) {
    let donors = DashMap::new();
    let acceptors = DashMap::new();

    let ss_donors = DashSet::new();
    let ss_acceptors = DashSet::new();

    match detect_region_format(&regions) {
        Some(RegionFormat::Bed) => process_reader::<Bed12>(
            &regions,
            &genome,
            &donors,
            &acceptors,
            &ss_donors,
            &ss_acceptors,
        ),
        Some(RegionFormat::Gtf) => process_reader::<Gtf>(
            &regions,
            &genome,
            &donors,
            &acceptors,
            &ss_donors,
            &ss_acceptors,
        ),
        Some(RegionFormat::Gff) => process_reader::<Gff>(
            &regions,
            &genome,
            &donors,
            &acceptors,
            &ss_donors,
            &ss_acceptors,
        ),
        None => panic!("ERROR: Unsupported file format"),
    }

    (donors, acceptors, ss_donors, ss_acceptors)
}

/// Processes annotation records in parallel using a specific format reader.
///
/// # Arguments
/// * `regions` - Path to annotation file
/// * `genome` - Chromosome sequences
/// * `donors` - Output map for donor dinucleotide counts
/// * `acceptors` - Output map for acceptor dinucleotide counts
/// * `ss_donors` - Set of donor coordinates
/// * `ss_acceptors` - Set of acceptor coordinates
///
/// # Example
/// ```rust,ignore
/// process_reader::<Gtf>(&path, &genome, &donors, &acceptors, &ss_donors, &ss_acceptors);
/// ```
fn process_reader<R>(
    regions: &Path,
    genome: &HashMap<Vec<u8>, Vec<u8>>,
    donors: &DashMap<Vec<u8>, usize>,
    acceptors: &DashMap<Vec<u8>, usize>,
    ss_donors: &DashSet<Vec<u8>>,
    ss_acceptors: &DashSet<Vec<u8>>,
) where
    R: BedFormat + Into<GenePred> + Send,
{
    info!("Processing regions from file {}", regions.display());

    read_regions::<R>(regions)
        .unwrap_or_else(|e| panic!("{}", e))
        // .par_chunks(chunks)
        .par_records()
        .unwrap_or_else(|e| panic!("{}", e))
        .for_each(|record| {
            fill_collectors(
                &record,
                &genome,
                &donors,
                &acceptors,
                &ss_donors,
                &ss_acceptors,
            );
        });
}

/// Fills collectors with splice site information from a single annotation record.
///
/// # Arguments
/// * `record` - Parsed gene prediction record
/// * `genome` - Chromosome sequences
/// * `donors` - Donor dinucleotide counts
/// * `acceptors` - Acceptor dinucleotide counts
/// * `ss_donors` - Donor coordinate set
/// * `ss_acceptors` - Acceptor coordinate set
pub fn fill_collectors(
    record: &Result<GenePred, ReaderError>,
    genome: &HashMap<Vec<u8>, Vec<u8>>,
    donors: &DashMap<Vec<u8>, usize>,
    acceptors: &DashMap<Vec<u8>, usize>,
    ss_donors: &DashSet<Vec<u8>>,
    ss_acceptors: &DashSet<Vec<u8>>,
) {
    let record = match record {
        Ok(record) => record,
        Err(e) => {
            error!("ERROR: Failed to process record: {}", e);
            std::process::exit(1);
        }
    };

    let seq = genome.get(&record.chrom).unwrap_or_else(|| {
        panic!(
            "ERROR: Chromosome {} not found!",
            String::from_utf8_lossy(&record.chrom)
        )
    });

    extract_splice_dinucleotides(&record, seq, donors, acceptors, ss_donors, ss_acceptors);
}

/// Extracts donor and acceptor dinucleotides from introns in a gene record.
///
/// # Arguments
/// * `record` - Gene prediction with introns
/// * `seq` - Chromosome sequence
/// * `donors` - Output donor counts
/// * `acceptors` - Output acceptor counts
/// * `ss_donors` - Donor coordinate set
/// * `ss_acceptors` - Acceptor coordinate set
fn extract_splice_dinucleotides(
    record: &GenePred,
    seq: &[u8],
    donors: &DashMap<Vec<u8>, usize>,
    acceptors: &DashMap<Vec<u8>, usize>,
    ss_donors: &DashSet<Vec<u8>>,
    ss_acceptors: &DashSet<Vec<u8>>,
) {
    let chrom = &record.chrom;

    for (start, end) in record.introns().iter() {
        let mut donor = *start as usize;
        let mut acceptor = *end as usize - 1;

        let strand = record
            .strand
            .unwrap_or_else(|| panic!("ERROR: strand not found for record {:?}!", record.name));

        let mut donor_seq = Vec::new();
        let mut acceptor_seq = Vec::new();

        let Some(donor_end) = donor.checked_add(2) else {
            warn!(
                "Skipping donor outside chromosome bounds: {}:{}",
                String::from_utf8_lossy(chrom),
                donor
            );
            continue;
        };
        if let Some(slice) = seq.get(donor..donor_end) {
            donor_seq.extend_from_slice(slice);
        }

        let Some(acceptor_start) = acceptor.checked_sub(1) else {
            warn!(
                "Skipping acceptor outside chromosome bounds: {}:{}",
                String::from_utf8_lossy(chrom),
                acceptor
            );
            continue;
        };
        let Some(acceptor_end) = acceptor.checked_add(1) else {
            warn!(
                "Skipping acceptor outside chromosome bounds: {}:{}",
                String::from_utf8_lossy(chrom),
                acceptor
            );
            continue;
        };
        if let Some(slice) = seq.get(acceptor_start..acceptor_end) {
            acceptor_seq.extend_from_slice(slice);
        }

        if donor_seq.len() != 2 || acceptor_seq.len() != 2 {
            warn!(
                "Skipping splice site with incomplete dinucleotide sequence: {} donor={} acceptor={}",
                String::from_utf8_lossy(chrom),
                donor,
                acceptor
            );
            continue;
        }

        match &record.strand {
            Some(Strand::Forward) => {}
            Some(Strand::Reverse) => {
                reverse_complement_in_place(&mut donor_seq);
                reverse_complement_in_place(&mut acceptor_seq);

                // WARN: for reverse strand, donor and acceptor are reversed!
                let tmp = donor_seq;
                donor_seq = acceptor_seq;
                acceptor_seq = tmp;

                donor = *end as usize;
                acceptor = *start as usize;
            }
            Some(Strand::Unknown) | None => {}
        }

        uppercase_in_place(&mut donor_seq);
        uppercase_in_place(&mut acceptor_seq);

        // INFO: key for Set is {chrom}:{position}({strand}) as bytes
        let donor_key = build_coordinate_key(chrom, donor, strand);
        let acceptor_key = build_coordinate_key(chrom, acceptor, strand);

        if ss_donors.insert(donor_key) {
            donors.entry(donor_seq).and_modify(|v| *v += 1).or_insert(1);
        }

        if ss_acceptors.insert(acceptor_key) {
            acceptors
                .entry(acceptor_seq)
                .and_modify(|v| *v += 1)
                .or_insert(1);
        }
    }
}

/// Reads genomic regions from an annotation file.
///
/// # Arguments
///
/// - `regions`: Path to the annotation file
///
/// # Example
///
/// ```rust,ignore
/// use genepred::Gtf;
/// let reader = read_regions::<Gtf>(std::path::Path::new("regions.gtf"));
/// ```
fn read_regions<R>(regions: &Path) -> genepred::ReaderResult<Reader<R>>
where
    R: BedFormat + Into<GenePred> + Send,
{
    if is_compressed_path(regions) {
        Reader::<R>::from_path(regions)
    } else {
        Reader::<R>::from_mmap(regions)
    }
}

/// Checks if a file path indicates a compressed file based on extension.
///
/// # Arguments
///
/// - `path`: Path to check for compression extension
///
/// # Example
///
/// ```rust,ignore
/// use std::path::Path;
/// assert!(is_compressed_path(Path::new("file.gz")));
/// assert!(is_compressed_path(Path::new("file.zst")));
/// assert!(!is_compressed_path(Path::new("file.bed")));
/// ```
fn is_compressed_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("gz" | "zst" | "zstd" | "bz2" | "bzip2")
    )
}

/// Supported genomic annotation file formats.
///
/// # Variants
///
/// - `Bed`: BED format (12-column)
/// - `Gtf`: GTF format
/// - `Gff`: GFF format
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::RegionFormat;
///
/// let format = detect_region_format(Path::new("annotations.gtf"));
/// assert_eq!(format, Some(RegionFormat::Gtf));
/// ```
#[derive(Clone, Copy)]
enum RegionFormat {
    Bed,
    Gtf,
    Gff,
}

/// Detects the genomic annotation format from file extension.
///
/// # Arguments
///
/// - `path`: Path to the annotation file
///
/// # Example
///
/// ```rust,ignore
/// use std::path::Path;
///
/// assert_eq!(detect_region_format(Path::new("file.bed")), Some(RegionFormat::Bed));
/// assert_eq!(detect_region_format(Path::new("file.gtf")), Some(RegionFormat::Gtf));
/// assert_eq!(detect_region_format(Path::new("file.gff")), Some(RegionFormat::Gff));
/// assert_eq!(detect_region_format(Path::new("file.gtf.gz")), Some(RegionFormat::Gtf));
/// assert_eq!(detect_region_format(Path::new("file.txt")), None);
/// ```
fn detect_region_format(path: &Path) -> Option<RegionFormat> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("bed") => Some(RegionFormat::Bed),
        Some("gtf") => Some(RegionFormat::Gtf),
        Some("gff") => Some(RegionFormat::Gff),
        Some("gz") => {
            let stem = path.file_stem()?.to_str()?;
            if stem.ends_with(".bed") {
                Some(RegionFormat::Bed)
            } else if stem.ends_with(".gtf") {
                Some(RegionFormat::Gtf)
            } else if stem.ends_with(".gff") {
                Some(RegionFormat::Gff)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Loads genome sequences from a FASTA format file (optionally gzipped).
///
/// # Arguments
///
/// - `f`: Path to the FASTA file (.fa or .fa.gz)
///
/// # Example
///
/// ```rust,ignore
/// use std::path::PathBuf;
///
/// let sequences = from_fa(PathBuf::from("genome.fa"));
/// let sequences = from_fa(PathBuf::from("genome.fa.gz"));
/// let chr1 = sequences.get(b"chr1");
/// ```
pub fn from_fa<F: AsRef<Path> + Debug>(f: F) -> HashMap<Vec<u8>, Vec<u8>> {
    let path = f.as_ref();
    let file = File::open(path)
        .unwrap_or_else(|e| panic!("ERROR: cannot open FASTA {}: {}", path.display(), e));

    let mut reader: Box<dyn BufRead> = match path.extension().and_then(|ext| ext.to_str()) {
        Some("gz") => Box::new(BufReader::new(MultiGzDecoder::new(file))),
        _ => Box::new(BufReader::new(file)),
    };

    let mut acc = HashMap::new();
    let mut line = Vec::new();
    let mut header: Option<Vec<u8>> = None;
    let mut seq = Vec::new();

    loop {
        line.clear();
        let bytes_read = reader
            .read_until(b'\n', &mut line)
            .unwrap_or_else(|e| panic!("ERROR: cannot read FASTA {}: {}", path.display(), e));

        if bytes_read == 0 {
            break;
        }

        if line.ends_with(b"\n") {
            line.pop();
        }

        if line.ends_with(b"\r") {
            line.pop();
        }

        if line.is_empty() {
            continue;
        }

        if line[0] == b'>' {
            if let Some(prev_header) = header.replace(line[1..].to_vec()) {
                acc.insert(prev_header, std::mem::take(&mut seq));
            }
        } else {
            seq.extend_from_slice(&line);
        }
    }

    if let Some(last_header) = header {
        acc.insert(last_header, seq);
    }

    info!("Read {} sequences from file {:#?}", acc.len(), f);

    acc
}
