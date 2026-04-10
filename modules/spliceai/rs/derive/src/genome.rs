// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
// Distributed under the terms of the Apache License, Version 2.0.

use flate2::read::MultiGzDecoder;
use log::info;
use memmap2::Mmap;
use rayon::prelude::*;

use std::{
    collections::HashMap,
    fmt::{self, Debug},
    fs::File,
    io::{self, BufWriter, Read, Write},
    path::Path,
};

use crate::utils::reverse_complement_dinucleotide;

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
const TWOBIT_MAGIC: [u8; 4] = [0x1a, 0x41, 0x27, 0x43];
const TWOBIT_MAGIC_REV: [u8; 4] = [0x43, 0x27, 0x41, 0x1a];

/// Counts a dinucleotide pair and its reverse complement.
///
/// # Arguments
/// * `local` - Map to update
/// * `pair` - 2-byte dinucleotide slice
///
/// # Example
/// ```rust,ignore
/// let mut counts = HashMap::new();
/// count_dinucleotide_pair(&mut counts, b"AC");
/// ```
fn count_dinucleotide_pair(local: &mut HashMap<Vec<u8>, usize>, pair: &[u8]) {
    let forward = vec![pair[0].to_ascii_uppercase(), pair[1].to_ascii_uppercase()];
    *local.entry(forward).or_insert(0) += 1;

    let reverse = reverse_complement_dinucleotide(pair);
    *local.entry(reverse).or_insert(0) += 1;
}

/// Extracts chromosome sizes from a sequence file.
///
/// This function processes a sequence file (from file path or stdin) and returns
/// a vector of tuples containing chromosome names and their corresponding sizes.
/// Input format is detected by content (FASTA header or 2bit signature).
///
/// # Arguments
///
/// * `sequence` - Path to a FASTA/2bit file or "-" for stdin
///
/// # Returns
///
/// Vector of (chromosome_name, size) pairs
///
/// # Examples
///
/// ```ignore
/// let sizes = get_sizes("genome.fa")?;
/// // sizes: [("chr1", 248956422), ("chr2", 242193529), ...]
/// ```
pub fn get_dinucleotide_count<T: AsRef<Path> + Debug>(
    sequence: T,
) -> Result<(HashMap<Vec<u8>, Vec<u8>>, HashMap<Vec<u8>, usize>), SpliceCountError> {
    let path = sequence.as_ref();
    let data = if is_stdin(path) {
        from_stdin()?
    } else {
        from_file(path)?
    };

    from_bytes(data.as_ref())
}

/// Error types for the chromsize application.
///
/// This enum represents all possible error conditions that can occur
/// during sequence processing and chromosome size calculation.
#[derive(Debug)]
pub enum SpliceCountError {
    /// I/O related errors from file operations
    Io(io::Error),
    /// Empty input data (no content to process)
    EmptyInput,
    /// Invalid input format or unsupported content
    InvalidInput(String),
    /// Invalid FASTA format with descriptive message
    InvalidFasta(String),
}

impl fmt::Display for SpliceCountError {
    /// Formats the error for display purposes.
    ///
    /// Provides human-readable error messages for each error variant.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpliceCountError::Io(err) => write!(f, "I/O error: {}", err),
            SpliceCountError::EmptyInput => write!(f, "Input is empty"),
            SpliceCountError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            SpliceCountError::InvalidFasta(msg) => write!(f, "Invalid FASTA: {}", msg),
        }
    }
}

impl std::error::Error for SpliceCountError {
    /// Returns the underlying error source for error chaining.
    ///
    /// Only IO errors have an underlying source, other errors are standalone.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SpliceCountError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for SpliceCountError {
    /// Converts I/O errors to SpliceCountError.
    ///
    /// This enables the `?` operator for I/O operations throughout the codebase.
    fn from(err: io::Error) -> Self {
        SpliceCountError::Io(err)
    }
}

/// Represents different types of input data storage.
///
/// This enum allows handling both memory-mapped files and owned buffers,
/// enabling efficient processing of different input sources.
enum InputData {
    /// Memory-mapped file data (zero-copy for regular files)
    Mmap(Mmap),
    /// Owned byte buffer (for stdin or decompressed data)
    Owned(Vec<u8>),
}

/// Detected input format (FASTA or 2bit).
///
/// # Variants
/// - Fasta: Plain or gzipped FASTA format
/// - TwoBit: 2bit binary format
///
/// # Example
/// ```rust,ignore
/// let format = sniff_format(b">chr1\nATGC");
/// assert_eq!(format, Some(InputFormat::Fasta));
/// ```
enum InputFormat {
    Fasta,
    TwoBit,
}

impl AsRef<[u8]> for InputData {
    /// Provides read access to the underlying byte data.
    ///
    /// This enables uniform processing regardless of whether the data
    /// comes from a memory-mapped file or an owned buffer.
    fn as_ref(&self) -> &[u8] {
        match self {
            InputData::Mmap(m) => m.as_ref(),
            InputData::Owned(b) => b.as_slice(),
        }
    }
}

/// Checks if the path represents stdin input.
///
/// Returns true if the path is "-" which conventionally means read from stdin.
///
/// # Arguments
/// * `path` - Path to check
///
/// # Example
/// ```rust,ignore
/// assert!(is_stdin(std::path::Path::new("-")));
/// assert!(!is_stdin(std::path::Path::new("file.fa")));
/// ```
fn is_stdin(path: &Path) -> bool {
    path == Path::new("-")
}

/// Reads data from standard input with optional gzip decompression.
///
/// Reads all data from stdin, detects if gzip-compressed, and decompresses if needed.
///
/// # Returns
/// InputData::Owned containing the raw or decompressed input data
///
/// # Example
/// ```rust,ignore
/// let input = from_stdin()?;
/// ```
fn from_stdin() -> Result<InputData, SpliceCountError> {
    let mut buffer = Vec::with_capacity(1024 * 1024);
    let mut handle = io::stdin().lock();
    handle.read_to_end(&mut buffer)?;

    if buffer.is_empty() {
        return Err(SpliceCountError::EmptyInput);
    }

    if is_gzip(&buffer) {
        let decompressed = decompress_gzip(&buffer)?;

        if decompressed.is_empty() {
            return Err(SpliceCountError::EmptyInput);
        }

        Ok(InputData::Owned(decompressed))
    } else {
        Ok(InputData::Owned(buffer))
    }
}

/// Reads data from a file with memory mapping and optional gzip decompression.
///
/// Attempts to memory-map the file for efficient access. If gzip-compressed,
/// decompresses the entire content into an owned buffer.
///
/// # Arguments
/// * `path` - Path to the input file
///
/// # Returns
/// InputData::Mmap for uncompressed files, InputData::Owned for compressed files
///
/// # Example
/// ```rust,ignore
/// let data = from_file(std::path::Path::new("genome.fa"))?;
/// ```
fn from_file(path: &Path) -> Result<InputData, SpliceCountError> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    if mmap.is_empty() {
        return Err(SpliceCountError::EmptyInput);
    }

    if is_gzip(&mmap) {
        let decompressed = decompress_gzip(&mmap)?;
        if decompressed.is_empty() {
            return Err(SpliceCountError::EmptyInput);
        }
        Ok(InputData::Owned(decompressed))
    } else {
        Ok(InputData::Mmap(mmap))
    }
}

/// Reads 2bit data and extracts sequences with dinucleotide counts.
///
/// Opens and reads 2bit data using the `twobit` crate, extracts chromosome sequences
/// and counts dinucleotides in parallel.
///
/// # Arguments
/// * `twobit` - Raw 2bit data to process
///
/// # Returns
/// (genome_map, dinucleotide_counts) or error
///
/// # Example
/// ```rust,ignore
/// let (genome, counts) = from_2bit(&data)?;
/// ```
fn from_2bit(
    twobit: &[u8],
) -> Result<(HashMap<Vec<u8>, Vec<u8>>, HashMap<Vec<u8>, usize>), SpliceCountError> {
    let mut genome = twobit::TwoBitFile::from_buf(twobit)
        .map_err(|e| SpliceCountError::InvalidInput(format!("Invalid 2bit data: {e}")))?;

    // Read all sequences sequentially first
    let mut sequences: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
    genome.chrom_names().iter().for_each(|chr| {
        let seq = genome
            .read_sequence(chr, ..)
            .unwrap_or_else(|e| panic!("ERROR: failed to read sequence for {}: {}", chr, e))
            .into_bytes();

        sequences.insert(chr.as_bytes().to_vec(), seq);
    });

    // Now parallelize over the owned sequences
    let totals = sequences
        .par_iter()
        .map(|(_chr, seq)| {
            let mut local: HashMap<Vec<u8>, usize> = HashMap::new();
            seq.windows(2)
                .filter(|pair| {
                    pair[0] != b'N' && pair[1] != b'N' && pair[0] != b'n' && pair[1] != b'n'
                })
                .for_each(|pair| {
                    count_dinucleotide_pair(&mut local, pair);
                });
            local
        })
        .reduce(HashMap::new, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            a
        });

    info!("Read {} sequences from .2bit", sequences.len());

    Ok((sequences, totals))
}

/// Processes raw bytes to extract chromosome sizes by detecting format.
///
/// This function acts as a dispatcher that determines the input format
/// (FASTA or 2bit) and routes the data to the appropriate processor.
/// It returns an error if the format is not recognized.
///
/// # Arguments
///
/// * `data` - Raw byte data to be processed
///
/// # Returns
///
/// Vector of (chromosome_name, size) pairs or error for unsupported format
///
/// # Examples
///
/// ```ignore
/// let data = b">chr1\nATGC\n>chr2\nGCTA";
/// let sizes = sizes_from_bytes(data)?;
/// // sizes: [("chr1", 4), ("chr2", 4)]
/// ```
fn from_bytes(
    data: &[u8],
) -> Result<(HashMap<Vec<u8>, Vec<u8>>, HashMap<Vec<u8>, usize>), SpliceCountError> {
    match sniff_format(data) {
        Some(InputFormat::TwoBit) => from_2bit(data),
        Some(InputFormat::Fasta) => from_fa(data),
        None => Err(SpliceCountError::InvalidInput(
            "Input format not recognized (expected FASTA or 2bit)".to_string(),
        )),
    }
}

/// Detects the input format by examining file magic bytes and patterns.
///
/// This function identifies whether the data is in 2bit format (by checking
/// the 4-byte magic signature) or FASTA format (by checking for the '>' header
/// character). Returns None for unrecognized formats.
///
/// # Arguments
///
/// * `data` - Raw byte data to examine for format identification
///
/// # Returns
///
/// Some(InputFormat) if format is recognized, None otherwise
fn sniff_format(data: &[u8]) -> Option<InputFormat> {
    if data.len() >= TWOBIT_MAGIC.len()
        && (data[..TWOBIT_MAGIC.len()] == TWOBIT_MAGIC
            || data[..TWOBIT_MAGIC_REV.len()] == TWOBIT_MAGIC_REV)
    {
        return Some(InputFormat::TwoBit);
    }

    if data.first() == Some(&b'>') {
        return Some(InputFormat::Fasta);
    }

    None
}

/// Detects if data is gzip-compressed by checking magic bytes (0x1f 0x8b).
///
/// # Arguments
/// * `bytes` - Byte slice to check
///
/// # Returns
/// true if the data appears to be gzip-compressed
///
/// # Example
/// ```rust,ignore
/// assert!(is_gzip(&[0x1f, 0x8b, 0x00]));
/// assert!(!is_gzip(b"ATGC"));
/// ```
fn is_gzip(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == GZIP_MAGIC[0] && bytes[1] == GZIP_MAGIC[1]
}

/// Decompresses gzip data into a byte vector.
///
/// Handles potentially large gzip files efficiently using a reasonably sized buffer.
///
/// # Arguments
/// * `data` - Compressed gzip data
///
/// # Returns
/// Decompressed data in a new Vec<u8>
///
/// # Example
/// ```rust,ignore
/// let decompressed = decompress_gzip(&compressed_data)?;
/// ```
fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>, SpliceCountError> {
    let mut decoder = MultiGzDecoder::new(data);
    let mut buffer = Vec::with_capacity(8 * 1024 * 1024);
    decoder.read_to_end(&mut buffer)?;
    Ok(buffer)
}

/// Processes FASTA data to extract sequences and dinucleotide counts in parallel.
///
/// Validates FASTA format and uses parallel processing to handle multiple chromosomes.
///
/// # Arguments
/// * `data` - Raw FASTA data as bytes
///
/// # Returns
/// (genome_map, dinucleotide_counts)
///
/// # Example
/// ```rust,ignore
/// let (genome, counts) = from_fa(b">chr1\nATGC\n>chr2\nGCTA")?;
/// ```
fn from_fa(
    data: &[u8],
) -> Result<(HashMap<Vec<u8>, Vec<u8>>, HashMap<Vec<u8>, usize>), SpliceCountError> {
    if data.is_empty() {
        return Err(SpliceCountError::EmptyInput);
    }

    if data[0] != b'>' {
        return Err(SpliceCountError::InvalidFasta(
            "Input does not start with '>'".to_string(),
        ));
    }

    let totals = data
        .par_split(|&c| c == b'>')
        .filter(|chunk| !chunk.is_empty())
        .map(process_record)
        .map(|x| x.unwrap())
        .reduce(
            || (HashMap::new(), HashMap::new()),
            |mut a, b| {
                for (k, v) in b.1.iter() {
                    *a.1.entry(k.clone()).or_insert(0) += v;
                }

                for (k, v) in b.0.iter() {
                    *a.0.entry(k.clone()).or_insert(Vec::new()) = v.clone();
                }

                (a.0, a.1)
            },
        );

    info!("Read {} sequences from FASTA", totals.0.len());

    Ok(totals)
}

/// Processes a single FASTA record to extract header and count sequence length.
///
/// This function parses a FASTA record (header + sequence), validates the format,
/// and counts the total number of valid sequence characters while enforcing
/// Parses a single FASTA record to extract header and sequence, counts dinucleotides.
///
/// Validates format and enforces strict FASTA compliance.
///
/// # Arguments
/// * `chunk` - FASTA record data without the leading '>' character
///
/// # Returns
/// (header_map, dinucleotide_counts)
///
/// # Example
/// ```rust,ignore
/// let (header, counts) = process_record(b"chr1\nATGC")?;
/// ```
fn process_record(
    chunk: &[u8],
) -> Result<(HashMap<Vec<u8>, Vec<u8>>, HashMap<Vec<u8>, usize>), SpliceCountError> {
    let Some(stop) = memchr::memchr(b'\n', chunk) else {
        return Err(SpliceCountError::InvalidFasta(
            "Record header is not terminated by a newline".to_string(),
        ));
    };

    let header = std::str::from_utf8(&chunk[..stop])
        .map_err(|_| SpliceCountError::InvalidFasta("Record header is not UTF-8".to_string()))?
        .trim();

    if header.is_empty() {
        return Err(SpliceCountError::InvalidFasta(
            "Record has an empty header".to_string(),
        ));
    }

    let raw = &chunk[stop + 1..];

    if memchr::memchr2(b' ', b'\t', raw).is_some() {
        return Err(SpliceCountError::InvalidFasta(
            "Record contains whitespace inside sequence data".to_string(),
        ));
    }

    if memchr::memchr(b'>', raw).is_some() {
        return Err(SpliceCountError::InvalidFasta(
            "Record contains '>' inside sequence data".to_string(),
        ));
    }

    let sequence: Vec<u8> = raw
        .iter()
        .copied()
        .filter(|&b| b != b'\n' && b != b'\r')
        .collect();

    let mut dinucleotides: HashMap<Vec<u8>, usize> = HashMap::new();
    for pair in sequence.windows(2) {
        if pair[0] == b'N' || pair[1] == b'N' || pair[0] == b'n' || pair[1] == b'n' {
            continue;
        }
        count_dinucleotide_pair(&mut dinucleotides, pair);
    }

    let mut chr = HashMap::new();
    chr.insert(header.as_bytes().to_vec(), sequence);

    Ok((chr, dinucleotides))
}

/// Writes chromosome sizes to a tab-delimited file.
///
/// Writes to a two-column format: chromosome_name (tab) size
///
/// # Arguments
///
/// * `sizes` - Vector of (chromosome_name, size) tuples
/// * `outdir` - Output directory path
/// * `prefix` - Output file prefix
///
/// # Examples
///
/// ```ignore
/// let sizes = vec![("chr1".to_string(), 248956422), ("chr2".to_string(), 242193529)];
/// writer(&sizes, "output/", "chrom.sizes")?;
/// // Output file contains:
/// // chr1    248956422
/// // chr2    242193529
/// ```
pub fn writer<T>(sizes: &[(String, u64)], outdir: T, prefix: String) -> Result<(), SpliceCountError>
where
    T: AsRef<Path> + Debug,
{
    std::fs::create_dir_all(&outdir)?;
    let file = File::create(outdir.as_ref().join(prefix))?;
    let mut writer = BufWriter::with_capacity(64 * 1024, file);

    for (k, v) in sizes.iter() {
        writeln!(writer, "{}\t{}", k, v)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_dinucleotide_pair_tracks_reverse_complement() {
        let mut counts = HashMap::new();
        count_dinucleotide_pair(&mut counts, b"AC");

        assert_eq!(counts.get(b"AC".as_slice()), Some(&1));
        assert_eq!(counts.get(b"GT".as_slice()), Some(&1));
    }

    #[test]
    fn count_dinucleotide_pair_double_counts_palindromes() {
        let mut counts = HashMap::new();
        count_dinucleotide_pair(&mut counts, b"AT");

        assert_eq!(counts.get(b"AT".as_slice()), Some(&2));
    }
}
