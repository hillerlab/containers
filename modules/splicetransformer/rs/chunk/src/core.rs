// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
// Distributed under the terms of the Apache License, Version 2.0.

use flate2::{read::MultiGzDecoder, write::GzEncoder, Compression};
use log::info;
use rayon::prelude::*;
use twobit::TwoBitFile;

use std::{
    collections::HashMap,
    fs::{create_dir_all, File},
    io::{BufRead, BufReader, BufWriter, Cursor, Read, Seek, Write},
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
};

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
const TWOBIT_MAGIC: [u8; 4] = [0x43, 0x27, 0x41, 0x1a];
const TWOBIT_REV_MAGIC: [u8; 4] = [0x1a, 0x41, 0x27, 0x43];

/// Configuration for chunking genome sequences.
///
/// # Arguments
///
/// - `chunk_size`: Target chunk size in bases
/// - `flank_size`: Flank size added to both ends
/// - `min_contig_size`: Minimum contig size to process
/// - `gzip`: Whether to compress output as .fa.gz
/// - `outdir`: Output directory path
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::Config;
///
/// let config = Config {
///     chunk_size: 100_000,
///     flank_size: 100,
///     min_contig_size: 100,
///     gzip: false,
///     outdir: std::path::PathBuf::from("."),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct Config {
    pub chunk_size: u32,
    pub flank_size: u32,
    pub min_contig_size: u32,
    pub gzip: bool,
    pub outdir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_size: 100_000,
            flank_size: 100,
            min_contig_size: 100,
            gzip: false,
            outdir: PathBuf::from("."),
        }
    }
}

/// DNA strand orientation for genomic coordinates.
///
/// # Variants
///
/// - `Forward`: Positive strand (+)
/// - `Reverse`: Negative strand (-)
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::Strand;
///
/// let strand = Strand::Forward;
/// assert_eq!(strand.to_string(), "+");
/// ```
enum Strand {
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

/// Request to write a chunk file to disk.
///
/// # Arguments
///
/// - `path`: Output file path
/// - `payload`: FASTA record bytes
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::ChunkWriteRequest;
/// use std::path::PathBuf;
///
/// let req = ChunkWriteRequest {
///     path: PathBuf::from("chunk.fa"),
///     payload: b">chr1:1-100(+)\nACGT\n".to_vec(),
/// };
/// ```
struct ChunkWriteRequest {
    path: PathBuf,
    payload: Vec<u8>,
}

/// Normalizes an ambiguous nucleotide base to a canonical base.
///
/// # Arguments
///
/// - `base`: Input nucleotide base (A, C, G, T, N, or lowercase)
///
/// Returns `A` for `N` or `n`, otherwise returns the base unchanged.
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::normalize_base;
///
/// assert_eq!(normalize_base(b'A'), b'A');
/// assert_eq!(normalize_base(b'N'), b'A');
/// assert_eq!(normalize_base(b't'), b't');
/// ```
fn normalize_base(base: u8) -> u8 {
    match base {
        b'N' | b'n' => b'A',
        base => base,
    }
}

/// Normalizes all bases in a sequence by converting ambiguous bases to A.
///
/// # Arguments
///
/// - `seq`: Input sequence bytes
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::normalize_owned_sequence;
///
/// let seq = normalize_owned_sequence(b"ACGN".to_vec());
/// assert_eq!(seq, b"ACGA".to_vec());
/// ```
fn normalize_owned_sequence(seq: Vec<u8>) -> Vec<u8> {
    seq.into_iter().map(normalize_base).collect()
}

/// Sanitizes a string for use in filenames.
///
/// # Arguments
///
/// - `raw`: Input bytes
///
/// Keeps alphanumeric, '.', '_', and '-'. Replaces invalid chars with '_'.
/// Returns "unknown" if empty.
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::sanitize_filename_component;
///
/// let name = sanitize_filename_component(b"chr1:1000");
/// assert!(name.contains("chr1"));
/// ```
fn sanitize_filename_component(raw: &[u8]) -> String {
    let mut sanitized: String = raw
        .iter()
        .map(|base| match *base {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'-' => *base as char,
            _ => '_',
        })
        .collect();

    if sanitized.is_empty() {
        sanitized.push_str("unknown");
    }

    sanitized
}

/// Appends a FASTA record to a buffer.
///
/// # Arguments
///
/// - `buffer`: Output buffer
/// - `header`: Record header (without '>')
/// - `sequence`: Sequence bytes
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::push_fasta_record;
///
/// let mut buf = Vec::new();
/// push_fasta_record(&mut buf, b"chr1:1-100(+)", b"ACGT");
/// assert!(buf.starts_with(b">"));
/// ```
fn push_fasta_record(buffer: &mut Vec<u8>, header: &[u8], sequence: &[u8]) {
    buffer.push(b'>');
    buffer.extend_from_slice(header);
    buffer.push(b'\n');
    buffer.extend_from_slice(sequence);
    buffer.push(b'\n');
}

/// Builds a chunk payload with forward and reverse strand FASTA records.
///
/// # Arguments
///
/// - `chrom`: Chromosome name
/// - `chunk_start`: Inner chunk start coordinate
/// - `chunk_end`: Inner chunk end coordinate
/// - `sequence`: Sequence bytes
///
/// Returns a buffer containing two FASTA records: one for each strand.
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::build_chunk_payload;
///
/// let payload = build_chunk_payload(b"chr1", 1, 100, b"ACGT");
/// assert!(payload.starts_with(b">"));
/// ```
fn build_chunk_payload(
    chrom: &[u8],
    chunk_start: usize,
    chunk_end: usize,
    sequence: &[u8],
) -> Vec<u8> {
    let forward_id = generic_identifier(chrom, chunk_start, chunk_end, Strand::Forward);
    let reverse_id = generic_identifier(chrom, chunk_start, chunk_end, Strand::Reverse);
    // let reverse_sequence = reverse_complement(sequence);

    let mut payload =
        Vec::with_capacity((sequence.len() * 2) + forward_id.len() + reverse_id.len() + 8);
    push_fasta_record(&mut payload, &forward_id, sequence);
    push_fasta_record(&mut payload, &reverse_id, sequence);

    payload
}

/// Writes a chunk file to disk, optionally compressing with gzip.
///
/// # Arguments
///
/// - `path`: Output file path
/// - `payload`: FASTA record bytes
/// - `gzip`: Whether to compress output
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::write_chunk_file;
/// use std::path::PathBuf;
///
/// write_chunk_file(&PathBuf::from("out.fa"), b">chr1\nACGT\n", false).ok();
/// ```
fn write_chunk_file(path: &Path, payload: &[u8], gzip: bool) -> std::io::Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    if gzip {
        let mut encoder = GzEncoder::new(writer, Compression::default());
        encoder.write_all(payload)?;
        encoder.finish()?;
    } else {
        let mut writer = writer;
        writer.write_all(payload)?;
        writer.flush()?;
    }

    Ok(())
}

/// Chunks genome sequences into smaller FASTA files for SpliceAI processing.
///
/// # Arguments
///
/// - `sizes`: Chromosome name to length map
/// - `genome`: Chromosome name to sequence bytes map
/// - `config`: Chunking configuration
///
/// Writes chunk files to `<outdir>/chunks/` with forward and reverse strand records.
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::{chunk, Config};
/// use std::collections::HashMap;
///
/// let mut sizes = HashMap::new();
/// sizes.insert(b"chr1".to_vec(), 1000);
/// let mut genome = HashMap::new();
/// genome.insert(b"chr1".to_vec(), b"ACGTACGT".to_vec());
/// let config = Config::default();
/// chunk(&sizes, &genome, config);
/// ```
fn chunk(sizes: &HashMap<Vec<u8>, u32>, genome: &HashMap<Vec<u8>, Vec<u8>>, config: Config) {
    if config.chunk_size == 0 {
        log::error!("ERROR: chunk_size must be greater than zero");
        std::process::exit(1);
    }

    let chunk_dir = config.outdir.join("chunks");
    create_dir_all(&chunk_dir).unwrap_or_else(|e| {
        log::error!("ERROR: Cannot create output directory: {}", e);
        std::process::exit(1);
    });

    info!("Writing chunks to {}", chunk_dir.display());

    let gzip = config.gzip;
    let queue_size = rayon::current_num_threads().max(1) * 2;
    let (sender, receiver) = mpsc::sync_channel::<ChunkWriteRequest>(queue_size);
    let writer_handle = thread::spawn(move || -> Result<usize, String> {
        let mut written = 0usize;

        while let Ok(request) = receiver.recv() {
            write_chunk_file(&request.path, &request.payload, gzip).map_err(|e| {
                format!(
                    "ERROR: Cannot write chunk {}: {}",
                    request.path.display(),
                    e
                )
            })?;
            written += 1;
        }

        Ok(written)
    });

    info!(
        "Start processing chromosomes with chunk size {}",
        config.chunk_size
    );

    let chunk_size = config.chunk_size as usize;
    let flank_size = config.flank_size as usize;
    let min_contig_size = config.min_contig_size as usize;
    let file_extension = if gzip { "fa.gz" } else { "fa" };

    // INFO: par iter on sizes, chunk each chrom and add flanks
    let produce_result: Result<(), String> =
        sizes
            .par_iter()
            .try_for_each_with(sender.clone(), |tx, (chr, length)| {
                info!("Processing chromosome {}", String::from_utf8_lossy(chr));
                let length = *length as usize;

                if length < min_contig_size {
                    return Ok(());
                }

                let sequence = genome.get(chr).ok_or_else(|| {
                    format!(
                        "ERROR: Chromosome {} not found",
                        String::from_utf8_lossy(chr)
                    )
                })?;

                if sequence.len() < length {
                    return Err(format!(
                        "ERROR: Chromosome {} size mismatch: declared {}, sequence {}",
                        String::from_utf8_lossy(chr),
                        length,
                        sequence.len()
                    ));
                }

                let chrom_label = sanitize_filename_component(chr);

                // INFO: header stores the inner chunk interval, while the sequence still
                // contains the available flanks around that interval.
                for (chunk_index, start) in (0..length).step_by(chunk_size).enumerate() {
                    let end = (start + chunk_size).min(length);
                    let flanked_start = start.saturating_sub(flank_size);
                    let flanked_end = end.saturating_add(flank_size).min(length);

                    let chunk_sequence =
                        sequence.get(flanked_start..flanked_end).ok_or_else(|| {
                            format!(
                                "ERROR: Failed to extract sequence from {} with {}-{}",
                                String::from_utf8_lossy(chr),
                                flanked_start,
                                flanked_end
                            )
                        })?;

                    let path = chunk_dir.join(format!(
                        "tmp.{chrom_label}.chunk.{chunk_index}.{file_extension}"
                    ));
                    let payload = build_chunk_payload(chr, start, end, chunk_sequence);

                    tx.send(ChunkWriteRequest { path, payload }).map_err(|e| {
                        format!(
                            "ERROR: Failed to queue chunk {}:{}-{}: {}",
                            String::from_utf8_lossy(chr),
                            start + 1,
                            end,
                            e
                        )
                    })?;
                }

                Ok(())
            });

    drop(sender);

    let write_result = match writer_handle.join() {
        Ok(result) => result,
        Err(_) => Err(String::from("ERROR: Chunk writer thread panicked")),
    };

    if let Err(err) = produce_result {
        log::error!("{}", err);
        std::process::exit(1);
    }

    match write_result {
        Ok(count) => info!("Wrote {} chunk files to {}", count, chunk_dir.display()),
        Err(err) => {
            log::error!("{}", err);
            std::process::exit(1);
        }
    }
}

/// Runs the chunking pipeline with the given sequence file and configuration.
///
/// # Arguments
///
/// - `sequence`: Path to the genome file
/// - `config`: Chunking configuration
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::{run, Config};
/// use std::path::PathBuf;
///
/// let config = Config::default();
/// run(PathBuf::from("genome.fa"), config);
/// ```
pub fn run(sequence: PathBuf, config: Config) {
    let (genome, sizes) = get_sequences(sequence);
    chunk(&sizes, &genome, config);
}

/// Creates a generic identifier from genomic coordinates.
///
/// # Arguments
///
/// - `chrom`: Chromosome name
/// - `start`: Start coordinate
/// - `end`: End coordinate
/// - `strand`: Strand information
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::generic_identifier;
///
/// let id = generic_identifier(b"chr1", 1000, 2000, Strand::Forward);
/// assert_eq!(id, b"chr1:1000-2000(+)");
/// ```
fn generic_identifier(chrom: &[u8], start: usize, end: usize, strand: Strand) -> Vec<u8> {
    let start = start.to_string();
    let end = end.to_string();
    let mut identifier = Vec::with_capacity(chrom.len() + start.len() + end.len() + 5);

    let strand = match strand {
        Strand::Forward => b'+',
        Strand::Reverse => b'-',
    };

    identifier.extend_from_slice(chrom);
    identifier.push(b':');
    identifier.extend_from_slice(start.as_bytes());
    identifier.push(b'-');
    identifier.extend_from_slice(end.as_bytes());
    identifier.push(b'(');
    identifier.push(strand);
    identifier.push(b')');

    identifier
}

/// Loads genome sequences from a file (2bit or FASTA format).
///
/// # Arguments
///
/// - `sequence`: Path to the genome file (.fa, .fa.gz, or .2bit)
///
/// # Example
///
/// ```rust,ignore
/// use std::path::PathBuf;
///
/// let genome = get_sequences(PathBuf::from("genome.2bit"));
/// let genome = get_sequences(PathBuf::from("genome.fa"));
/// let genome = get_sequences(PathBuf::from("genome.fa.gz"));
/// ```
pub fn get_sequences(sequence: PathBuf) -> (HashMap<Vec<u8>, Vec<u8>>, HashMap<Vec<u8>, u32>) {
    if sequence == *"-" {
        return from_stdin();
    }

    info!("Reading sequences from file {}", sequence.display());
    match sequence.extension() {
        Some(ext) => match ext.to_str() {
            Some("2bit") => from_2bit(sequence),
            Some("fa") | Some("fasta") | Some("fna") | Some("gz") => from_fa(sequence),
            _ => panic!("ERROR: Unsupported file format"),
        },
        None => panic!("ERROR: No file extension"),
    }
}

/// Loads genome sequences from stdin.
///
/// # Example
///
/// ```rust,ignore
/// use std::io::Write;
/// use std::process::{Command, Stdio};
///
/// let mut child = Command::new("cat")
///     .arg("genome.fa")
///     .stdout(Stdio::piped())
///     .spawn()
///     .unwrap_or_else(|e| panic!("ERROR: cannot spawn cat: {}", e));
///
/// let genome = from_stdin();
/// ```
fn from_stdin() -> (HashMap<Vec<u8>, Vec<u8>>, HashMap<Vec<u8>, u32>) {
    info!("Reading sequences from stdin");

    let mut input = Vec::new();
    std::io::stdin()
        .read_to_end(&mut input)
        .unwrap_or_else(|e| panic!("ERROR: cannot read stdin: {}", e));

    if input.is_empty() {
        panic!("ERROR: Missing --sequence and stdin is empty");
    }

    if input.starts_with(&GZIP_MAGIC) {
        return parse_fasta_reader(
            BufReader::new(MultiGzDecoder::new(Cursor::new(input))),
            "stdin",
        );
    }

    if input.starts_with(&TWOBIT_MAGIC) || input.starts_with(&TWOBIT_REV_MAGIC) {
        return from_2bit_buf(input, "stdin");
    }

    if input
        .iter()
        .copied()
        .find(|b| !b.is_ascii_whitespace())
        .is_some_and(|b| b == b'>')
    {
        return parse_fasta_reader(BufReader::new(Cursor::new(input)), "stdin");
    }

    panic!("ERROR: Unsupported stdin sequence format");
}

/// Loads genome sequences from a 2bit compressed format file.
///
/// # Arguments
///
/// - `twobit`: Path to the 2bit file
///
/// # Example
///
/// ```rust,ignore
/// use std::path::PathBuf;
///
/// let sequences = from_2bit(PathBuf::from("genome.2bit"));
/// let chr1 = sequences.get(b"chr1");
/// ```
fn from_2bit(twobit: PathBuf) -> (HashMap<Vec<u8>, Vec<u8>>, HashMap<Vec<u8>, u32>) {
    let genome = TwoBitFile::open_and_read(&twobit).expect("ERROR: Cannot open 2bit file");
    let source = format!("file {}", twobit.display());
    collect_2bit_sequences(genome, &source)
}

/// Loads genome sequences from a 2bit buffer.
///
/// # Arguments
///
/// - `buf`: 2bit file contents as bytes
/// - `source`: Source description for error messages
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::from_2bit_buf;
/// // let genome = from_2bit_buf(data, "stdin");
/// ```
fn from_2bit_buf(buf: Vec<u8>, source: &str) -> (HashMap<Vec<u8>, Vec<u8>>, HashMap<Vec<u8>, u32>) {
    let genome = TwoBitFile::from_buf(buf)
        .unwrap_or_else(|e| panic!("ERROR: Cannot read 2bit from {}: {}", source, e));
    collect_2bit_sequences(genome, source)
}

/// Collects sequences from a 2bit reader into a HashMap.
///
/// # Arguments
///
/// - `genome`: TwoBitFile reader
/// - `source`: Source description for logging
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::collect_2bit_sequences;
/// ```
fn collect_2bit_sequences<R: Read + Seek>(
    mut genome: TwoBitFile<R>,
    source: &str,
) -> (HashMap<Vec<u8>, Vec<u8>>, HashMap<Vec<u8>, u32>) {
    let mut sequences = HashMap::new();
    let mut chrom_sizes = HashMap::new();

    genome
        .chrom_names()
        .iter()
        .zip(genome.chrom_sizes())
        .for_each(|(chr, size)| {
            chrom_sizes.insert(chr.as_bytes().to_vec(), size as u32);
        });

    genome.chrom_names().iter().for_each(|chr| {
        let seq = genome
            .read_sequence(chr, ..)
            .unwrap_or_else(|e| panic!("ERROR: {}", e))
            .as_bytes()
            .to_vec()
            .into_iter()
            .map(normalize_base)
            .collect();

        sequences.insert(chr.as_bytes().to_vec(), seq);
    });

    info!("Read {} sequences from {}", sequences.len(), source);

    (sequences, chrom_sizes)
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
pub fn from_fa<P: AsRef<Path>>(f: P) -> (HashMap<Vec<u8>, Vec<u8>>, HashMap<Vec<u8>, u32>) {
    let path = f.as_ref();
    let file = File::open(path)
        .unwrap_or_else(|e| panic!("ERROR: cannot open FASTA {}: {}", path.display(), e));

    let reader: Box<dyn BufRead> = match path.extension().and_then(|ext| ext.to_str()) {
        Some("gz") => Box::new(BufReader::new(MultiGzDecoder::new(file))),
        _ => Box::new(BufReader::new(file)),
    };

    let source = format!("file {}", path.display());
    parse_fasta_reader(reader, &source)
}

/// Parses FASTA format reader into a HashMap of sequences.
///
/// # Arguments
///
/// - `reader`: Buffered reader containing FASTA data
/// - `source`: Source description for logging
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::parse_fasta_reader;
/// use std::io::BufReader;
/// // let genome = parse_fasta_reader(BufReader::new(file), "file.fasta");
/// ```
fn parse_fasta_reader<R: BufRead>(
    mut reader: R,
    source: &str,
) -> (HashMap<Vec<u8>, Vec<u8>>, HashMap<Vec<u8>, u32>) {
    let mut line = Vec::new();
    let mut header: Option<Vec<u8>> = None;
    let mut seq = Vec::new();

    let mut acc = HashMap::new();
    let mut chrom_sizes = HashMap::new();

    loop {
        line.clear();
        let bytes_read = reader
            .read_until(b'\n', &mut line)
            .unwrap_or_else(|e| panic!("ERROR: cannot read FASTA {}: {}", source, e));

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
                acc.insert(
                    prev_header,
                    normalize_owned_sequence(std::mem::take(&mut seq)),
                );
            }
        } else {
            seq.extend_from_slice(&line);
        }
    }

    if let Some(last_header) = header {
        acc.insert(last_header, normalize_owned_sequence(seq));
    }

    acc.iter().for_each(|(k, v)| {
        chrom_sizes.entry(k.clone()).or_insert(v.len() as u32);
    });

    info!("Read {} sequences from {}", acc.len(), source);

    (acc, chrom_sizes)
}

/// Error type for range calculation failures during sequence extraction.
///
/// # Variants
///
/// - `Underflow`: Coordinate minus flank would result in negative value
///
/// # Example
///
/// ```rust,ignore
/// use xloci::core::RangeError;
///
/// let err = RangeError::Underflow {
///     feature_coord: 5,
///     flank: 10,
/// };
/// println!("{}", err); // "ERROR: Feature coordinate 5 is underflowing by 10 bases"
/// ```
#[derive(Debug)]
#[allow(dead_code)]
enum RangeError {
    Underflow { feature_coord: usize, flank: usize },
    Overflow { feature_coord: usize, flank: usize },
}

/// Formats the RangeError as a human-readable error message.
///
/// # Arguments
///
/// - `f`: The formatter to write to
///
/// # Example
///
/// ```rust,ignore
/// let err = RangeError::Underflow { feature_coord: 5, flank: 10 };
/// assert!(err.to_string().contains("underflowing"));
/// ```
impl std::fmt::Display for RangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RangeError::Underflow {
                feature_coord,
                flank,
            } => write!(
                f,
                "ERROR: Feature coordinate {} is underflowing by {} bases",
                feature_coord, flank
            ),
            RangeError::Overflow {
                feature_coord,
                flank,
            } => write!(
                f,
                "ERROR: Feature coordinate {} is overflowing by {} bases",
                feature_coord, flank
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        io::Cursor,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn parse_fasta_reader_normalizes_last_sequence() {
        let fasta = b">chr1\nACN\n>chr2\nTTN\n";
        let (genome, sizes) = parse_fasta_reader(BufReader::new(Cursor::new(&fasta[..])), "test");

        assert_eq!(genome.get(b"chr1".as_slice()), Some(&b"ACA".to_vec()));
        assert_eq!(genome.get(b"chr2".as_slice()), Some(&b"TTA".to_vec()));
        assert_eq!(sizes.get(b"chr2".as_slice()), Some(&3));
    }

    #[test]
    fn chunk_writes_expected_files_for_exact_multiple_lengths() {
        let mut sizes = HashMap::new();
        sizes.insert(b"chr1".to_vec(), 8);

        let mut genome = HashMap::new();
        genome.insert(b"chr1".to_vec(), b"ACGTACGT".to_vec());

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let outdir = std::env::temp_dir().join(format!(
            "spliceai_chunk_test_{}_{}",
            std::process::id(),
            unique
        ));

        let config = Config {
            chunk_size: 4,
            flank_size: 1,
            min_contig_size: 1,
            gzip: false,
            outdir: outdir.clone(),
        };

        chunk(&sizes, &genome, config);

        let chunks_dir = outdir.join("chunks");
        let mut files: Vec<_> = fs::read_dir(&chunks_dir)
            .unwrap_or_else(|e| panic!("ERROR: Cannot list chunk directory: {}", e))
            .map(|entry| {
                entry
                    .unwrap_or_else(|e| panic!("ERROR: Cannot read chunk directory entry: {}", e))
                    .file_name()
                    .into_string()
                    .unwrap_or_else(|_| String::from("invalid"))
            })
            .collect();
        files.sort();

        assert_eq!(
            files,
            vec![
                String::from("tmp.chr1.chunk.0.fa"),
                String::from("tmp.chr1.chunk.1.fa"),
            ]
        );

        let first = fs::read(chunks_dir.join("tmp.chr1.chunk.0.fa"))
            .unwrap_or_else(|e| panic!("ERROR: Cannot read first chunk: {}", e));
        assert_eq!(
            String::from_utf8(first).unwrap_or_else(|e| panic!("ERROR: Invalid UTF-8: {}", e)),
            String::from(">chr1:0-4(+)\nACGTA\n>chr1:0-4(-)\nACGTA\n")
        );

        let second = fs::read(chunks_dir.join("tmp.chr1.chunk.1.fa"))
            .unwrap_or_else(|e| panic!("ERROR: Cannot read second chunk: {}", e));
        assert_eq!(
            String::from_utf8(second).unwrap_or_else(|e| panic!("ERROR: Invalid UTF-8: {}", e)),
            String::from(">chr1:4-8(+)\nTACGT\n>chr1:4-8(-)\nTACGT\n")
        );

        fs::remove_dir_all(&outdir)
            .unwrap_or_else(|e| panic!("ERROR: Cannot remove temporary directory: {}", e));
    }
}
