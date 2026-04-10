// Copyright (c) 2026 Alejandro Gonzalez-Irribarren <alejandrxgzi@gmail.com>
// Distributed under the terms of the Apache License, Version 2.0.

use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use clap::{self, Parser};
use dashmap::DashMap;
use flate2::{read::MultiGzDecoder, write::GzEncoder, Compression};
use genepred::{Bed12, Extras, GenePred};
use log::info;
use packbed::OverlapType;
use rayon::prelude::*;
use twobit::TwoBitFile;

const CHUNKS_DIR_NAME: &str = "chunks";

#[derive(Debug, Parser)]
#[clap(
    name = "aparent",
    version = env!("CARGO_PKG_VERSION"),
    author = env!("CARGO_PKG_AUTHORS"),
    about = "chunk wrapper for APARENT"
)]
struct Args {
    #[arg(
        short = 'b',
        long = "bed",
        required = true,
        value_name = "PATH",
        help = "Path to BED12 file"
    )]
    pub bed: PathBuf,

    #[arg(
        short = 'g',
        long = "genome",
        required = true,
        value_name = "PATH",
        help = "Path to genome file"
    )]
    pub genome: PathBuf,

    #[arg(
        short = 'u',
        long = "upstream",
        default_value_t = 100,
        value_name = "INT",
        help = "Upstream distance"
    )]
    pub upstream: usize,

    #[arg(
        short = 'd',
        long = "downstream",
        default_value_t = 100,
        value_name = "INT",
        help = "Downstream distance"
    )]
    pub downstream: usize,

    #[arg(
        short = 'o',
        long = "output",
        value_name = "PATH",
        help = "Output file"
    )]
    pub output: PathBuf,

    #[arg(
        short = 'c',
        long = "chunks",
        value_name = "INT",
        help = "Records per chunked output file"
    )]
    pub chunks: Option<NonZeroUsize>,

    #[arg(
        short = 'p',
        long = "prefix",
        value_name = "PREFIX",
        default_value = "part",
        help = "File name prefix for chunked files"
    )]
    pub prefix: String,

    #[arg(
        short = 'G',
        long = "gz",
        action = clap::ArgAction::SetTrue,
        help = "Gzip-compress output files"
    )]
    pub gz: bool,

    #[arg(
        short = 't',
        long = "threads",
        help = "Number of threads",
        value_name = "THREADS",
        default_value_t = num_cpus::get()
    )]
    pub threads: usize,

    #[arg(
        short = 'L',
        long = "level",
        help = "Log level",
        value_name = "LEVEL",
        default_value_t = log::Level::Info
    )]
    pub level: log::Level,
}

fn main() {
    let start = std::time::Instant::now();

    let args = Args::parse();
    simple_logger::init_with_level(args.level).unwrap_or_else(|e| {
        log::error!("Cannot initialize logger: {}", e);
        std::process::exit(1);
    });

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap_or_else(|e| {
            log::error!("Cannot configure thread pool: {}", e);
            std::process::exit(1);
        });

    run(args).unwrap_or_else(|e| {
        log::error!("ERROR: {}", e);
        std::process::exit(1);
    });

    info!("Finished in {}s", start.elapsed().as_secs_f32());
}

/// A writer that outputs data either as plain text or gzip-compressed.
///
/// # Example
///
/// ```rust,ignore
/// let writer = OutputWriter::create(Path::new("output.tsv"), false)?;
/// ```
enum OutputWriter {
    Plain(BufWriter<File>),
    Gzip(GzEncoder<BufWriter<File>>),
}

/// Methods for creating and finalizing an OutputWriter.
impl OutputWriter {
    /// Creates a new OutputWriter for the given file path.
    ///
    /// # Arguments
    ///
    /// - `path`: Path to the output file
    /// - `gz`: Whether to gzip-compress the output
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let writer = OutputWriter::create(Path::new("output.tsv"), false)?;
    /// ```
    fn create(path: &Path, gz: bool) -> io::Result<Self> {
        let file = File::create(path)?;

        if gz {
            Ok(Self::Gzip(GzEncoder::new(
                BufWriter::new(file),
                Compression::default(),
            )))
        } else {
            Ok(Self::Plain(BufWriter::new(file)))
        }
    }

    fn finish(self) -> io::Result<()> {
        match self {
            Self::Plain(mut writer) => writer.flush(),
            Self::Gzip(mut writer) => {
                writer.flush()?;
                writer.try_finish()
            }
        }
    }
}

/// Implements the Write trait for OutputWriter.
///
/// Delegates to the underlying writer (plain or gzip) transparently.
impl Write for OutputWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Plain(writer) => writer.write(buf),
            Self::Gzip(writer) => writer.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Plain(writer) => writer.flush(),
            Self::Gzip(writer) => writer.flush(),
        }
    }
}

/// Writes output to multiple chunked files, rotating when the record limit is reached.
///
/// # Example
///
/// ```rust,ignore
/// let mut writer = ChunkedWriter::new(dir, "part".to_string(), 1000, false);
/// writer.write_record("chr1", 0, 100, Strand::Forward, b"ACGT")?;
/// writer.finish()?;
/// ```
struct ChunkedWriter {
    output_dir: PathBuf,
    prefix: String,
    chunk_size: usize,
    gz: bool,
    file_index: usize,
    records_in_file: usize,
    writer: Option<OutputWriter>,
}

/// Methods for managing chunked file output.
impl ChunkedWriter {
    /// Creates a new ChunkedWriter.
    ///
    /// # Arguments
    ///
    /// - `output_dir`: Directory for output files
    /// - `prefix`: Prefix for chunk file names
    /// - `chunk_size`: Number of records per chunk file
    /// - `gz`: Whether to gzip-compress output
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let writer = ChunkedWriter::new(dir, "part".to_string(), 1000, false);
    /// ```
    fn new(output_dir: PathBuf, prefix: String, chunk_size: usize, gz: bool) -> Self {
        Self {
            output_dir,
            prefix,
            chunk_size,
            gz,
            file_index: 0,
            records_in_file: 0,
            writer: None,
        }
    }

    fn write_record(
        &mut self,
        chr: &str,
        start: u64,
        end: u64,
        strand: genepred::Strand,
        target: &[u8],
    ) -> io::Result<()> {
        if self.writer.is_none() || self.records_in_file == self.chunk_size {
            self.rotate()?;
        }

        write_record(
            self.writer
                .as_mut()
                .expect("chunk writer must be initialized before writing"),
            chr,
            start,
            end,
            strand,
            target,
        )?;
        self.records_in_file += 1;

        Ok(())
    }

    fn rotate(&mut self) -> io::Result<()> {
        if let Some(writer) = self.writer.take() {
            writer.finish()?;
        }

        self.file_index += 1;
        self.records_in_file = 0;

        let path = chunk_file_path(&self.output_dir, &self.prefix, self.file_index, self.gz);
        self.writer = Some(OutputWriter::create(&path, self.gz)?);

        Ok(())
    }

    fn finish(self) -> io::Result<()> {
        if let Some(writer) = self.writer {
            writer.finish()?;
        }

        Ok(())
    }
}

/// An output sink that writes to either a single file or multiple chunked files.
///
/// # Example
///
/// ```rust,ignore
/// let sink = OutputSink::new(Path::new("output.tsv"), None, "part".to_string(), false)?;
/// ```
enum OutputSink {
    Single(OutputWriter),
    Chunked(ChunkedWriter),
}

/// Methods for creating and writing to an OutputSink.
impl OutputSink {
    /// Creates a new OutputSink.
    ///
    /// # Arguments
    ///
    /// - `output`: Path to output file or directory
    /// - `chunks`: If Some, enables chunked output with this chunk size
    /// - `prefix`: Prefix for chunk file names
    /// - `gz`: Whether to gzip-compress output
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sink = OutputSink::new(Path::new("output.tsv"), None, "part".to_string(), false)?;
    /// let chunked = OutputSink::new(Path::new("output/"), Some(1000), "part".to_string(), true)?;
    /// ```
    fn new(output: &Path, chunks: Option<usize>, prefix: String, gz: bool) -> io::Result<Self> {
        match chunks {
            Some(chunk_size) => {
                let output_dir = resolve_chunk_output_dir(output);
                std::fs::create_dir_all(&output_dir)?;

                Ok(Self::Chunked(ChunkedWriter::new(
                    output_dir, prefix, chunk_size, gz,
                )))
            }
            None => {
                if output.exists() && output.is_dir() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "--output {} is a directory; provide a file path or use --chunks",
                            output.display()
                        ),
                    ));
                }

                let output_path = resolve_single_output_path(output, gz);
                std::fs::create_dir_all(parent_dir_or_current(&output_path))?;

                Ok(Self::Single(OutputWriter::create(&output_path, gz)?))
            }
        }
    }

    fn write_record(
        &mut self,
        chr: &str,
        start: u64,
        end: u64,
        strand: genepred::Strand,
        target: &[u8],
    ) -> io::Result<()> {
        match self {
            Self::Single(writer) => write_record(writer, chr, start, end, strand, target),
            Self::Chunked(writer) => writer.write_record(chr, start, end, strand, target),
        }
    }

    fn finish(self) -> io::Result<()> {
        match self {
            Self::Single(writer) => writer.finish(),
            Self::Chunked(writer) => writer.finish(),
        }
    }
}

/// Writes a single record in BED-like format.
///
/// # Arguments
///
/// - `writer`: The writer to write to
/// - `chr`: Chromosome name
/// - `start`: Start position
/// - `end`: End position
/// - `strand`: DNA strand (forward or reverse)
/// - `target`: Target sequence data
///
/// # Example
///
/// ```rust,ignore
/// write_record(&mut file, "chr1", 0, 100, Strand::Forward, b"ACGT")?;
/// ```
fn write_record(
    writer: &mut impl Write,
    chr: &str,
    start: u64,
    end: u64,
    strand: genepred::Strand,
    target: &[u8],
) -> io::Result<()> {
    write!(writer, "{chr}\t{start}\t{end}\t{strand}\t")?;
    writer.write_all(target)?;
    writer.write_all(b"\n")
}

/// Returns the parent directory of a path, or "." if no parent exists.
///
/// # Arguments
///
/// - `path`: The path to get the parent of
///
/// # Example
///
/// ```rust,ignore
/// let parent = parent_dir_or_current(Path::new("/foo/bar/file.txt"));
/// ```
fn parent_dir_or_current(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

/// Resolves the output directory for chunked files.
///
/// # Arguments
///
/// - `output`: The base output path
///
/// # Example
///
/// ```rust,ignore
/// let dir = resolve_chunk_output_dir(Path::new("/output/"));
/// ```
fn resolve_chunk_output_dir(output: &Path) -> PathBuf {
    if output.exists() && output.is_dir() {
        output.join(CHUNKS_DIR_NAME)
    } else {
        parent_dir_or_current(output).join(CHUNKS_DIR_NAME)
    }
}

/// Resolves the output path, appending .gz extension if needed.
///
/// # Arguments
///
/// - `output`: The base output path
/// - `gz`: Whether to append .gz extension
///
/// # Example
///
/// ```rust,ignore
/// let path = resolve_single_output_path(Path::new("out.tsv"), true);
/// ```
fn resolve_single_output_path(output: &Path, gz: bool) -> PathBuf {
    if gz && output.extension().and_then(|ext| ext.to_str()) != Some("gz") {
        match output.file_name() {
            Some(file_name) => output.with_file_name(format!("{}.gz", file_name.to_string_lossy())),
            None => output.to_path_buf(),
        }
    } else {
        output.to_path_buf()
    }
}

/// Generates a chunk file path with indexed naming.
///
/// # Arguments
///
/// - `output_dir`: Directory for the chunk file
/// - `prefix`: Prefix for the file name
/// - `file_index`: Index for the file (zero-padded to 5 digits)
/// - `gz`: Whether to use .tsv.gz extension
///
/// # Example
///
/// ```rust,ignore
/// let path = chunk_file_path(Path::new("/output"), "part", 1, false);
/// ```
fn chunk_file_path(output_dir: &Path, prefix: &str, file_index: usize, gz: bool) -> PathBuf {
    let suffix = if gz { ".tsv.gz" } else { ".tsv" };
    output_dir.join(format!("{prefix}.{file_index:05}{suffix}"))
}

/// Calculates the start and end bounds of a gene component.
///
/// # Arguments
///
/// - `component`: A slice of GenePred records
///
/// # Example
///
/// ```rust,ignore
/// let bounds = component_bounds(&gene_preds);
/// ```
fn component_bounds(component: &[GenePred]) -> (u64, u64) {
    component
        .iter()
        .fold((u64::MAX, 0), |(start, end), interval| {
            (start.min(interval.start()), end.max(interval.end()))
        })
}

/// Runs the main logic of the tool.
///
/// # Arguments
///
/// - `args`: Parsed command-line arguments
///
/// # Example
///
/// ```rust,ignore
/// run(Args::parse())?;
/// ```
fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut output = OutputSink::new(
        &args.output,
        args.chunks.map(NonZeroUsize::get),
        args.prefix.clone(),
        args.gz,
    )
    .unwrap_or_else(|e| {
        log::error!("Cannot initialize output: {}", e);
        std::process::exit(1);
    });

    let sequences = get_sequences(args.genome);
    let intervals = make_intervals(&args.bed, args.upstream, args.downstream);

    log::info!("Buckerizing intervals");
    let mut buckets: Vec<_> = packbed::buckerize(intervals, OverlapType::Exon)
        .into_iter()
        .collect();
    buckets.sort_unstable_by(|left, right| left.0.cmp(&right.0));

    for (key, mut components) in buckets {
        // INFO: loop over components and write to file
        // INFO: key is in fmt "chr:strand"
        let chr = key.split(':').next().unwrap_or_else(|| {
            log::error!("Could not split key {} into chromosome and strand", key);
            std::process::exit(1);
        });
        let sequence = sequences.get(chr.as_bytes()).unwrap_or_else(|| {
            log::error!(
                "ERROR: Chromosome {} from {} not found in genome. Chromosomes available are {:?}",
                chr,
                key,
                sequences.keys()
            );
            std::process::exit(1);
        });

        components.sort_unstable_by_key(|component| component_bounds(component));

        for component in components.iter() {
            let mut target = Vec::new();
            let (start, end) = component_bounds(component);
            let strand = component[0].strand().unwrap_or_else(|| {
                log::error!("BED12 record {} does not have a strand", component[0]);
                std::process::exit(1);
            });

            if let Some(slice) = sequence.get(start as usize..end as usize) {
                target.extend_from_slice(slice);
            }

            match strand {
                genepred::Strand::Forward => {}
                genepred::Strand::Reverse => {
                    target.reverse();

                    for base in target.iter_mut() {
                        *base = match *base {
                            b'A' => b'T',
                            b'C' => b'G',
                            b'G' => b'C',
                            b'T' => b'A',
                            b'N' => b'N',
                            b'a' => b't',
                            b'c' => b'g',
                            b'g' => b'c',
                            b't' => b'a',
                            b'n' => b'n',
                            _ => panic!("ERROR: Invalid base"),
                        }
                    }
                }
                _ => {}
            }

            output
                .write_record(chr, start, end, strand, &target)
                .unwrap_or_else(|e| {
                    log::error!("Cannot write to output file: {}", e);
                    std::process::exit(1);
                });
        }
    }

    output.finish().unwrap_or_else(|e| {
        log::error!("Cannot write to output file: {}", e);
        std::process::exit(1);
    });

    Ok(())
}

/// Makes a map of intervals from a BED12 file.
///
/// # Arguments
///
/// - `bed`: Path to the BED12 file
/// - `upstream`: Upstream distance
/// - `downstream`: Downstream distance
///
/// # Example
///
/// ```rust,ignore
/// use std::path::PathBuf;
///
/// let intervals = make_intervals(PathBuf::from("genes.bed"), 100, 100);
/// ```
fn make_intervals(
    bed: &PathBuf,
    upstream: usize,
    downstream: usize,
) -> DashMap<String, Vec<GenePred>> {
    log::info!("Making intervals from {}", bed.display());

    let intervals = DashMap::new();
    genepred::Reader::<Bed12>::from_mmap(bed)
        .unwrap_or_else(|e| panic!("{}", e))
        .par_records()
        .unwrap_or_else(|e| panic!("{}", e))
        .for_each(|record| {
            // INFO: collect 3UTR intervals and only preserve bounds as BED6
            let record = record.unwrap();
            let name = record.name().unwrap_or_else(|| {
                log::error!("BED12 record {} does not have a name", record);
                std::process::exit(1);
            });
            let strand = record.strand().unwrap_or_else(|| {
                log::error!("BED12 record {} does not have a strand", record);
                std::process::exit(1);
            });
            let three_prime_utr = record.three_prime_utr();

            let start = three_prime_utr
                .first()
                .unwrap_or_else(|| {
                    log::warn!(
                        "BED12 record {} does not contain a 3UTR interval, will be ignored",
                        record
                    );
                    &(0, 0)
                })
                .0
                .saturating_sub(upstream as u64);

            if start == 0 {
                return;
            }

            let end = three_prime_utr
                .last()
                .unwrap_or_else(|| {
                    log::warn!(
                        "BED12 record {} does not contain a 3UTR interval, will be ignored",
                        record
                    );
                    &(0, 0)
                })
                .1
                .saturating_add(downstream as u64);

            if end == 100 {
                return;
            }

            let extras = Extras::new();
            let mut transcript = GenePred::from_coords(record.chrom().to_vec(), start, end, extras);
            transcript.set_name(Some(name.to_vec()));
            transcript.set_strand(Some(strand));
            transcript.set_thick_start(Some(start));
            transcript.set_thick_end(Some(end));

            let key = format!(
                "{}:{}",
                std::str::from_utf8(record.chrom()).unwrap(),
                strand
            );

            intervals
                .entry(key)
                .or_insert_with(Vec::new)
                .push(transcript);
        });

    log::info!(
        "Build {} interval groups from {}",
        intervals.len(),
        bed.display()
    );
    intervals
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
pub fn get_sequences(sequence: PathBuf) -> HashMap<Vec<u8>, Vec<u8>> {
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
fn from_2bit(twobit: PathBuf) -> HashMap<Vec<u8>, Vec<u8>> {
    let mut genome = TwoBitFile::open_and_read(&twobit).expect("ERROR: Cannot open 2bit file");
    let source = format!("file {}", twobit.display());

    log::debug!("Chromosomes in {}: {:?}", source, genome.chrom_names());

    let mut sequences = HashMap::new();
    genome.chrom_names().iter().for_each(|chr| {
        let seq = genome
            .read_sequence(chr, ..)
            .unwrap_or_else(|e| panic!("ERROR: {}", e))
            .as_bytes()
            .to_vec();

        sequences.insert(chr.as_bytes().to_vec(), seq);
    });

    info!("Read {} sequences from {}", sequences.len(), source);

    sequences
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
pub fn from_fa<P: AsRef<Path>>(f: P) -> HashMap<Vec<u8>, Vec<u8>> {
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

/// Parses a FASTA file.
///
/// # Arguments
///
/// - `reader`: A reader for the FASTA file
/// - `source`: The source of the FASTA file
///
/// # Example
///
/// ```rust,ignore
/// use std::io::BufReader;
///
/// let reader = BufReader::new(File::open("genome.fa").unwrap());
/// let sequences = parse_fasta_reader(reader, "genome.fa");
/// ```
fn parse_fasta_reader<R: BufRead>(mut reader: R, source: &str) -> HashMap<Vec<u8>, Vec<u8>> {
    let mut acc = HashMap::new();
    let mut line = Vec::new();
    let mut header: Option<Vec<u8>> = None;
    let mut seq = Vec::new();

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
                acc.insert(prev_header, std::mem::take(&mut seq));
            }
        } else {
            seq.extend_from_slice(&line);
        }
    }

    if let Some(last_header) = header {
        acc.insert(last_header, seq);
    }

    info!("Read {} sequences from {}", acc.len(), source);

    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::GzDecoder;
    use std::{
        fs,
        io::Read,
        sync::atomic::{AtomicUsize, Ordering},
    };

    static TMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// Creates a temporary directory with a unique name for testing.
    ///
    /// # Arguments
    ///
    /// - `name`: Prefix for the directory name
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dir = temp_dir("my-test");
    /// ```
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "aparent-rs-{name}-{}",
            TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Reads a gzip-compressed file and returns its contents as a String.
    ///
    /// # Arguments
    ///
    /// - `path`: Path to the gzip file
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let content = read_gz_to_string(Path::new("output.tsv.gz"));
    /// ```
    fn read_gz_to_string(path: &Path) -> String {
        let file = File::open(path).unwrap();
        let mut decoder = GzDecoder::new(file);
        let mut data = String::new();
        decoder.read_to_string(&mut data).unwrap();
        data
    }

    /// Tests that chunk output directory resolves to a "chunks" subdirectory.
    #[test]
    fn resolve_chunk_output_dir_uses_chunks_subdir() {
        let dir = temp_dir("resolve-chunks");
        let file_path = dir.join("results.tsv");

        assert_eq!(resolve_chunk_output_dir(&dir), dir.join(CHUNKS_DIR_NAME));
        assert_eq!(
            resolve_chunk_output_dir(&file_path),
            dir.join(CHUNKS_DIR_NAME)
        );
        assert_eq!(
            resolve_chunk_output_dir(Path::new("results.tsv")),
            PathBuf::from(".").join(CHUNKS_DIR_NAME)
        );

        fs::remove_dir_all(dir).unwrap();
    }

    /// Tests that .gz suffix is correctly appended when gzip is enabled.
    #[test]
    fn resolve_single_output_path_handles_gz_suffix() {
        assert_eq!(
            resolve_single_output_path(Path::new("out.tsv"), true),
            PathBuf::from("out.tsv.gz")
        );
        assert_eq!(
            resolve_single_output_path(Path::new("out.tsv.gz"), true),
            PathBuf::from("out.tsv.gz")
        );
        assert_eq!(
            resolve_single_output_path(Path::new("out.tsv"), false),
            PathBuf::from("out.tsv")
        );
    }

    /// Tests that chunk file paths include prefix, zero-padded index, and optional .gz extension.
    #[test]
    fn chunk_file_path_uses_prefix_index_and_gz_suffix() {
        let output_dir = PathBuf::from("/tmp/output");

        assert_eq!(
            chunk_file_path(&output_dir, "part", 2, false),
            output_dir.join("part.00002.tsv")
        );
        assert_eq!(
            chunk_file_path(&output_dir, "part", 2, true),
            output_dir.join("part.00002.tsv.gz")
        );
    }

    /// Tests that ChunkedWriter creates new files when the record limit is reached.
    #[test]
    fn chunked_writer_rotates_every_n_records() {
        let dir = temp_dir("rotate");
        let mut writer = ChunkedWriter::new(dir.clone(), "batch".to_string(), 2, false);

        for idx in 0..5 {
            writer
                .write_record("chr1", idx, idx + 1, genepred::Strand::Forward, b"ACGT")
                .unwrap();
        }

        writer.finish().unwrap();

        let mut files: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        files.sort();

        assert_eq!(files.len(), 3);
        assert_eq!(fs::read_to_string(&files[0]).unwrap().lines().count(), 2);
        assert_eq!(fs::read_to_string(&files[1]).unwrap().lines().count(), 2);
        assert_eq!(fs::read_to_string(&files[2]).unwrap().lines().count(), 1);

        fs::remove_dir_all(dir).unwrap();
    }

    /// Tests that gzip-compressed chunk output produces valid gzip files.
    #[test]
    fn chunked_writer_gzip_outputs_are_valid() {
        let dir = temp_dir("gzip");
        let mut writer = ChunkedWriter::new(dir.clone(), "batch".to_string(), 2, true);

        writer
            .write_record("chr1", 1, 2, genepred::Strand::Forward, b"ACGT")
            .unwrap();
        writer.finish().unwrap();

        let path = dir.join("batch.00001.tsv.gz");
        assert_eq!(read_gz_to_string(&path), "chr1\t1\t2\t+\tACGT\n");

        fs::remove_dir_all(dir).unwrap();
    }
}
