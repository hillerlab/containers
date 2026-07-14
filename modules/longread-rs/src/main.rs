//! `longread` command-line interface.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

use longread::error::{Error, Result};
use longread::generate::{EventWeights, NewIsoformsMode};
use longread::model::EventType;
use longread::pacbio::check::{self, CheckParams};
use longread::pacbio::normalize::{self, NormalizeParams};
use longread::pbsim::{self, PbsimParams};
use longread::prepare::{self, PrepareParams};
use longread::split::{self, SplitParams};
use longread::validate;

/// BED12-native transcript inventory and expression engine.
#[derive(Debug, Parser)]
#[command(name = "longread", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Build the final transcript inventory and assign gene/isoform expression.
    Prepare(PrepareArgs),
    /// Validate a BED12 annotation and its transcript→gene mapping.
    Validate(ValidateArgs),
    /// Build the PBSIM3 transcript-mode input from sequences and isoform depths.
    Pbsim(PbsimArgs),
    /// Split a PBSIM3 transcript file into balanced chunks.
    Split(SplitArgs),
    /// Normalize PBSIM3 subread BAMs into one synthetic PacBio movie with globally unique ZMWs
    /// and one specification-compliant `SUBREAD` read group.
    Rg(RgArgs),
    /// Validate that CCS chunks do not overlap in PacBio's movie/ZMW key space.
    Check(CheckArgs),
}

#[derive(Debug, Args)]
struct PrepareArgs {
    /// Input BED12 transcript annotation.
    #[arg(long)]
    bed: PathBuf,
    /// Transcript→gene mapping TSV.
    #[arg(long)]
    transcript_gene: PathBuf,
    /// Chromosome sizes file (enables coordinate-bounds validation).
    #[arg(long)]
    chrom_sizes: Option<PathBuf>,
    /// Output prefix (may include directories).
    #[arg(long)]
    output_prefix: String,

    /// Global RNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Threads (0 = all available cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Fixed number of new isoforms per gene (mutually exclusive with the mean/max options).
    #[arg(long)]
    new_isoforms_per_gene: Option<u32>,
    /// Poisson mean for new isoforms per gene.
    #[arg(long)]
    mean_new_isoforms_per_gene: Option<f64>,
    /// Poisson cap for new isoforms per gene.
    #[arg(long)]
    max_new_isoforms_per_gene: Option<u32>,

    /// Event weights, e.g. `donor=1,acceptor=1,skip=1,retention=1,truncation=1`.
    #[arg(long)]
    event_weights: Option<String>,
    /// Maximum attempts per requested isoform.
    #[arg(long, default_value_t = 100)]
    max_event_attempts: u32,
    /// Minimum transcript (exonic) length.
    #[arg(long, default_value_t = 0)]
    minimum_transcript_length: u64,

    /// Requested number of unique fusion transcripts.
    #[arg(long, default_value_t = 0)]
    fusion_count: u32,
    /// Minimum genomic intron between fusion partners.
    #[arg(long, default_value_t = 1)]
    min_fusion_intron: u64,
    /// Maximum genomic distance between fusion partner gene spans (unlimited if unset).
    #[arg(long)]
    max_fusion_distance: Option<u64>,
    /// Multiplier applied to fusion pseudo-gene raw expression weights.
    #[arg(long, default_value_t = 1.0)]
    fusion_expression_scale: f64,

    /// Total molecule budget across all genes.
    #[arg(long, default_value_t = 100_000)]
    total_molecules: u64,
    /// Zipf coefficient for within-gene isoform allocation.
    #[arg(long, default_value_t = 4.0)]
    alpha: f64,

    /// Fail if the requested event/fusion counts are not met exactly.
    #[arg(long, default_value_t = false)]
    require_exact_event_count: bool,
}

#[derive(Debug, Args)]
struct ValidateArgs {
    /// Input BED12 transcript annotation.
    #[arg(long)]
    bed: PathBuf,
    /// Transcript→gene mapping TSV.
    #[arg(long)]
    transcript_gene: PathBuf,
    /// Chromosome sizes file (enables coordinate-bounds validation).
    #[arg(long)]
    chrom_sizes: Option<PathBuf>,

    /// Minimum transcript length (exonic length).
    #[arg(long, default_value_t = 100)]
    min_transcript_length: u64,
}

#[derive(Debug, Args)]
struct PbsimArgs {
    /// Extracted transcript sequences (`xloci --as-tsv`): TRANSCRIPT_ID<TAB>SEQUENCE.
    #[arg(long)]
    sequences: PathBuf,
    /// `${prefix}.isoform_depth.tsv` from `longread prepare`.
    #[arg(long)]
    isoform_depth: PathBuf,
    /// Optional `${prefix}.manifest.tsv` to validate fusion sequences against their components.
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Output PBSIM3 transcript-mode file.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct SplitArgs {
    /// Input PBSIM3 four-column transcript file.
    #[arg(long)]
    transcript: PathBuf,
    /// Number of chunks (bins).
    #[arg(long = "pbsim-chunks", default_value_t = 1)]
    pbsim_chunks: usize,
    /// Number of sequencing passes (for the work estimate).
    #[arg(long, default_value_t = 1)]
    pass_count: u64,
    /// Output directory for chunk files and `chunks.tsv`.
    #[arg(long)]
    outdir: PathBuf,
    /// Prefix stem for chunk task prefixes.
    #[arg(long, default_value = "simulation")]
    prefix: String,
}

#[derive(Debug, Args)]
struct RgArgs {
    /// File listing the input subread BAMs, one path per line.
    #[arg(long)]
    bams: PathBuf,
    /// Synthetic movie name (e.g. `movie.<id>`); sanitized to `[A-Za-z0-9_.-]`.
    #[arg(long)]
    movie: String,
    /// Output directory for `*.normalized.bam` files.
    #[arg(long, default_value = ".")]
    outdir: PathBuf,
    /// Output path for the ZMW map TSV.
    #[arg(long, default_value = "zmw_map.tsv")]
    zmw_map: PathBuf,
    /// Threads (0 = all available cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Upper bound on simultaneously open file descriptors.
    #[arg(long, default_value_t = 512)]
    max_open_files: usize,
}

#[derive(Debug, Args)]
struct CheckArgs {
    /// File listing the input CCS chunk BAMs, one path per line.
    #[arg(long)]
    bams: PathBuf,
    /// Marker file to create on success.
    #[arg(long, default_value = "ccs_chunks.valid")]
    out: PathBuf,
    /// Threads (0 = all available cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Upper bound on simultaneously open file descriptors.
    #[arg(long, default_value_t = 512)]
    max_open_files: usize,
}

/// Parse `donor=1,acceptor=1,...` into [`EventWeights`], starting from a uniform baseline.
fn parse_event_weights(spec: &str) -> Result<EventWeights> {
    let mut weights = [1.0f64; 5];
    for token in spec.split(',').filter(|s| !s.trim().is_empty()) {
        let (key, value) = token.split_once('=').ok_or_else(|| {
            Error::config(format!(
                "invalid event weight '{token}' (expected key=value)"
            ))
        })?;
        let event = EventType::from_weight_key(key.trim()).ok_or_else(|| {
            Error::config(format!(
                "unknown event weight key '{key}' (expected donor|acceptor|skip|retention|truncation)"
            ))
        })?;
        let w: f64 = value
            .trim()
            .parse()
            .map_err(|_| Error::config(format!("invalid event weight value '{value}'")))?;
        if w < 0.0 || !w.is_finite() {
            return Err(Error::config(format!(
                "event weight must be finite and >= 0, got '{value}'"
            )));
        }
        let idx = EventType::ALL.iter().position(|&e| e == event).unwrap();
        weights[idx] = w;
    }
    Ok(EventWeights::from_array(weights))
}

/// Resolve the new-isoform generation mode, enforcing mutual exclusivity.
fn resolve_mode(args: &PrepareArgs) -> Result<NewIsoformsMode> {
    match args.new_isoforms_per_gene {
        Some(n) => {
            if args.mean_new_isoforms_per_gene.is_some() || args.max_new_isoforms_per_gene.is_some()
            {
                return Err(Error::config(
                    "--new-isoforms-per-gene is mutually exclusive with --mean-new-isoforms-per-gene / --max-new-isoforms-per-gene",
                ));
            }
            Ok(NewIsoformsMode::Fixed(n))
        }
        None => {
            let mean = args.mean_new_isoforms_per_gene.unwrap_or(1.0);
            let max = args.max_new_isoforms_per_gene.unwrap_or(5);
            if mean < 0.0 || !mean.is_finite() {
                return Err(Error::config(
                    "--mean-new-isoforms-per-gene must be finite and >= 0",
                ));
            }
            Ok(NewIsoformsMode::PoissonCapped { mean, max })
        }
    }
}

fn run_prepare(args: &PrepareArgs) -> Result<()> {
    let mode = resolve_mode(args)?;
    let weights = match &args.event_weights {
        Some(spec) => parse_event_weights(spec)?,
        None => EventWeights::uniform(),
    };
    if !weights.any_positive() {
        return Err(Error::config("at least one event weight must be positive"));
    }
    if args.alpha <= 0.0 || !args.alpha.is_finite() {
        return Err(Error::config("--alpha must be finite and > 0"));
    }
    if args.fusion_expression_scale < 0.0 || !args.fusion_expression_scale.is_finite() {
        return Err(Error::config(
            "--fusion-expression-scale must be finite and >= 0",
        ));
    }

    let params = PrepareParams {
        bed: args.bed.clone(),
        transcript_gene: args.transcript_gene.clone(),
        chrom_sizes: args.chrom_sizes.clone(),
        output_prefix: args.output_prefix.clone(),
        seed: args.seed,
        threads: args.threads,
        new_isoforms_mode: mode,
        event_weights: weights,
        max_event_attempts: args.max_event_attempts,
        min_transcript_length: args.minimum_transcript_length,
        fusion_count: args.fusion_count,
        min_fusion_intron: args.min_fusion_intron,
        max_fusion_distance: args.max_fusion_distance,
        fusion_expression_scale: args.fusion_expression_scale,
        total_molecules: args.total_molecules,
        alpha: args.alpha,
        require_exact_event_count: args.require_exact_event_count,
    };

    let (stats, outputs) = prepare::run(&params)?;
    eprintln!(
        "longread prepare: {} input transcripts, {} genes -> {} output transcripts ({} fusions)",
        stats.input_transcripts,
        stats.input_genes,
        stats.output_transcripts,
        stats.successful_fusions,
    );
    eprintln!(
        "  molecules: gene={} isoform={} (requested {})",
        stats.total_allocated_gene_molecules,
        stats.total_allocated_isoform_molecules,
        stats.total_requested_molecules,
    );
    eprintln!("  wrote: {}", outputs.isoforms_bed.display());
    eprintln!("         {}", outputs.transcript_gene.display());
    eprintln!("         {}", outputs.gene_depth.display());
    eprintln!("         {}", outputs.isoform_depth.display());
    eprintln!("         {}", outputs.manifest.display());
    eprintln!("         {}", outputs.stats.display());
    Ok(())
}

fn run_validate(args: &ValidateArgs) -> Result<()> {
    let input = validate::load_and_validate(
        &args.bed,
        &args.transcript_gene,
        args.chrom_sizes.as_ref(),
        0,
        args.min_transcript_length,
    )?;

    eprintln!(
        "longread validate: OK — {} transcripts across {} genes",
        input.input_transcript_count,
        input.genes.len(),
    );
    Ok(())
}

fn run_pbsim(args: &PbsimArgs) -> Result<()> {
    let params = PbsimParams {
        sequences: args.sequences.clone(),
        isoform_depth: args.isoform_depth.clone(),
        manifest: args.manifest.clone(),
        output: args.output.clone(),
    };
    let stats = pbsim::run(&params)?;
    eprintln!(
        "longread pbsim: wrote {} expressed transcripts ({} zero-count omitted, {} fusions validated) -> {}",
        stats.written,
        stats.omitted,
        stats.fusions_validated,
        params.output.display(),
    );
    Ok(())
}

fn run_split(args: &SplitArgs) -> Result<()> {
    let params = SplitParams {
        transcript: args.transcript.clone(),
        chunks: args.pbsim_chunks,
        pass_count: args.pass_count,
        outdir: args.outdir.clone(),
        prefix: args.prefix.clone(),
    };
    let stats = split::run(&params)?;
    eprintln!(
        "longread split: {} transcripts -> {} chunk(s) ({})",
        stats.transcripts,
        stats.chunks_written,
        stats.chunks_manifest.display(),
    );
    Ok(())
}

/// Read a BAM-list file (one path per line; blank lines ignored).
fn read_bam_list(path: &PathBuf) -> Result<Vec<PathBuf>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| Error::pacbio(format!("cannot read BAM list {}: {e}", path.display())))?;
    let bams: Vec<PathBuf> = content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .collect();
    if bams.is_empty() {
        return Err(Error::pacbio(format!(
            "BAM list {} is empty",
            path.display()
        )));
    }
    Ok(bams)
}

fn run_rg(args: &RgArgs) -> Result<()> {
    let params = NormalizeParams {
        bams: read_bam_list(&args.bams)?,
        movie: args.movie.clone(),
        outdir: args.outdir.clone(),
        zmw_map: args.zmw_map.clone(),
        threads: args.threads,
        max_open_files: args.max_open_files,
    };
    let stats = normalize::run(&params)?;
    eprintln!(
        "longread rg: {} BAM(s), {} source file(s), {} record(s) -> {} normalized BAM(s)",
        stats.input_bams,
        stats.source_files,
        stats.records,
        stats.outputs.len(),
    );
    eprintln!(
        "  global ZMW capacity {}, read group {}",
        stats.zmw_capacity, stats.rg_id,
    );
    Ok(())
}

fn run_check(args: &CheckArgs) -> Result<()> {
    let params = CheckParams {
        bams: read_bam_list(&args.bams)?,
        out: args.out.clone(),
        threads: args.threads,
        max_open_files: args.max_open_files,
    };
    let stats = check::run(&params)?;
    eprintln!(
        "longread check: OK — {} BAM(s), {} record(s), {} distinct movie/ZMW key(s)",
        stats.input_bams, stats.records, stats.distinct_keys,
    );
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Prepare(args) => run_prepare(args),
        Command::Validate(args) => run_validate(args),
        Command::Pbsim(args) => run_pbsim(args),
        Command::Split(args) => run_split(args),
        Command::Rg(args) => run_rg(args),
        Command::Check(args) => run_check(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
