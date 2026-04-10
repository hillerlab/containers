// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
// Distributed under the terms of the Apache License, Version 2.0.

use clap::Parser;
use log::Level;
use std::path::PathBuf;

/// Minimum SpliceAI score to recover from BigWig files.
pub const SPLICE_AI_SCORE_RECOVERY_THRESHOLD: f32 = 0.001;
/// Default minimum derived score value.
pub const DEFAULT_SCORE_FLOOR: i32 = -4;
/// Default maximum derived score value.
pub const DEFAULT_SCORE_CEILING: i32 = 13;

#[derive(Debug, Parser)]
#[command(name = "spliceai-derive", about = "derive spliceAi scores", version = env!("CARGO_PKG_VERSION"))]
pub struct Args {
    #[arg(
        short = 'b',
        long = "bigwig-dir",
        required = true,
        value_name = "PATH",
        help = "Path to directory containing SpliceAI BigWig files; filenames must include donor/acceptor and plus/minus tokens"
    )]
    pub bw_dir: PathBuf,

    #[clap(
        short = 's',
        long = "sequence",
        help = "Path to sequence file (FASTA/2bit, use '-' or omit to read stdin)",
        value_name = "SEQUENCE",
        default_value = "-"
    )]
    pub sequence: PathBuf,

    #[clap(
        short = 'r',
        long = "regions",
        help = "Path to regions file (BED/GTF/GFF/GZ)",
        value_name = "REGIONS",
        required = true
    )]
    pub regions: PathBuf,

    #[arg(
        short = 't',
        long = "threads",
        help = "Number of threads",
        value_name = "THREADS",
        default_value_t = num_cpus::get()
    )]
    pub threads: usize,

    #[arg(
        short = 'l',
        long = "level",
        help = "Log level",
        value_name = "LEVEL",
        default_value_t = log::Level::Info,
    )]
    pub level: Level,

    #[arg(
        short = 'p',
        long = "prefix",
        required = false,
        value_name = "PATH",
        help = "Prefix for output files",
        default_value_t = String::from("spliceai")
    )]
    pub prefix: String,

    #[arg(
        short = 'o',
        long = "outdir",
        required = false,
        value_name = "PATH",
        help = "Output directory",
        default_value = "."
    )]
    pub output_dir: PathBuf,

    #[arg(
        long = "floor",
        required = false,
        value_name = "INT",
        help = "Minimum final rounded derived score",
        default_value_t = DEFAULT_SCORE_FLOOR
    )]
    pub floor: i32,

    #[arg(
        long = "ceiling",
        required = false,
        value_name = "INT",
        help = "Maximum final rounded derived score",
        default_value_t = DEFAULT_SCORE_CEILING
    )]
    pub ceiling: i32,
}
