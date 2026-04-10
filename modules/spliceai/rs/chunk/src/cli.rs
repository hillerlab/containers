// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
// Distributed under the terms of the Apache License, Version 2.0.

use clap::{ArgAction, Parser};
use log::Level;
use std::path::PathBuf;

use crate::core::Config;

pub const DEFAULT_CHUNK_SIZE: u32 = 6_000_000;
pub const DEFAULT_FLANK_SIZE: u32 = 50_000;
pub const DEFAULT_MIN_CONTIG_SIZE: u32 = 500;

#[derive(Debug, Parser)]
#[command(
    name = "spliceai-chunk",
    about = "Chunk a genome FASTA/2bit into SpliceAI-ready FASTA shards",
    version = env!("CARGO_PKG_VERSION")
)]
pub struct Args {
    #[arg(
        short = 's',
        long = "sequence",
        help = "Path to input sequence (.fa, .fa.gz, .fna, .2bit) or '-' for stdin",
        value_name = "SEQUENCE",
        default_value = "-"
    )]
    pub sequence: PathBuf,

    #[arg(
        short = 't',
        long = "threads",
        help = "Number of Rayon worker threads",
        value_name = "THREADS",
        default_value_t = num_cpus::get()
    )]
    pub threads: usize,

    #[arg(
        short = 'l',
        long = "level",
        help = "Log level",
        value_name = "LEVEL",
        default_value_t = Level::Info
    )]
    pub level: Level,

    #[arg(
        short = 'c',
        long = "chunk-size",
        help = "Target chunk size in bases before adding flanks",
        value_name = "BP",
        default_value_t = DEFAULT_CHUNK_SIZE
    )]
    pub chunk_size: u32,

    #[arg(
        short = 'f',
        long = "flank-size",
        help = "Flank size in bases added to both chunk ends",
        value_name = "BP",
        default_value_t = DEFAULT_FLANK_SIZE
    )]
    pub flank_size: u32,

    #[arg(
        short = 'm',
        long = "min-contig-size",
        help = "Skip contigs shorter than this size",
        value_name = "BP",
        default_value_t = DEFAULT_MIN_CONTIG_SIZE
    )]
    pub min_contig_size: u32,

    #[arg(
        short = 'z',
        long = "gzip",
        help = "Compress each output chunk as .fa.gz",
        action = ArgAction::SetTrue
    )]
    pub gzip: bool,

    #[arg(
        short = 'o',
        long = "outdir",
        help = "Output directory",
        value_name = "PATH",
        default_value = "."
    )]
    pub outdir: PathBuf,
}

impl Args {
    /// Converts command-line arguments to a Config struct.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use xloci::cli::Args;
    ///
    /// let args = Args::parse_from(["spliceai-chunk", "-s", "genome.fa"]);
    /// let config = args.config();
    /// ```
    pub fn config(&self) -> Config {
        Config {
            chunk_size: self.chunk_size,
            flank_size: self.flank_size,
            min_contig_size: self.min_contig_size,
            gzip: self.gzip,
            outdir: self.outdir.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_gzip_to_false_and_outdir_to_current_directory() {
        let args = Args::parse_from(["spliceai-chunk"]);

        assert!(!args.gzip);
        assert_eq!(args.outdir, PathBuf::from("."));
    }

    #[test]
    fn cli_enables_gzip_flag_when_requested() {
        let args = Args::parse_from(["spliceai-chunk", "--gzip"]);

        assert!(args.gzip);
    }
}
