// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
// Distributed under the terms of the Apache License, Version 2.0.

use clap::Parser;
use log::error;
use simple_logger::init_with_level;

use spliceai_chunk::{cli::Args, core};

fn main() {
    let start = std::time::Instant::now();
    let args = Args::parse();

    init_with_level(args.level).unwrap_or_else(|e| {
        panic!("ERROR: Cannot initialize logger: {}", e);
    });

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap_or_else(|e| {
            error!("ERROR: failed to initialize Rayon thread pool: {}", e);
            std::process::exit(1);
        });

    core::run(args.sequence.clone(), args.config());
    log::info!("Finished chunking in {:?}", start.elapsed());
}
