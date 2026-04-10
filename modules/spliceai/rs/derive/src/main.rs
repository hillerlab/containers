// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
// Distributed under the terms of the Apache License, Version 2.0.

use clap::Parser;
use log::error;
use simple_logger::init_with_level;

use spliceai_derive::{
    annotation::get_ss_from_annotation,
    calibration::{build_null_profile, derive_spliceai_score, make_bins},
    cli::Args,
    genome::get_dinucleotide_count,
    spliceai::get_splice_scores,
};

fn main() {
    let args = Args::parse();
    init_with_level(args.level).unwrap_or_else(|_| panic!("ERROR: Cannot initialize logger!"));

    if args.floor > args.ceiling {
        error!(
            "ERROR: floor ({}) cannot be greater than ceiling ({})",
            args.floor, args.ceiling
        );
        std::process::exit(1);
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .unwrap_or_else(|e| {
            error!("ERROR: failed to initialize thread pool: {}", e);
            std::process::exit(1);
        });

    let (genome, dinucleotide_count) = get_dinucleotide_count(&args.sequence).unwrap_or_else(|e| {
        error!("ERROR: failed to get dinucleotide counts: {}", e);
        std::process::exit(1);
    });
    let chroms = genome.keys().cloned().collect::<Vec<Vec<u8>>>();

    let (donors, acceptors, ss_coord_donors, ss_coord_acceptors) =
        get_ss_from_annotation(&genome, args.regions);

    let donor_profile = build_null_profile("donor", &donors, &dinucleotide_count);
    let acceptor_profile = build_null_profile("acceptor", &acceptors, &dinucleotide_count);

    let (spliceai_donor_scores, spliceai_acceptor_scores) =
        get_splice_scores(args.bw_dir, chroms, genome).unwrap_or_else(|e| {
            error!("ERROR: failed to load SpliceAI BigWigs: {}", e);
            std::process::exit(1);
        });

    let donor_bins = make_bins(
        &spliceai_donor_scores,
        &ss_coord_donors,
        &donors,
        &dinucleotide_count,
    );
    let acceptor_bins = make_bins(
        &spliceai_acceptor_scores,
        &ss_coord_acceptors,
        &acceptors,
        &dinucleotide_count,
    );

    derive_spliceai_score(
        &spliceai_donor_scores,
        &spliceai_acceptor_scores,
        &donor_bins,
        &acceptor_bins,
        &donor_profile,
        &acceptor_profile,
        args.floor,
        args.ceiling,
        &args.prefix,
        &args.output_dir,
    )
    .unwrap_or_else(|e| {
        error!("ERROR: failed to derive splice scores: {}", e);
        std::process::exit(1);
    });
}
