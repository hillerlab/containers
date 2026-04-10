#!/usr/bin/env python

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

import argparse
from _collections_abc import Sequence
from pathlib import Path
from typing import Union

__author__ = "Alejandro Gonzales-Irribarren"
__email__ = "alejandrxgzi@gmail.com"
__github__ = "https://github.com/alejandrogzi"
__version__ = "0.0.1"


DEFAULT_JUNCTION_LENGTH = 50  # 10bp
DEFAULT_JUNCTION_COVERAGE = 5  # 5 reads

JunctionHash = dict[str, dict[str, Union[str, int]]]


def run(args: argparse.Namespace) -> None:
    """
    Run the junction joining process.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    junction_paths = [Path(j) for j in args.junctions]
    junctions = read_junctions(junction_paths, args.min_junction_length)

    _write_junctions(junctions, args.outdir, args.min_junction_coverage)


def read_junctions(junctions: Sequence[Path], min_junction_length: int) -> JunctionHash:
    """
    Read junctions from a list of files and returns a dictionary of junctions.

    Parameters
    ----------
    junctions : Sequence[Path]
        List of paths to junction files.
    min_junction_length : int
        Minimum length of a junction to be considered.

    Returns
    -------
    JunctionHash
        A dictionary of junctions.
    """
    accumulator: JunctionHash = dict()

    for junction in junctions:
        with open(junction) as f:
            for line in f:
                # INFO: format of junction file:
                # chr start end strand donor/acceptor annotated coverage multimap-cov alignment-overhang
                fields = line.strip().split()

                if len(fields) < 8:
                    continue

                (
                    chr,
                    start,
                    end,
                    strand,
                    splice_sites,
                    annotated,
                    coverage,
                    multimap_coverage,
                    alignment_overhang,
                ) = fields

                if int(end) - int(start) < min_junction_length:
                    print(
                        f"INFO: Skipping junction {chr}:{start}-{end} as it is shorter than {DEFAULT_JUNCTION_LENGTH}"
                    )
                    continue

                # INFO: means that this junction is in GTF [unannotated: 0, annotated: 1]
                if int(annotated) == 1:
                    continue

                key = f"{chr}:{start}-{end}:{strand}"

                if key in accumulator:
                    # INFO: add up coverage, multimap-cov
                    # INFO: max alignment-overhang
                    accumulator[key]["coverage"] += int(coverage)
                    accumulator[key]["multimap_coverage"] += int(multimap_coverage)
                    accumulator[key]["alignment_overhang"] = max(
                        accumulator[key]["alignment_overhang"], int(alignment_overhang)
                    )

                else:
                    # INFO: avoiding annotation -> all of them will be 1 at the end
                    value = {
                        "chr": chr,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "splice_sites": splice_sites,
                        "coverage": int(coverage),
                        "multimap_coverage": int(multimap_coverage),
                        "alignment_overhang": int(alignment_overhang),
                    }

                    accumulator[key] = value

    return accumulator


def _write_junctions(
    junctions: JunctionHash, outdir: Path, min_junction_coverage: int
) -> None:
    """
    Write junctions to a file.

    Parameters
    ----------
    junctions : JunctionHash
        A dictionary of junctions.
    outdir : Path
        Path to the output directory.
    min_junction_coverage : int
        Minimum coverage of a junction to be considered.
    """
    with open(outdir / "ALL_SJ_out_filtered.tab", "w") as f:
        for _key, value in junctions.items():
            if value["coverage"] < min_junction_coverage:
                continue
            line = f"{value['chr']}\t{value['start']}\t{value['end']}\t{value['strand']}\t{value['splice_sites']}\t1\t{value['coverage']}\t{value['multimap_coverage']}\t{value['alignment_overhang']}"
            f.write(line)
            f.write("\n")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments

    Examples
    --------
    >>> args = parse_arguments()
    """
    parser = argparse.ArgumentParser(
        description="Build pan-genome indexes using DEACON with optional background datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -g genome1.fa,genome2.fa -o output/
  %(prog)s -g genome.fa -F -R -k 31 -w 15 -t 16
        """,
    )

    parser.add_argument(
        "-j",
        "--junctions",
        required=True,
        nargs="+",
        metavar="JUNCTION1 JUNCTION2 ...",
        help="Space-separated list of junctions",
    )
    parser.add_argument(
        "-l",
        "--min-junction-length",
        type=int,
        default=DEFAULT_JUNCTION_LENGTH,
        metavar="INT",
        help=f"Minimum length for junctions (default: {DEFAULT_JUNCTION_LENGTH})",
    )
    parser.add_argument(
        "-m",
        "--min-junction-coverage",
        type=int,
        default=DEFAULT_JUNCTION_COVERAGE,
        metavar="INT",
        help=f"Minimum coverage for junctions (default: {DEFAULT_JUNCTION_COVERAGE})",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=".",
        metavar="DIR",
        help="Output directory for indexes and downloads (default: current directory)",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
