#!/usr/bin/env python3

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

"""
Run SpliceAI on FASTA or FASTA.GZ input chunks and write WIG tracks.
"""

from __future__ import annotations

__author__ = "Alejandro Gonzales-Irribarren"
__credits__ = ["Yury V. Malovichko", "Michael Hiller"]
__email__ = "alejandrxgzi@gmail.com"
__github__ = "https://github.com/alejandrogzi"
__version__ = "0.0.2"

import argparse
import gzip
import importlib.metadata as importlib_metadata
import re
import sys
from pathlib import Path
from typing import Iterator, Sequence, TextIO

FASTA_HEADER_START = ">"
WIGGLE_HEADER_TEMPLATE = "fixedStep chrom={} start={} step=1 span=1\n"
HEADER_PATTERN = re.compile(
    r"^(?P<chrom>.+):(?P<region_start>\d+)-(?P<region_end>\d+)\((?P<strand>[+-])\)$"
)
MODEL_FILENAMES = tuple(f"spliceai{x}.h5" for x in range(1, 6))
CONTEXT = 10000
DEFAULT_FLANK_SIZE = 50000


class Logger:
    """Logger for verbose output with configurable levels.

    Example
    -------
    >>> logger = Logger(verbose=True)
    >>> logger.info("Processing file")
    [INFO] Processing file
    """

    def __init__(self, verbose: bool = False, stream: TextIO = sys.stderr) -> None:
        self.verbose = verbose
        self.stream = stream

    def _emit(self, level: str, message: str, *, force: bool = True) -> None:
        if not force and not self.verbose:
            return
        print(f"[{level}] {message}", file=self.stream)

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warn(self, message: str) -> None:
        self._emit("WARN", message)

    def error(self, message: str) -> None:
        self._emit("ERROR", message)

    def debug(self, message: str) -> None:
        self._emit("DEBUG", message, force=False)


def existing_file(value: str) -> Path:
    """Validate that a path exists and is a file.

    Parameters
    ----------
    value : str
        Path string to validate

    Returns
    -------
    Path
        Validated Path object

    Example
    -------
    >>> existing_file("data/test.fasta")
    PosixPath('data/test.fasta')
    """
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {value}")
    return path


def positive_int(value: str) -> int:
    """Validate that a string is a positive integer (> 0).

    Parameters
    ----------
    value : str
        String to parse as integer

    Returns
    -------
    int
        Parsed positive integer

    Example
    -------
    >>> positive_int("10")
    10
    """
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be >= 1")
    return parsed


def non_negative_int(value: str) -> int:
    """Validate that a string is a non-negative integer (>= 0).

    Parameters
    ----------
    value : str
        String to parse as integer

    Returns
    -------
    int
        Parsed non-negative integer

    Example
    -------
    >>> non_negative_int("0")
    0
    """
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0")
    return parsed


def probability(value: str) -> float:
    """Validate that a string is a float in [0.0, 1.0].

    Parameters
    ----------
    value : str
        String to parse as float

    Returns
    -------
    float
        Parsed probability value

    Example
    -------
    >>> probability("0.5")
    0.5
    """
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("Value must be between 0.0 and 1.0")
    return parsed


def fail(logger: Logger, message: str) -> None:
    """Log an error message and exit with code 1.

    Parameters
    ----------
    logger : Logger
        Logger instance for error output
    message : str
        Error message to display

    Example
    -------
    >>> fail(logger, "Invalid input")
    [ERROR] Invalid input
    """
    logger.error(message)
    raise SystemExit(1)


def build_output_prefix(sequence_path: Path) -> str:
    """Build output prefix from sequence file path.

    Strips .gz suffix and common FASTA extensions to generate
    a clean prefix for output files.

    Parameters
    ----------
    sequence_path : Path
        Path to the input FASTA file

    Returns
    -------
    str
        Clean prefix for output files

    Example
    -------
    >>> build_output_prefix(Path("sample.fasta.gz"))
    'sample'
    """
    name = sequence_path.name
    if name.endswith(".gz"):
        name = name[:-3]
    for suffix in (".fasta", ".fa", ".fna", ".fas"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    stem = Path(name).stem
    return stem or name


def is_gzip_file(path: Path) -> bool:
    """Check if a file is gzip compressed.

    Parameters
    ----------
    path : Path
        Path to file to check

    Returns
    -------
    bool
        True if file starts with gzip magic number

    Example
    -------
    >>> is_gzip_file(Path("input.fasta.gz"))
    True
    """
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def iter_fasta_records(
    sequence_path: Path, logger: Logger
) -> Iterator[tuple[str, str]]:
    """Iterate over FASTA records in a file (supports gzip).

    Parameters
    ----------
    sequence_path : Path
        Path to FASTA or FASTA.GZ file
    logger : Logger
        Logger instance for debug output

    Yields
    ------
    tuple[str, str]
        Header (without >) and sequence for each record

    Example
    -------
    >>> list(iter_fasta_records(Path("test.fa"), logger))
    [('chr1:100-200(+)', 'ACGTACGT')]
    """
    compressed = is_gzip_file(sequence_path)
    logger.debug(
        f"Detected {'gzip-compressed' if compressed else 'plain'} FASTA input: {sequence_path}"
    )
    opener = gzip.open if compressed else open

    with opener(sequence_path, "rt", encoding="utf-8") as handle:
        header: str | None = None
        sequence_chunks: list[str] = []
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(FASTA_HEADER_START):
                if header is not None:
                    yield header, "".join(sequence_chunks)
                header = line[1:]
                sequence_chunks = []
                continue
            if header is None:
                raise ValueError(
                    f"Invalid FASTA formatting at line {line_number}: sequence data before header"
                )
            sequence_chunks.append(line)

        if header is not None:
            yield header, "".join(sequence_chunks)


def parse_header(header: str, logger: Logger) -> tuple[str, int, int, bool]:
    """Parse FASTA header into genomic coordinates.

    Expected format: {chrom}:{start}-{end}({strand})

    Parameters
    ----------
    header : str
        FASTA header string (without >)
    logger : Logger
        Logger instance for error output

    Returns
    -------
    tuple[str, int, int, bool]
        Chromosome, start, end, and strand (+ as True)

    Example
    -------
    >>> parse_header("chr1:100-200(+)", logger)
    ('chr1', 100, 200, True)
    """
    match = HEADER_PATTERN.fullmatch(header)
    if match is None:
        fail(
            logger,
            (
                "Unsupported FASTA header format. Expected "
                "{chr}:{flank_start}-{flank_end}({strand}), got: "
                f"{header}"
            ),
        )

    chrom = match.group("chrom")
    start = int(match.group("region_start"))
    end = int(match.group("region_end"))
    strand = match.group("strand") == "+"

    if end <= start:
        fail(logger, f"Sequence {header} has invalid coordinates: end must be > start")

    return chrom, start, end, strand


def one_hot_encode(seq: str):
    """One-hot encode a DNA sequence.

    Parameters
    ----------
    seq : str
        DNA sequence string (A, C, G, T, N)

    Returns
    -------
    np.ndarray
        One-hot encoded array of shape (len(seq), 4)

    Example
    -------
    >>> one_hot_encode("ACGT").shape
    (4, 4)
    """
    import numpy as np

    encoding_map = np.asarray(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    seq = seq.upper().replace("A", "\x01").replace("C", "\x02")
    seq = seq.replace("G", "\x03").replace("T", "\x04").replace("N", "\x00")

    return encoding_map[np.frombuffer(seq.encode("latin1"), dtype=np.int8) % 5]


def bundled_model_paths(logger: Logger) -> list[Path]:
    """Get paths to bundled SpliceAI model files.

    Parameters
    ----------
    logger : Logger
        Logger instance for error output

    Returns
    -------
    list[Path]
        List of paths to the 5 bundled model .h5 files

    Example
    -------
    >>> paths = bundled_model_paths(logger)
    >>> len(paths)
    5
    """
    try:
        distribution = importlib_metadata.distribution("spliceai")
    except importlib_metadata.PackageNotFoundError:
        fail(
            logger,
            (
                "The Python package 'spliceai' is not installed. Install it before running "
                "this script, or use the container image that bundles the models."
            ),
        )

    files = distribution.files or []
    file_map = {str(file): file for file in files}
    model_paths: list[Path] = []

    for model_filename in MODEL_FILENAMES:
        relative_path = f"spliceai/models/{model_filename}"
        package_path = file_map.get(relative_path)
        if package_path is None:
            fail(
                logger,
                f"Bundled SpliceAI model not found in installed package: {relative_path}",
            )
        model_paths.append(Path(distribution.locate_file(package_path)))

    return model_paths


def load_models(logger: Logger) -> list[object]:
    """Load SpliceAI models from bundled paths.

    Parameters
    ----------
    logger : Logger
        Logger instance for progress output

    Returns
    -------
    list[object]
        List of loaded Keras models

    Example
    -------
    >>> models = load_models(logger)
    >>> len(models)
    5
    """
    from keras.models import load_model

    model_paths = bundled_model_paths(logger)
    logger.info(f"Loading {len(model_paths)} bundled model(s)")
    logger.debug("Model paths: " + ", ".join(str(path) for path in model_paths))
    models = [load_model(str(model_path)) for model_path in model_paths]
    logger.info("Finished loading model(s)")
    return models


def write_probabilities(
    handle: TextIO, probabilities, round_to: int, min_prob: float
) -> None:
    """Write probability values to WIG file.

    Parameters
    ----------
    handle : TextIO
        Output file handle
    probabilities : iterable
        Probability values to write
    round_to : int
        Number of decimal places to round to
    min_prob : float
        Minimum probability threshold

    Example
    -------
    >>> import io
    >>> buf = io.StringIO()
    >>> write_probabilities(buf, [0.5, 0.001], 2, 0.01)
    >>> buf.getvalue()
    '0.5\\n0.0\\n'
    """
    for value in probabilities:
        parsed = round(float(value), round_to) if value >= min_prob else 0.0
        handle.write(f"{parsed}\n")


def process_record(
    header: str,
    seq: str,
    models: Sequence[object],
    round_to: int,
    min_prob: float,
    offset: int,
    wig_handles: tuple[TextIO, TextIO, TextIO, TextIO],
    logger: Logger,
) -> None:
    """Process one FASTA record and write WIG output.

    Parameters
    ----------
    header : str
        FASTA header with genomic coordinates
    seq : str
        DNA sequence
    models : Sequence[object]
        List of loaded SpliceAI models
    round_to : int
        Decimal places for rounding probabilities
    min_prob : float
        Minimum probability threshold
    offset : int
        Flank size used for chunking
    wig_handles : tuple
        Tuple of (acc_plus, donor_plus, acc_minus, donor_minus) handles
    logger : Logger
        Logger instance for debug output

    Example
    -------
    # ruff: noqa: E501
    >>> process_record("chr1:100-200(+)", "ACGT"*50, models, 4, 0.001, 1000, handles, logger)
    """
    import numpy as np

    chrom, start, end, strand = parse_header(header, logger)
    seq = seq.upper()
    chunk_length = end - start

    if not seq:
        fail(logger, f"Sequence {header} is empty")
    if len(seq) < chunk_length:
        fail(
            logger,
            (
                f"Sequence {header} has length {len(seq)}, shorter than the inner chunk "
                f"length {chunk_length}"
            ),
        )

    start_offset = min(offset, start)
    end_offset = len(seq) - chunk_length - start_offset
    if end_offset < 0 or end_offset > offset:
        fail(
            logger,
            (
                f"Sequence {header} has inconsistent flanks for --offset {offset}: "
                f"derived start_offset={start_offset}, end_offset={end_offset}, "
                f"sequence_length={len(seq)}"
            ),
        )

    logger.debug(
        f"{header}: chunk_length={chunk_length}, start_offset={start_offset}, "
        f"end_offset={end_offset}, sequence_length={len(seq)}"
    )

    x = one_hot_encode("N" * (CONTEXT // 2) + seq + "N" * (CONTEXT // 2))[None, :]
    y = np.mean([model.predict(x, verbose=0) for model in models], axis=0)

    acceptor_prob = y[0, :, 1]
    donor_prob = y[0, :, 2]
    wiggle_header = WIGGLE_HEADER_TEMPLATE.format(chrom, start + 2)

    acc_plus_handle, donor_plus_handle, acc_minus_handle, donor_minus_handle = (
        wig_handles
    )

    if strand:
        acc_plus_handle.write(wiggle_header)
        donor_plus_handle.write(wiggle_header)

        start_index = start_offset
        end_index = len(seq) - end_offset
        acceptor_prob = acceptor_prob[start_index:end_index]
        donor_prob = donor_prob[start_index:end_index]

        write_probabilities(donor_plus_handle, donor_prob, round_to, min_prob)
        write_probabilities(acc_plus_handle, acceptor_prob, round_to, min_prob)
        return

    acc_minus_handle.write(wiggle_header)
    donor_minus_handle.write(wiggle_header)

    start_index = end_offset
    end_index = len(seq) - start_offset
    acceptor_prob = acceptor_prob[start_index:end_index][::-1]
    donor_prob = donor_prob[start_index:end_index][::-1]

    write_probabilities(donor_minus_handle, donor_prob, round_to, min_prob)
    write_probabilities(acc_minus_handle, acceptor_prob, round_to, min_prob)


def run(args: argparse.Namespace, logger: Logger) -> None:
    """Run SpliceAI prediction on FASTA input.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    logger : Logger
        Logger instance for progress output

    Example
    -------
    >>> args = parser()
    >>> logger = Logger()
    >>> run(args, logger)
    [INFO] Writing output to output/
    """
    args.outdir.mkdir(parents=True, exist_ok=True)

    prefix = build_output_prefix(args.sequence)
    acc_plus_file = args.outdir / f"{prefix}.acceptor_plus.wig"
    donor_plus_file = args.outdir / f"{prefix}.donor_plus.wig"
    acc_minus_file = args.outdir / f"{prefix}.acceptor_minus.wig"
    donor_minus_file = args.outdir / f"{prefix}.donor_minus.wig"

    logger.info(f"Writing output to {args.outdir}")
    logger.debug(f"Output prefix: {prefix}")

    models = load_models(logger)

    records_processed = 0
    with (
        acc_plus_file.open("w", encoding="utf-8") as acc_plus_handle,
        donor_plus_file.open("w", encoding="utf-8") as donor_plus_handle,
        acc_minus_file.open("w", encoding="utf-8") as acc_minus_handle,
        donor_minus_file.open("w", encoding="utf-8") as donor_minus_handle,
    ):
        for header, seq in iter_fasta_records(args.sequence, logger):
            process_record(
                header=header,
                seq=seq,
                models=models,
                round_to=args.round_to,
                min_prob=args.min_prob,
                offset=args.offset,
                wig_handles=(
                    acc_plus_handle,
                    donor_plus_handle,
                    acc_minus_handle,
                    donor_minus_handle,
                ),
                logger=logger,
            )
            records_processed += 1

    if records_processed == 0:
        fail(logger, f"No FASTA records were found in {args.sequence}")

    logger.info(f"Processed {records_processed} FASTA record(s)")


def parser() -> argparse.Namespace:
    """Create CLI argument parser for SpliceAI runner.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments

    Example
    -------
    >>> args = parser()
    >>> args.sequence
    PosixPath('input.fasta')
    """
    cli = argparse.ArgumentParser(
        description="Run SpliceAI on FASTA/FASTA.GZ chunks and write WIG files."
    )
    cli.add_argument(
        "-s",
        "--sequence",
        required=True,
        type=existing_file,
        metavar="FASTA/FASTA.GZ",
        help="Input FASTA or FASTA.GZ file",
    )
    cli.add_argument(
        "-r",
        "--round-to",
        default=4,
        type=positive_int,
        metavar="INT",
        help="Number of decimal digits to round predicted probabilities to",
    )
    cli.add_argument(
        "-p",
        "--min-prob",
        default=0.001,
        type=probability,
        metavar="FLOAT",
        help="Minimum probability to report in WIG output",
    )
    cli.add_argument(
        "-f",
        "--offset",
        required=False,
        type=non_negative_int,
        default=DEFAULT_FLANK_SIZE,
        metavar="INT",
        help="Symmetric flank size; must match the chunker's --flank-size value",
    )
    cli.add_argument(
        "-o",
        "--outdir",
        required=False,
        default=Path("."),
        type=Path,
        metavar="PATH",
        help="Output directory for generated WIG files",
    )
    cli.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    cli.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"spliceai-predict {__version__}",
    )
    return cli.parse_args()


def main() -> None:
    """Entry point for SpliceAI runner.

    Example
    -------
    >>> main()
    [INFO] Writing output to output/
    """
    args = parser()
    logger = Logger(verbose=args.verbose)

    try:
        run(args, logger)
    except KeyboardInterrupt:
        logger.warn("Execution interrupted")
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:
        logger.error(str(exc))
        if args.verbose:
            raise
        raise SystemExit(1)


if __name__ == "__main__":
    main()
