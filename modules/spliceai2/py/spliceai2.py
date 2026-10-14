#!/usr/bin/env python3

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

"""Run SpliceAI2 on spliceai-chunk FASTA records and genomic BED loci.

The WIG track starts at 1-based ``chunk_start + 2``, matching spliceai.py.
For transcripts, BED12 scores are ``round(1000 * sigmoid(sum of the selected
out_tx junction logits))``. Each BED locus is decoded once, with up to 65,536
bases of context per side by default. SpliceAI2 sees ±65,536 bp of context;
introns longer than about 65 kb are outside its view, may come out split into
~70-kb pieces, and are not recovered reliably. SpliceAI2 decodes intron chains
and does not predict TSS/TES; the locus gives the terminal exon ends.
"""

from __future__ import annotations

__author__ = "Alejandro Gonzales-Irribarren"
__credits__ = ["Yury V. Malovichko", "Michael Hiller"]
__email__ = "alejandrxgzi@gmail.com"
__github__ = "https://github.com/alejandrogzi"
__version__ = "0.0.5"

import argparse
import gzip
import hashlib
import math
import os
import re
import sys
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import TextIO

HEADER = re.compile(r"^(?P<chrom>.+):(?P<start>\d+)-(?P<end>\d+)\((?P<strand>[+-])\)$")
WIG_HEADER = "fixedStep chrom={} start={} step=1 span=1\n"
CONTEXT = 131_072
FLANK = CONTEXT // 2
ALIGNMENT = 16
MINUS_GENOMIC_SHIFT = 2
MODEL_FOLDERS = ("spliceai2_multispecies_13M_01", "spliceai2_multispecies_13M_02")
CHECKPOINT = "model.ckpt"
HF_REPO = "illumina-ai/SpliceAI2"
SPECIES = (
    "GRCh38", "NHGRI_mPanTro3-v2.0", "Panubis1.0", "T2T-MFA8v1.1", "Mmul_10",
    "ARS-UCD2.0", "ARS-UI_Ramb_v3.0", "Sscrofa11.1", "GRCm38", "GRCr8",
)


class Logger:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def info(self, message: str) -> None:
        print(f"[INFO] {message}", file=sys.stderr)

    def debug(self, message: str) -> None:
        if self.verbose:
            print(f"[DEBUG] {message}", file=sys.stderr)

    def warning(self, message: str) -> None:
        print(f"[WARNING] {message}", file=sys.stderr)

    def error(self, message: str) -> None:
        print(f"[ERROR] {message}", file=sys.stderr)


def default_models_dir() -> Path:
    return Path(os.environ.get("SPLICEAI2_MODELS") or "/models")


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    with path.open("rb") as raw:
        compressed = raw.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as handle:
        header: str | None = None
        bases: list[str] = []
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(bases)
                header, bases = line[1:], []
            elif header is None:
                raise ValueError(f"FASTA sequence data before the first header at line {line_no}")
            else:
                bases.append(line)
        if header is not None:
            yield header, "".join(bases)


def parse_header(header: str) -> tuple[str, int, int, bool]:
    match = HEADER.fullmatch(header)
    if match is None:
        raise ValueError(f"Expected chunk header CHROM:START-END(+/-), got {header!r}")
    chrom, start, end, strand = (
        match.group("chrom"), int(match.group("start")), int(match.group("end")),
        match.group("strand") == "+",
    )
    if end <= start:
        raise ValueError(f"Invalid half-open interval in FASTA header: {header}")
    return chrom, start, end, strand


def record_layout(header: str, sequence: str, offset: int):
    chrom, start, end, plus = parse_header(header)
    sequence = sequence.upper()
    if not sequence:
        raise ValueError(f"{header}: sequence must not be empty")
    length = end - start
    left = min(offset, start)
    right = len(sequence) - left - length
    if right < 0 or right > offset:
        raise ValueError(
            f"{header}: sequence length {len(sequence)} is inconsistent with --offset {offset}"
        )
    sense = sequence if plus else sequence[::-1]
    core_start = left if plus else right
    return chrom, start, end, plus, sense, core_start, right


def windows(length: int, size: int):
    for start in range(0, length, size):
        actual = min(size, length - start)
        yield start, actual, (actual + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def window_batches(items, batch_size: int):
    batch = []
    for item in items:
        if batch and (len(batch) == batch_size or batch[0][2] != item[2]):
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch


def window_sequence(
    sequence: str, core_start: int, start: int, actual: int, model_length: int
) -> str:
    """Use 65,536 nt of context per side; N fills context absent at sequence ends."""
    left_end = core_start + start
    right_start = left_end + actual
    prefix_length = model_length - actual
    prefix_start = max(0, left_end - prefix_length)
    prefix = sequence[prefix_start:left_end]
    prefix = "N" * (prefix_length - len(prefix)) + prefix
    left = sequence[max(0, prefix_start - FLANK):prefix_start]
    right = sequence[right_start:right_start + FLANK]
    core = prefix + sequence[left_end:right_start]
    return "N" * (FLANK - len(left)) + left + core + right + "N" * (FLANK - len(right))


def locus_sequence(chromosome, start: int, end: int, plus: bool, offset: int):
    """Return model input and the decoded-mask offset of a half-open locus."""
    chrom_length = len(chromosome)
    left_start, right_end = max(0, start - offset), min(chrom_length, end + offset)
    left = str(chromosome[left_start:start]).upper()
    core = str(chromosome[start:end]).upper()
    right = str(chromosome[end:right_end]).upper()
    left = "N" * (offset - len(left)) + left
    right += "N" * (offset - len(right))
    flank = max(offset, FLANK)
    left = "N" * (flank - offset) + left
    right += "N" * (flank - offset)
    aligned = left + core + right
    padding = "N" * (-len(aligned) % ALIGNMENT)
    sequence = aligned + padding if plus else (padding + aligned)[::-1]
    return sequence, flank - FLANK


def open_genome(path: Path):
    from pyfaidx import Fasta

    fai = Path(f"{path}.fai")
    missing_index = not fai.is_file()
    if missing_index and not os.access(path.parent, os.W_OK):
        raise ValueError(f"FASTA index is missing and cannot be created; run `samtools faidx {path}`")
    try:
        genome = Fasta(str(path), as_raw=True, sequence_always_upper=True)
    except Exception as error:
        if missing_index:
            raise ValueError(
                f"FASTA index is missing and cannot be created; run `samtools faidx {path}`"
            ) from error
        raise
    if not fai.is_file():
        genome.close()
        raise ValueError(f"FASTA index was not created; run `samtools faidx {path}`")
    return genome


def read_loci(path: Path, genome):
    loci = []
    seen = set()
    chroms = set(genome.keys())
    with path.open(encoding="utf-8") as bed:
        for line_no, line in enumerate(bed, 1):
            raw = line.rstrip("\r\n")
            if not raw or raw.startswith("#"):
                continue
            fields = raw.split("\t")
            if len(fields) < 6:
                raise ValueError(f"{path}: line {line_no}: expected BED6 or more columns")
            chrom, start_text, end_text, name, score, strand = fields[:6]
            if chrom not in chroms:
                raise ValueError(f"{path}: line {line_no}: unknown chromosome {chrom!r}")
            try:
                start, end = int(start_text), int(end_text)
            except ValueError:
                raise ValueError(f"{path}: line {line_no}: start and end must be integers") from None
            if start < 0 or start >= end:
                raise ValueError(f"{path}: line {line_no}: invalid half-open interval {start}-{end}")
            if end > len(genome[chrom]):
                raise ValueError(f"{path}: line {line_no}: end {end} exceeds {chrom} length {len(genome[chrom])}")
            if strand not in ("+", "-"):
                raise ValueError(f"{path}: line {line_no}: strand must be + or -")
            if name in seen:
                raise ValueError(f"{path}: line {line_no}: duplicate locus name {name!r}")
            seen.add(name)
            loci.append((chrom, start, end, name, strand == "+", raw))
    if not loci:
        raise ValueError(f"No BED loci found in {path}")
    return loci


def model_paths(models_dir: Path) -> list[Path]:
    paths = [models_dir / folder / CHECKPOINT for folder in MODEL_FOLDERS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        absent = ", ".join(str(path) for path in missing)
        print(
            f"[ERROR] Missing SpliceAI2 checkpoint(s): {absent}. Mount both model folders with "
            "docker run -v <host dir>:/models … or apptainer run --bind <host dir>:/models …, "
            "or fetch them with spliceai2 fetch-models --out DIR.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return paths


def load_models(models_dir: Path, device: str, no_compile: bool, logger: Logger):
    paths = model_paths(models_dir)
    import torch
    from spliceai2.model import SpliceAI2

    models = []
    for path in paths:
        model = SpliceAI2.load_from_checkpoint(str(path), map_location=device).eval().to(device)
        if not no_compile:
            model = torch.compile(model)
        models.append(model)
    logger.info(f"Loaded {len(models)} SpliceAI2 members on {device}")
    return models


class Predictor:
    def __init__(self, models, device: str, precision: str) -> None:
        import torch

        self.models = models
        self.device = device
        self.precision = precision
        self.torch = torch

    def encode(self, sequences: Sequence[str], species: str, plus: bool = True):
        import numpy as np
        from spliceai2.dataset import _one_hot_encode

        forward = sequences if plus else [sequence[::-1] for sequence in sequences]
        encoded = np.stack([_one_hot_encode(sequence, species) for sequence in forward])
        if not plus:
            encoded[:, :4] = encoded[:, :4][:, ::-1, ::-1]
        return self.torch.tensor(encoded, dtype=self.torch.float32, device=self.device)

    def forward(self, x):
        torch = self.torch
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device == "cuda" and self.precision == "fp16-mixed"
            else nullcontext()
        )
        with torch.no_grad(), amp:
            features = [model(x) for model in self.models]
            out_ss = torch.stack(
                [model.forward_1d(feature) for model, feature in zip(self.models, features)]
            )
        return out_ss, features

    def splice_sites(self, sequences: Sequence[str], species: str, plus: bool = True):
        out_ss, _ = self.forward(self.encode(sequences, species, plus))
        return self.torch.sigmoid(out_ss).mean(dim=0).float().cpu().numpy()

    def transcripts(
        self, sequence: str, species: str, k: int, plus: bool,
        mask_start: int, locus_length: int, max_candidates: int,
        min_candidate_prob: float, locus_name: str, logger: Logger,
    ):
        from spliceai2.utils import decode_topk_tx

        torch = self.torch
        member_ss, features = self.forward(self.encode([sequence], species, plus))
        out_ss = member_ss.mean(dim=0)
        out_1d_len = out_ss.shape[2]
        mask_end = mask_start + locus_length
        if mask_start < 0 or locus_length < 1 or mask_end > out_1d_len:
            raise ValueError(
                f"Decoded output length {out_1d_len} does not cover locus mask "
                f"[{mask_start}, {mask_end})"
            )
        scores = member_ss.sigmoid().mean(dim=0).amax(dim=1)[:, mask_start:mask_end]
        positions = torch.nonzero(scores[0] >= min_candidate_prob).flatten()
        candidate_count = positions.numel()
        if not candidate_count:
            positions = scores[0].argmax().reshape(1)
        elif candidate_count > max_candidates:
            logger.warning(
                f"{locus_name}: {candidate_count} positions passed --min-candidate-prob; "
                f"keeping the top {max_candidates}"
            )
            positions = positions[scores[0, positions].topk(max_candidates).indices]
        idxs = positions.sort().values.add(mask_start).unsqueeze(0)
        out_2d_len = idxs.shape[1]
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device == "cuda" and self.precision == "fp16-mixed"
            else nullcontext()
        )
        with torch.no_grad(), amp:
            out_jxn = torch.stack(
                [model.forward_2d(feature, idxs) for model, feature in zip(self.models, features)]
            ).mean(dim=0)
        out_tx = (
            out_ss[:, 0].gather(1, idxs).unsqueeze(2)
            + out_jxn
            + out_ss[:, 1].gather(1, idxs).unsqueeze(1)
        )
        masks = decode_topk_tx(
            out_tx[0].cpu().numpy().copy(), idxs[0].cpu().numpy(),
            out_1d_len, out_2d_len, k=k,
        )
        return masks, out_tx[0].cpu().numpy(), idxs[0].cpu().numpy()


def resolve_runtime(device: str, precision: str | None, logger: Logger):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable")
    if precision == "fp16-mixed" and device == "cpu":
        raise ValueError("--precision fp16-mixed requires CUDA; use --precision fp32 on CPU")
    precision = precision or ("fp16-mixed" if device == "cuda" else "fp32")
    logger.info(f"Using device {device}, precision {precision}")
    return device, precision


def roll_left(values, shift: int):
    values = list(values)
    if shift <= 0:
        return values
    return values[shift:] + [0.0] * min(shift, len(values))


def genomic_wig_values(values, plus: bool, terminal: bool):
    """Convert transcript-order values to genomic order using spliceai.py's offset."""
    values = list(values)
    if plus:
        return values[:-1] if terminal else values
    if terminal:
        values = values[1:]
    return roll_left(values[::-1], MINUS_GENOMIC_SHIFT)


def write_probabilities(handle: TextIO, values, round_to: int, min_prob: float) -> None:
    for value in values:
        value = float(value)
        handle.write(f"{round(value, round_to) if value >= min_prob else 0.0}\n")


def predict_record(header: str, sequence: str, args, predictor: Predictor, handles, logger: Logger):
    chrom, start, end, plus, sense, core_start, right_flank = record_layout(
        header, sequence, args.offset
    )
    length = end - start
    values = [[], []]
    tiles = list(windows(length, args.window))
    for batch in window_batches(tiles, args.batch_size):
        sequences = [window_sequence(sense, core_start, at, actual, model_len)
                     for at, actual, model_len in batch]
        probs = predictor.splice_sites(sequences, args.species, plus)
        for index, (_, actual, model_len) in enumerate(batch):
            prefix = model_len - actual
            values[0].extend(probs[index, 0, prefix:prefix + actual].tolist())
            values[1].extend(probs[index, 1, prefix:prefix + actual].tolist())
    if any(len(channel) != length for channel in values):
        raise ValueError(f"{header}: predicted values do not cover the {length}-nt inner interval")

    terminal = right_flank == 0
    donor = genomic_wig_values(values[0], plus, terminal)
    acceptor = genomic_wig_values(values[1], plus, terminal)
    wig_header = WIG_HEADER.format(chrom, start + 2)
    acceptor_handle, donor_handle = handles[0:2] if plus else handles[2:4]
    if donor or acceptor:
        acceptor_handle.write(wig_header)
        donor_handle.write(wig_header)
        write_probabilities(donor_handle, donor, args.round_to, args.min_prob)
        write_probabilities(acceptor_handle, acceptor, args.round_to, args.min_prob)
    logger.debug(f"{header}: wrote {len(donor)} genomic predictions")


def run_predict(args, logger: Logger) -> None:
    model_paths(args.models)
    device, precision = resolve_runtime(args.device, args.precision, logger)
    models = load_models(args.models, device, args.no_compile, logger)
    predictor = Predictor(models, device, precision)
    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.sequence.name.removesuffix(".gz")
    for suffix in (".fasta", ".fa", ".fna", ".fas"):
        if prefix.endswith(suffix):
            prefix = prefix[:-len(suffix)]
            break
    paths = [args.outdir / f"{prefix}.{kind}_{strand}.wig"
             for strand in ("plus", "minus") for kind in ("acceptor", "donor")]
    count = 0
    with paths[0].open("w") as acc_plus, paths[1].open("w") as donor_plus, \
         paths[2].open("w") as acc_minus, paths[3].open("w") as donor_minus:
        for header, sequence in iter_fasta(args.sequence):
            predict_record(
                header, sequence, args, predictor,
                (acc_plus, donor_plus, acc_minus, donor_minus), logger,
            )
            count += 1
    if count == 0:
        raise ValueError(f"No FASTA records found in {args.sequence}")
    logger.info(f"Processed {count} FASTA record(s)")


def exon_blocks(mask) -> list[tuple[int, int]]:
    blocks = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            blocks.append((start, index))
            start = None
    if start is not None:
        blocks.append((start, len(mask)))
    return blocks


def bed12_row(chrom: str, start: int, end: int, locus_name: str, plus: bool,
              mask, rank: int, score: int) -> str | None:
    mask = list(mask)
    if mask:
        mask[0] = mask[-1] = 1
    blocks = exon_blocks(mask)
    if not blocks:
        return None
    if plus:
        genomic = [(start + a, start + b) for a, b in blocks]
    else:
        genomic = sorted((end - b, end - a) for a, b in blocks)
    chrom_start, chrom_end = genomic[0][0], genomic[-1][1]
    sizes = [b - a for a, b in genomic]
    starts = [a - chrom_start for a, _ in genomic]
    if any(size <= 0 for size in sizes) or any(a + size > chrom_end - chrom_start
                                                for a, size in zip(starts, sizes)):
        raise ValueError("Invalid BED12 blocks generated from transcript mask")
    name = f"{locus_name}.t{rank}"
    return "\t".join(map(str, (
        chrom, chrom_start, chrom_end, name, score, "+" if plus else "-",
        chrom_start, chrom_end, "255,0,0", len(sizes),
        ",".join(map(str, sizes)), ",".join(map(str, starts)),
    )))


def transcript_score(mask, out_tx, idxs, prefix: int = 0) -> int:
    """BED score: sigmoid of the selected out_tx junction-logit sum, scaled to 0..1000."""
    total = 0.0
    start = None
    for index, value in enumerate(mask):
        if not value and start is None:
            start = index
        elif value and start is not None:
            bounds = (start + prefix, index - 1 + prefix)
            start = None
            left = next((i for i, pos in enumerate(idxs) if pos == bounds[0]), None)
            right = next((i for i, pos in enumerate(idxs) if pos == bounds[1]), None)
            if left is not None and right is not None:
                total += float(out_tx[left, right])
    if start is not None:
        left = next((i for i, pos in enumerate(idxs) if pos == start + prefix), None)
        right = next((i for i, pos in enumerate(idxs) if pos == len(mask) - 1 + prefix), None)
        if left is not None and right is not None:
            total += float(out_tx[left, right])
    value = 1 / (1 + math.exp(-total)) if total >= 0 else math.exp(total) / (1 + math.exp(total))
    return round(1000 * value)


def run_transcripts(args, logger: Logger) -> None:
    genome = open_genome(args.genome)
    try:
        loci = read_loci(args.loci, genome)
        skipped = [locus for locus in loci if locus[2] - locus[1] > args.max_length]
        run = [locus for locus in loci if locus[2] - locus[1] <= args.max_length]
        prefix = args.loci.name.removesuffix(".bed")
        args.outdir.mkdir(parents=True, exist_ok=True)
        output = args.outdir / f"{prefix}.transcripts.bed"
        skipped_path = args.outdir / f"{prefix}.skipped.bed"
        with skipped_path.open("w", encoding="utf-8") as skipped_bed:
            for locus in skipped:
                skipped_bed.write(locus[5] + "\tmax_length\n")
        if skipped:
            logger.warning(f"Skipped {len(skipped)} locus/loci longer than --max-length; listed in {skipped_path}")

        chrom_order = {chrom: index for index, chrom in enumerate(genome.keys())}
        run.sort(key=lambda locus: (chrom_order[locus[0]], locus[1], locus[2]))
        out_of_memory = 0
        if run:
            model_paths(args.models)
            device, precision = resolve_runtime(args.device, args.precision, logger)
            models = load_models(args.models, device, args.no_compile, logger)
            predictor = Predictor(models, device, precision)
        with output.open("w", encoding="utf-8") as bed:
            for chrom, start, end, name, plus, raw in run:
                length = end - start
                try:
                    sequence, mask_start = locus_sequence(
                        genome[chrom], start, end, plus, args.offset
                    )
                    if device == "cuda":
                        predictor.torch.cuda.reset_peak_memory_stats(device)
                    masks, out_tx, idxs = predictor.transcripts(
                        sequence, args.species, args.k, plus, mask_start, length,
                        args.max_candidates, args.min_candidate_prob, name, logger,
                    )
                except (MemoryError, RuntimeError) as error:
                    message = str(error).lower()
                    if not (
                        isinstance(error, (MemoryError, predictor.torch.OutOfMemoryError))
                        or "out of memory" in message
                        or "can't allocate memory" in message
                    ):
                        raise
                    out_of_memory += 1
                    with skipped_path.open("a", encoding="utf-8") as skipped_bed:
                        skipped_bed.write(raw + "\tout_of_memory\n")
                    logger.warning(f"Out of memory at locus {name!r} ({length} nt); skipped")
                    if device == "cuda":
                        predictor.torch.cuda.empty_cache()
                    continue
                if device == "cuda":
                    peak_mib = predictor.torch.cuda.max_memory_allocated(device) / 1024**2
                    logger.info(f"{name}: locus length {length}: peak GPU memory {peak_mib:.1f} MiB")
                seen = set()
                for mask in masks:
                    core_mask = mask[mask_start:mask_start + length]
                    if len(core_mask) != length:
                        raise ValueError(f"{name}: decoded mask does not cover its {length}-nt locus")
                    bed_mask = list(core_mask)
                    bed_mask[0] = bed_mask[-1] = 1
                    mask_key = tuple(int(value) for value in bed_mask)
                    if mask_key in seen:
                        continue
                    seen.add(mask_key)
                    rank = len(seen)
                    row = bed12_row(
                        chrom, start, end, name, plus, core_mask, rank,
                        transcript_score(core_mask, out_tx, idxs, mask_start),
                    )
                    if row is not None:
                        bed.write(row + "\n")
    finally:
        genome.close()
    logger.info(
        f"Processed {len(loci)} locus/loci; ran {len(run) - out_of_memory}, "
        f"skipped {len(skipped) + out_of_memory} "
        f"(max_length={len(skipped)}, out_of_memory={out_of_memory}); "
        f"wrote {output} and {skipped_path}"
    )


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_models(out: Path, revision: str | None) -> None:
    from huggingface_hub import snapshot_download

    out.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": HF_REPO,
        "local_dir": str(out),
        "allow_patterns": [f"{folder}/{CHECKPOINT}" for folder in MODEL_FOLDERS],
    }
    if revision:
        kwargs["revision"] = revision
    try:
        snapshot_download(**kwargs)
    except Exception as error:
        response = getattr(error, "response", None)
        if type(error).__name__ == "GatedRepoError" or getattr(response, "status_code", None) in (401, 403):
            message = "SpliceAI2 access is gated: accept the terms on huggingface.co with your account, then log in with `hf auth login` or HF_TOKEN."
        else:
            message = "Hugging Face model download failed; check your login and network access."
        print(f"[ERROR] {message}", file=sys.stderr)
        raise SystemExit(2) from None
    paths = [out / folder / CHECKPOINT for folder in MODEL_FOLDERS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        print(f"[ERROR] Download did not include required checkpoint(s): {', '.join(map(str, missing))}",
              file=sys.stderr)
        raise SystemExit(2)
    for path in paths:
        print(f"{sha256_file(path)}  {path}")


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from None
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a probability between 0 and 1") from None
    if not 0 <= parsed <= 1 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a probability between 0 and 1")
    return parsed


def add_common(parser: argparse.ArgumentParser, transcripts: bool = False) -> None:
    if transcripts:
        parser.add_argument("--genome", required=True, type=Path, metavar="GENOME.fa")
        parser.add_argument("--loci", required=True, type=Path, metavar="LOCI.bed")
    else:
        parser.add_argument("-s", "--sequence", required=True, type=Path, metavar="FASTA/FASTA.GZ")
    parser.add_argument("-o", "--outdir", type=Path, default=Path("."), metavar="PATH")
    parser.add_argument("--models", type=Path, default=default_models_dir(), metavar="DIR",
                        help="Both gated model folders (default: $SPLICEAI2_MODELS or /models; image has no weights)")
    parser.add_argument("--species", choices=SPECIES, default="GRCh38",
                        help="Encoder species channel (default: GRCh38)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile")
    if not transcripts:
        parser.add_argument("--batch-size", type=positive_int, default=1,
                            help="Equal-length predict windows per batch")
    parser.add_argument("--precision", choices=("fp16-mixed", "fp32"), default=None,
                        help="Default: fp16-mixed on CUDA, fp32 on CPU")
    parser.add_argument("-f", "--offset", type=non_negative_int,
                        default=65_536 if transcripts else 50_000,
                        help=("Genomic context per side for transcripts" if transcripts else
                              "Chunker --flank-size used to make the FASTA records"))
    parser.add_argument("--verbose", action="store_true")


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description=("SpliceAI2: weights are gated and never included in the image; mount them or use fetch-models. "
                     "SpliceAI2 sees ±65,536 bp of context; introns longer than about 65 kb are outside its view, "
                     "may come out split into ~70-kb pieces, and are not recovered reliably.")
    )
    cli.add_argument("-v", "--version", action="version", version=f"spliceai2 {__version__}")
    commands = cli.add_subparsers(dest="command", required=True)
    convention = (
        "SpliceAI2 adds no end padding; N and end padding follow the benchmark "
        "convention (A on +, T on −). SpliceAI2 sees ±65,536 bp of context; introns longer than about 65 kb "
        "are outside its view, may come out split into ~70-kb pieces, and are not recovered reliably."
    )
    predict = commands.add_parser(
        "predict", help="Write donor/acceptor WIG tracks", description=convention
    )
    add_common(predict)
    predict.add_argument("-w", "--window", type=positive_int, default=65_536,
                         help="Inner window size; must be a multiple of 16 (default: 65536)")
    predict.add_argument("--round-to", type=non_negative_int, default=4)
    predict.add_argument("--min-prob", type=probability, default=0.001)
    transcripts = commands.add_parser(
        "transcripts", help="Decode top-k BED12 transcripts per BED locus",
        description=("Decode each locus once. SpliceAI2 predicts intron chains, not TSS/TES; "
                     "locus ends define terminal exons. SpliceAI2 sees ±65,536 bp of context; introns longer "
                     "than about 65 kb are outside its view, may come out split into ~70-kb pieces, and are not "
                     "recovered reliably.")
    )
    add_common(transcripts, transcripts=True)
    transcripts.add_argument("-k", type=positive_int, default=3, help="Transcripts per locus (default: 3)")
    transcripts.add_argument("--max-candidates", type=positive_int, default=2048,
                             help="Maximum splice-site candidates per locus (default: 2048)")
    transcripts.add_argument("--min-candidate-prob", type=probability, default=0.01,
                             help="Mean-member splice-site probability threshold (default: 0.01)")
    transcripts.add_argument("--max-length", type=positive_int, default=1_048_576,
                             help="Skip loci longer than this many bases")
    fetch = commands.add_parser(
        "fetch-models",
        help="Download gated weights using your Hugging Face login",
        description="Download gated model files to DIR. Weights are never included in the image.",
    )
    fetch.add_argument("--out", type=Path, required=True, metavar="DIR")
    fetch.add_argument("--revision", metavar="REV")
    return cli


def main() -> None:
    args = parser().parse_args()
    logger = Logger(args.verbose if hasattr(args, "verbose") else False)
    if args.command == "fetch-models":
        fetch_models(args.out, args.revision)
        return
    if args.command == "predict":
        if not args.sequence.is_file():
            logger.error(f"Input FASTA does not exist: {args.sequence}")
            raise SystemExit(2)
        if args.window % ALIGNMENT:
            logger.error(f"--window must be a multiple of {ALIGNMENT}")
            raise SystemExit(2)
    else:
        for option, path in (("--genome", args.genome), ("--loci", args.loci)):
            if not path.is_file():
                logger.error(f"{option} does not exist: {path}")
                raise SystemExit(2)
    try:
        if args.command == "predict":
            run_predict(args, logger)
        else:
            run_transcripts(args, logger)
    except SystemExit:
        raise
    except Exception as error:
        logger.error(str(error))
        if args.verbose:
            raise
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
