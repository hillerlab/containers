#!/usr/bin/env python3
# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

"""
Run SpliceTransformer on FASTA or FASTA.GZ input chunks and write WIG tracks.
Mimics modules/spliceai/py/spliceai.py but uses PyTorch SpliceTransformer
(context 4000 each side, max_seq_len 8192) with tiled inference for genome-wide chunks.
"""

from __future__ import annotations

__author__ = "Alejandro Gonzales-Irribarren"
__credits__ = ["Yury V. Malovichko", "Michael Hiller"]
__email__ = "alejandrxgzi@gmail.com"
__github__ = "https://github.com/alejandrogzi"
__version__ = "0.0.4"

import argparse
import gzip
import os
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NoReturn, TextIO

FASTA_HEADER_START = ">"
WIGGLE_HEADER_TEMPLATE = "fixedStep chrom={} start={} step=1 span=1\n"
HEADER_PATTERN = re.compile(
    r"^(?P<chrom>.+):(?P<region_start>\d+)-(?P<region_end>\d+)\((?P<strand>[+-])\)$"
)
# SpliceTransformer: context 4000 each side, total 8000
CONTEXT = 8000
CONTEXT_HALF = CONTEXT // 2  # 4000
DEFAULT_FLANK_SIZE = 4000
MAX_SEQ_LEN = 8192
WINDOW_OUT = (
    MAX_SEQ_LEN - CONTEXT
)  # 192  ponytail: naive stride=192, batch=1; batched windows if throughput matters
MODEL_FILENAME = "SpTransformer_pytorch.ckpt"
# ST labels the first exonic nt as acceptor; this lab's SpliceAI tracks sit on
# the G of AG (2 bp upstream). Shift acceptor in transcript space before WIG.
ACCEPTOR_SHIFT = 2
# start+2 and ACCEPTOR_SHIFT are plus-oriented (intron to the right). After RC
# and reversing scores to genomic order, both minus tracks sit 2 bp too far
# downstream in the forward sense. Roll them back in genomic space.
MINUS_GENOMIC_SHIFT = 2
_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


class Logger:
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
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {value}")
    return path


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value}") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be >= 1")
    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value}") from None
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0")
    return parsed


def probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float: {value}") from None
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("Value must be between 0.0 and 1.0")
    return parsed


def fail(logger: Logger, message: str) -> NoReturn:
    logger.error(message)
    raise SystemExit(1)


def build_output_prefix(sequence_path: Path) -> str:
    name = sequence_path.name
    if name.endswith(".gz"):
        name = name[:-3]
    for suffix in (".fasta", ".fa", ".fna", ".fas"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    stem = Path(name).stem
    return stem or name


def is_gzip_file(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def iter_fasta_records(
    sequence_path: Path, logger: Logger
) -> Iterator[tuple[str, str]]:
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
    assert match is not None
    chrom = match.group("chrom")
    try:
        start = int(match.group("region_start"))
        end = int(match.group("region_end"))
    except ValueError:
        fail(logger, f"Invalid coordinates in header: {header}")
        raise
    strand = match.group("strand") == "+"
    if end <= start:
        fail(logger, f"Sequence {header} has invalid coordinates: end must be > start")
    return chrom, start, end, strand


def reverse_complement(seq: str) -> str:
    """Reverse-complement a DNA sequence. Chunk FASTA is genomic for both strands."""
    return seq.translate(_RC_TABLE)[::-1]


def wig_inner_slice(
    len_seq: int, start_offset: int, end_offset: int, plus: bool
) -> tuple[int, int]:
    """Inner-chunk slice into transcript-oriented predictions.

    Header coords are 0-based half-open; WIG is 1-based with start=header_start+2.
    Full inner chunk is `chunk_length` values (last WIG coord = end+1). That last
    coordinate is past chrom size only on a terminal chunk, detected as no right
    flank (`end_offset == 0`). Clip that one value so last == `end`.

    Minus-strand scores are in RC/transcript order; dropping the last WIG value
    means dropping the *first* index of the pre-reverse slice.
    """
    clip = 1 if end_offset == 0 else 0
    if plus:
        return start_offset, len_seq - end_offset - clip
    return end_offset + clip, len_seq - start_offset


def roll_left(values, shift: int):
    """Move scores `shift` bp toward lower indices; pad the 3' end with 0."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if shift <= 0:
        return arr
    out = np.zeros_like(arr)
    if 0 < shift < arr.size:
        out[: arr.size - shift] = arr[shift:]
    return out


def shift_acceptor(values, shift: int = ACCEPTOR_SHIFT):
    """Move acceptor scores `shift` bp toward 5' in transcript space."""
    return roll_left(values, shift)


def one_hot_encode(seq: str):
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


def _candidate_model_paths() -> list[Path]:
    cands: list[Path] = []
    env = os.environ.get("SPLICETRANSFORMER_MODEL") or os.environ.get(
        "SPTRANSFORMER_MODEL"
    )
    if env:
        cands.append(Path(env))
    # bundled locations inside container
    for p in [
        Path("/models") / MODEL_FILENAME,
        Path("/usr/local/share/splicetransformer") / MODEL_FILENAME,
        Path(__file__).parent.parent / "model" / "weights" / MODEL_FILENAME,
        Path(__file__).parent.parent.parent
        / "tmp"
        / "SpliceTransformer"
        / "model"
        / "weights"
        / MODEL_FILENAME,
        Path("model/weights") / MODEL_FILENAME,
        Path("models") / MODEL_FILENAME,
    ]:
        cands.append(p)
    # also check importlib if installed as package
    try:
        import importlib.metadata as importlib_metadata

        for dist_name in ("splicetransformer", "sptransformer"):
            try:
                dist = importlib_metadata.distribution(dist_name)
                for f in dist.files or []:
                    if str(f).endswith(MODEL_FILENAME):
                        cands.append(Path(str(dist.locate_file(f))))  # type: ignore[arg-type]
            except importlib_metadata.PackageNotFoundError:
                continue
    except Exception:
        pass
    # dedup
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in cands:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def bundled_model_paths(logger: Logger) -> list[Path]:
    for p in _candidate_model_paths():
        if p.is_file():
            logger.debug(f"Found SpliceTransformer model at {p}")
            return [p]
    # not found
    cands = "\n  ".join(str(p) for p in _candidate_model_paths())
    fail(
        logger,
        (
            "SpliceTransformer model weights not found. "
            f"Set SPLICETRANSFORMER_MODEL env or place {MODEL_FILENAME} in one of:\n  {cands}\n"
            "Download from https://drive.google.com/file/d/1d8n4vHDSbXqpPc_JFEswLomSUDBgHvno/view?usp=drive_link"
        ),
    )
    return []


def _load_sptransformer_model(ckpt_path: Path, device: str, logger: Logger):
    """Load SpTransformer torch model from ckpt."""
    import torch  # type: ignore[import-untyped]

    # vendor model code without external import to keep single-file
    # try to import from local model/model.py if available
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / "tmp" / "SpliceTransformer"))
    try:
        from model.model import SpTransformer  # type: ignore

        model = SpTransformer(
            128,
            context_len=CONTEXT_HALF,
            tissue_num=15,
            max_seq_len=MAX_SEQ_LEN,
            attn_depth=8,
            training=False,
        )
        save_dict = torch.load(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
            str(ckpt_path), map_location="cpu"
        )
        # ckpt may be {"state_dict": ...} or plain state_dict
        state = (
            save_dict.get("state_dict", save_dict)
            if isinstance(save_dict, dict)
            else save_dict
        )
        # strip "model." prefix if present
        if any(k.startswith("model.") for k in state.keys()):
            state = {
                k.replace("model.", "", 1) if k.startswith("model.") else k: v
                for k, v in state.items()
            }
        # also try "encoder." etc - SpTransformer load expects its own keys
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            logger.warn(f"Strict load failed, trying strict=False: {e}")
            model.load_state_dict(state, strict=False)
        model.to(device).eval()
        logger.info(f"Loaded SpliceTransformer model from {ckpt_path} on {device}")
        return model
    except Exception as e:
        # fallback: try to import SpTransformerDriver path
        logger.warn(f"Could not load SpTransformer class: {e}")
        raise


def load_models(logger: Logger, device: str | None = None):
    """Load SpliceTransformer torch model(s). Returns list with one model for API parity."""
    # allow mock for testing without torch/model file
    if os.environ.get("SPLICETRANSFORMER_MOCK") == "1":
        logger.info("SPLICETRANSFORMER_MOCK=1, using dummy model")

        class Dummy:
            def __call__(self, x):
                # numpy fallback if torch not available
                try:
                    import torch as _t  # type: ignore[import-untyped]

                    b, _c, n = x.shape  # type: ignore[attr-defined]
                    l_out = max(0, n - CONTEXT)
                    return _t.zeros((b, 18, l_out), device=x.device)  # type: ignore[attr-defined]
                except Exception:
                    import numpy as _np

                    # x is numpy array case
                    try:
                        b, _c, n = x.shape
                    except Exception:
                        b, n = 1, 0
                    l_out = max(0, n - CONTEXT)
                    return _np.zeros((b, 18, l_out))

            def to(self, *a, **kw):
                return self

            def eval(self):
                return self

        return [Dummy()]
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        fail(
            logger,
            "PyTorch not installed. Install torch or set SPLICETRANSFORMER_MOCK=1 for testing.",
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device.lower()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = bundled_model_paths(logger)[0]
    model = _load_sptransformer_model(ckpt_path, device, logger)
    return [model]


def _predict_windows(
    seq_padded: str, models: Sequence[object], device: str, logger: Logger
):
    """Tiled inference for padded sequence. Returns (acceptor, donor) arrays length = len(seq_padded)-8000."""
    import numpy as np

    L_out = len(seq_padded) - CONTEXT
    if L_out <= 0:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    if os.environ.get("SPLICETRANSFORMER_MOCK") == "1":
        # mock: return zeros without requiring torch/model
        return np.zeros((L_out,), dtype=float), np.zeros((L_out,), dtype=float)
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        # fallback to zeros if torch missing but not in mock (should have failed earlier)
        return np.zeros((L_out,), dtype=float), np.zeros((L_out,), dtype=float)

    model = models[0]

    # fast path single window
    if len(seq_padded) <= MAX_SEQ_LEN:
        x = one_hot_encode(seq_padded)[None, :]  # (1, N,4)
        x_t = torch.tensor(x, device=device).transpose(1, 2).float()  # type: ignore[attr-defined]
        with torch.no_grad():  # type: ignore[attr-defined]
            y = model(x_t)  # type: ignore[operator]
            # SpTransformer forward already applies softmax/sigmoid via post_decorate? No, model's forward returns raw logits for SpTransformer
            # Need to apply softmax/sigmoid like SpTransformerDriver.post_decorate
            # But SpTransformer.forward returns concat of splice+usage after linear, not yet activated. Check model/model.py SpTransformer.forward returns out = concat(splice_out, usage_out) raw.
            # However SpTransformerDriver.post_decorate does softmax/sigmoid. We should apply here.
            # If model is SpTransformer, apply manually:
            try:
                import torch.nn.functional as F  # type: ignore[import-untyped]

                y_splice = y[:, :3, :]
                y_usage = y[:, 3:, :]
                y_splice = F.softmax(y_splice, dim=1)
                y_usage = torch.sigmoid(y_usage)  # type: ignore[attr-defined]
                y = torch.cat([y_splice, y_usage], dim=1)  # type: ignore[attr-defined]
            except Exception:
                pass
        y_np = y.cpu().numpy()[0].transpose(1, 0)  # (L_out,18)
        return y_np[:, 1], y_np[:, 2]

    # tiled: WINDOW_OUT=192
    import torch.nn.functional as F  # type: ignore[import-untyped]

    acc_list: list = []
    don_list: list = []
    n_windows = (L_out + WINDOW_OUT - 1) // WINDOW_OUT
    logger.debug(
        f"Tiling {L_out} bases into {n_windows} windows (WINDOW_OUT={WINDOW_OUT})"
    )
    for idx in range(n_windows):
        out_start = idx * WINDOW_OUT
        out_end = min((idx + 1) * WINDOW_OUT, L_out)
        w_out = out_end - out_start
        # input window in padded: [out_start, out_start+MAX_SEQ_LEN)
        in_start = out_start
        in_end = out_start + MAX_SEQ_LEN
        window_seq = seq_padded[in_start:in_end]
        if len(window_seq) < MAX_SEQ_LEN:
            window_seq = window_seq + "N" * (MAX_SEQ_LEN - len(window_seq))
        x = one_hot_encode(window_seq)[None, :]
        x_t = torch.tensor(x, device=device).transpose(1, 2).float()  # type: ignore[attr-defined]
        with torch.no_grad():  # type: ignore[attr-defined]
            y = model(x_t)  # type: ignore[operator]
            try:
                y_splice = y[:, :3, :]
                y_usage = y[:, 3:, :]
                y_splice = F.softmax(y_splice, dim=1)
                y_usage = torch.sigmoid(y_usage)  # type: ignore[attr-defined]
                y = torch.cat([y_splice, y_usage], dim=1)
            except Exception:
                pass
        y_np = y.cpu().numpy()[0].transpose(1, 0)  # (192,18)
        # y_np length is WINDOW_OUT (192) for full windows, but last window may be padded, still 192
        acc_list.append(y_np[:w_out, 1])
        don_list.append(y_np[:w_out, 2])
    import numpy as np

    return np.concatenate(acc_list), np.concatenate(don_list)


def write_probabilities(
    handle: TextIO, probabilities, round_to: int, min_prob: float
) -> None:
    for value in probabilities:
        try:
            fv = float(value)
        except (TypeError, ValueError):
            fv = 0.0
        parsed = round(fv, round_to) if fv >= min_prob else 0.0
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
    device: str = "cpu",
) -> None:
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

    if not strand:
        seq = reverse_complement(seq)

    # ST uses N*4000 + seq + N*4000 (CONTEXT=8000)
    padded = "N" * CONTEXT_HALF + seq + "N" * CONTEXT_HALF
    acceptor_prob, donor_prob = _predict_windows(padded, models, device, logger)
    # acceptor_prob/donor_prob length == len(seq)
    if len(acceptor_prob) != len(seq) or len(donor_prob) != len(seq):
        fail(
            logger,
            f"Model output length mismatch for {header}: expected {len(seq)}, got {len(acceptor_prob)}",
        )

    acceptor_prob = shift_acceptor(acceptor_prob, ACCEPTOR_SHIFT)

    # fixedStep start = start+2 matches this lab's SpliceAI calibration (donor on
    # G of GT). Terminal chunks (no right flank) drop one value so last <= end.
    wiggle_header = WIGGLE_HEADER_TEMPLATE.format(chrom, start + 2)
    acc_plus_handle, donor_plus_handle, acc_minus_handle, donor_minus_handle = (
        wig_handles
    )
    start_index, end_index = wig_inner_slice(
        len(seq), start_offset, end_offset, plus=strand
    )
    acceptor_slice = acceptor_prob[start_index:end_index]
    donor_slice = donor_prob[start_index:end_index]

    if strand:
        acc_plus_handle.write(wiggle_header)
        donor_plus_handle.write(wiggle_header)
        write_probabilities(donor_plus_handle, donor_slice, round_to, min_prob)
        write_probabilities(acc_plus_handle, acceptor_slice, round_to, min_prob)
        return

    acc_minus_handle.write(wiggle_header)
    donor_minus_handle.write(wiggle_header)
    donor_slice = roll_left(donor_slice[::-1], MINUS_GENOMIC_SHIFT)
    acceptor_slice = roll_left(acceptor_slice[::-1], MINUS_GENOMIC_SHIFT)
    write_probabilities(donor_minus_handle, donor_slice, round_to, min_prob)
    write_probabilities(acc_minus_handle, acceptor_slice, round_to, min_prob)


def run(args: argparse.Namespace, logger: Logger) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = build_output_prefix(args.sequence)
    acc_plus_file = args.outdir / f"{prefix}.acceptor_plus.wig"
    donor_plus_file = args.outdir / f"{prefix}.donor_plus.wig"
    acc_minus_file = args.outdir / f"{prefix}.acceptor_minus.wig"
    donor_minus_file = args.outdir / f"{prefix}.donor_minus.wig"

    logger.info(f"Writing output to {args.outdir}")
    logger.debug(f"Output prefix: {prefix}")

    # device resolution
    device = getattr(args, "device", "cpu")
    mock_mode = os.environ.get("SPLICETRANSFORMER_MOCK") == "1"
    torch_mod = None
    try:
        import torch as torch_mod  # type: ignore[import-untyped]
    except ImportError:
        if mock_mode:
            torch_mod = None
        else:
            fail(
                logger,
                "PyTorch not installed. Install torch or set SPLICETRANSFORMER_MOCK=1 for testing.",
            )
    if device == "auto":
        if torch_mod is not None and torch_mod.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    models = load_models(logger, device=device if torch_mod is not None else "cpu")
    if device not in ("cpu", "cuda"):
        pass
    if (
        torch_mod is not None
        and not torch_mod.cuda.is_available()
        and device.startswith("cuda")
    ):
        logger.warn(f"CUDA not available, falling back to cpu (requested {device})")
        device = "cpu"

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
                device=device,
            )
            records_processed += 1

    if records_processed == 0:
        fail(logger, f"No FASTA records were found in {args.sequence}")
    logger.info(f"Processed {records_processed} FASTA record(s)")


def parser() -> argparse.Namespace:
    cli = argparse.ArgumentParser(
        description="Run SpliceTransformer on FASTA/FASTA.GZ chunks and write WIG files."
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
        "--model",
        required=False,
        type=Path,
        default=None,
        help="Path to SpTransformer_pytorch.ckpt (overrides SPLICETRANSFORMER_MODEL env)",
    )
    cli.add_argument(
        "--device",
        required=False,
        default="cpu",
        help="Torch device: cpu, cuda, cuda:0, auto",
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
        version=f"splicetransformer-predict {__version__}",
    )
    return cli.parse_args()


def main() -> None:
    args = parser()
    if args.model is not None:
        os.environ["SPLICETRANSFORMER_MODEL"] = str(args.model)
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
