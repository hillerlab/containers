#!/usr/bin/env python3
"""Coordinate-mapping self-check for spliceai / splicetransformer WIG writers.

Covers reverse-complement of minus-strand input, terminal-only wigToBigWig clip
(middle chunks keep full inner length), and SpliceTransformer acceptor -2 shift.

Run: python3 test_wig_geometry.py   (no pytest, no torch, no model weights)
"""

from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPLICEAI_PY = HERE.parent.parent / "spliceai" / "py" / "spliceai.py"
SPLICETRANSFORMER_PY = HERE / "splicetransformer.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Logger:
    def info(self, msg):
        pass

    def warn(self, msg):
        pass

    def error(self, msg):
        pass

    def debug(self, msg):
        pass


LOGGER = _Logger()


def parse_wig(text: str) -> tuple[int, list[float]]:
    lines = text.splitlines()
    assert lines, "empty wig"
    assert lines[0].startswith("fixedStep"), lines[0]
    fields = dict(part.split("=", 1) for part in lines[0].split()[1:])
    start = int(fields["start"])
    values = [float(line) for line in lines[1:]]
    return start, values


def last_coord(start: int, n: int) -> int:
    return start + n - 1


def peak_index(values: list[float]) -> int:
    assert any(v > 0 for v in values), "expected a peak"
    return max(range(len(values)), key=lambda i: values[i])


def test_reverse_complement(st, spliceai) -> None:
    assert st.reverse_complement("ACGT") == "ACGT"
    assert st.reverse_complement("GT") == "AC"
    assert st.reverse_complement("AG") == "CT"
    assert spliceai.reverse_complement("TTGACA") == "TGTCAA"


def test_wig_inner_slice(mod) -> None:
    # middle chunk, both flanks: len=140, inner=100, offsets 20/20
    plus = mod.wig_inner_slice(140, 20, 20, plus=True)
    minus = mod.wig_inner_slice(140, 20, 20, plus=False)
    assert plus == (20, 120), plus
    assert minus == (20, 120), minus
    assert plus[1] - plus[0] == 100
    assert minus[1] - minus[0] == 100

    # terminal: no right flank, clip one value
    plus_t = mod.wig_inner_slice(120, 20, 0, plus=True)
    minus_t = mod.wig_inner_slice(120, 20, 0, plus=False)
    assert plus_t == (20, 119), plus_t
    assert minus_t == (1, 100), minus_t
    assert plus_t[1] - plus_t[0] == 99
    assert minus_t[1] - minus_t[0] == 99

    # chrom start: no left flank, keep full inner (right flank present)
    plus_s = mod.wig_inner_slice(120, 0, 20, plus=True)
    minus_s = mod.wig_inner_slice(120, 0, 20, plus=False)
    assert plus_s == (0, 100), plus_s
    assert minus_s == (20, 120), minus_s
    assert plus_s[1] - plus_s[0] == 100


def test_shift_acceptor(st) -> None:
    values = np.zeros(10)
    values[5] = 1.0
    shifted = st.shift_acceptor(values, 2)
    assert shifted[3] == 1.0
    assert shifted[5] == 0.0
    assert shifted[8] == 0.0 and shifted[9] == 0.0


class _CroppingZeros:
    """Mimic SpliceAI keras: output length = input length - CONTEXT."""

    def __init__(self, context: int, donor_at: int | None = None):
        self.context = context
        self.donor_at = donor_at

    def predict(self, x, verbose=0):
        cropped = x.shape[1] - self.context
        out = np.zeros((1, cropped, 3))
        if self.donor_at is not None and 0 <= self.donor_at < cropped:
            out[0, self.donor_at, 2] = 1.0
        return out


class _DecodeGtDonor:
    """Peak on GT in the cropped (transcript-oriented) SpliceAI input."""

    def __init__(self, context: int):
        self.context = context

    def predict(self, x, verbose=0):
        half = self.context // 2
        center = x[0, half : x.shape[1] - half]
        cropped = center.shape[0]
        out = np.zeros((1, cropped, 3))
        for i in range(cropped - 1):
            # G=[0,0,1,0] T=[0,0,0,1]
            if center[i, 2] > 0.5 and center[i + 1, 3] > 0.5:
                out[0, i, 2] = 1.0
                break
        return out


def _st_predict_gt_ag(seq_padded, models, device, logger, context_half: int):
    inner = seq_padded[context_half : len(seq_padded) - context_half]
    acc = np.zeros(len(inner))
    don = np.zeros(len(inner))
    for i in range(len(inner) - 1):
        if inner[i : i + 2] == "GT":
            don[i] = 1.0
        if inner[i : i + 2] == "AG" and i + 2 < len(inner):
            acc[i + 2] = 1.0
    return acc, don


def _st_predict_index(seq_padded, models, device, logger, context_half: int, idx: int):
    inner = seq_padded[context_half : len(seq_padded) - context_half]
    acc = np.zeros(len(inner))
    don = np.zeros(len(inner))
    acc[idx] = 1.0
    don[idx] = 1.0
    return acc, don


def run_process(module, header: str, seq: str, models, offset: int):
    handles = (io.StringIO(), io.StringIO(), io.StringIO(), io.StringIO())
    kwargs = dict(
        seq=seq,
        models=models,
        round_to=4,
        min_prob=0.001,
        offset=offset,
        wig_handles=handles,
        logger=LOGGER,
    )
    if hasattr(module, "ACCEPTOR_SHIFT"):
        kwargs["device"] = "cpu"
    module.process_record(header, **kwargs)
    return tuple(h.getvalue() for h in handles)


def test_terminal_clip(mod, make_models) -> None:
    # chrom end 1000, chunk [900, 1000), left flank 20, no right flank
    seq = "A" * 20 + "C" * 100
    acc_plus, donor_plus, acc_minus, donor_minus = run_process(
        mod, "chr19:900-1000(+)", seq, make_models(), 20
    )
    acc_plus2, donor_plus2, acc_minus2, donor_minus2 = run_process(
        mod, "chr19:900-1000(-)", seq, make_models(), 20
    )
    acc_plus += acc_plus2
    donor_plus += donor_plus2
    acc_minus += acc_minus2
    donor_minus += donor_minus2
    for name, wig in (
        ("acc_plus", acc_plus),
        ("donor_plus", donor_plus),
        ("acc_minus", acc_minus),
        ("donor_minus", donor_minus),
    ):
        start, values = parse_wig(wig)
        assert start == 902, f"{name}: start {start}"
        assert len(values) == 99, f"{name}: expected 99, got {len(values)}"
        assert last_coord(start, len(values)) == 1000, f"{name}: last past chrom end"


def test_middle_chunk_keeps_full_inner(mod, make_models) -> None:
    # chunk [100, 200), both flanks 20 — last WIG coord is end+1 (valid)
    seq = "A" * 20 + "C" * 100 + "G" * 20
    acc_plus, donor_plus, _, _ = run_process(
        mod, "chr19:100-200(+)", seq, make_models(), 20
    )
    for name, wig in (("acc_plus", acc_plus), ("donor_plus", donor_plus)):
        start, values = parse_wig(wig)
        assert start == 102, f"{name}: start {start}"
        assert len(values) == 100, f"{name}: expected 100, got {len(values)}"
        assert last_coord(start, len(values)) == 201, f"{name}: last {last_coord(start, len(values))}"


def test_st_acceptor_shift(st) -> None:
    seq = "N" * 20 + "N" * 100 + "N" * 20
    idx = 70  # within inner chunk [20, 120)

    def predict(seq_padded, models, device, logger):
        return _st_predict_index(
            seq_padded, models, device, logger, st.CONTEXT_HALF, idx
        )

    st._predict_windows = predict
    acc_plus, donor_plus, _, _ = run_process(
        st, "chr19:100-200(+)", seq, [object()], 20
    )
    acc_start, acc_vals = parse_wig(acc_plus)
    don_start, don_vals = parse_wig(donor_plus)
    assert acc_start == don_start == 102
    # slice starts at seq index 20; peak at 70 → unshifted slice index 50 → 1-based 152
    assert peak_index(don_vals) == 50, peak_index(don_vals)
    assert don_start + peak_index(don_vals) == 152
    # acceptor rolled 2 bp toward 5'
    assert peak_index(acc_vals) == 48, peak_index(acc_vals)
    assert acc_start + peak_index(acc_vals) == 150


def test_st_minus_rc_donor(st) -> None:
    # genomic AC at seq[70:72] is minus-strand GT after RC
    inner = ["N"] * 100
    inner[50] = "A"
    inner[51] = "C"
    seq = "N" * 20 + "".join(inner) + "N" * 20
    assert seq[70:72] == "AC"

    def predict(seq_padded, models, device, logger):
        return _st_predict_gt_ag(
            seq_padded, models, device, logger, st.CONTEXT_HALF
        )

    st._predict_windows = predict
    _, _, _, donor_minus = run_process(
        st, "chr19:100-200(-)", seq, [object()], 20
    )
    start, values = parse_wig(donor_minus)
    assert start == 102
    # RC: seq_rc[68] is G of GT (complement of genomic C at seq[71] = genomic 151)
    # minus slice [20:120][::-1] maps transcript 68 → reversed index 51 → 1-based 153
    assert peak_index(values) == 51, peak_index(values)
    assert start + peak_index(values) == 153


def test_spliceai_minus_rc_donor(spliceai) -> None:
    inner = ["N"] * 100
    inner[50] = "A"
    inner[51] = "C"
    seq = "N" * 20 + "".join(inner) + "N" * 20
    _, _, _, donor_minus = run_process(
        spliceai,
        "chr19:100-200(-)",
        seq,
        [_DecodeGtDonor(spliceai.CONTEXT)],
        20,
    )
    start, values = parse_wig(donor_minus)
    assert start == 102
    assert peak_index(values) == 51, peak_index(values)
    assert start + peak_index(values) == 153


def test_spliceai_plus_donor_unshifted(spliceai) -> None:
    # SpliceAI does not roll acceptor; donor at seq[70] stays slice index 50
    models = [_CroppingZeros(spliceai.CONTEXT, donor_at=70)]
    _, donor_plus, _, _ = run_process(
        spliceai, "chr19:100-200(+)", "N" * 140, models, 20
    )
    start, values = parse_wig(donor_plus)
    assert start == 102
    assert peak_index(values) == 50
    assert start + peak_index(values) == 152


def main() -> None:
    spliceai = load_module("spliceai_predict", SPLICEAI_PY)
    splicetransformer = load_module("splicetransformer_predict", SPLICETRANSFORMER_PY)

    test_reverse_complement(splicetransformer, spliceai)
    test_wig_inner_slice(splicetransformer)
    test_wig_inner_slice(spliceai)
    test_shift_acceptor(splicetransformer)

    os.environ["SPLICETRANSFORMER_MOCK"] = "1"

    def st_zeros():
        return [object()]

    def sa_zeros():
        return [_CroppingZeros(spliceai.CONTEXT)]

    test_terminal_clip(splicetransformer, st_zeros)
    test_terminal_clip(spliceai, sa_zeros)
    test_middle_chunk_keeps_full_inner(splicetransformer, st_zeros)
    test_middle_chunk_keeps_full_inner(spliceai, sa_zeros)

    os.environ.pop("SPLICETRANSFORMER_MOCK", None)
    test_st_acceptor_shift(splicetransformer)
    test_st_minus_rc_donor(splicetransformer)
    test_spliceai_minus_rc_donor(spliceai)
    test_spliceai_plus_donor_unshifted(spliceai)

    print(
        "OK: terminal clip last==end, middle n=chunk_length, "
        "minus RC donor maps to genomic GT, ST acceptor -2 vs donor"
    )


if __name__ == "__main__":
    main()
