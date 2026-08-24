#!/usr/bin/env python3
"""Self-check for the wigToBigWig off-by-one fix.

Reproduces the chr19.1023 crash geometry (terminal chunk of a chromosome):
the fixedStep wig must end at `end` (chromosome length for a terminal chunk),
never at `end+1` (which makes wigToBigWig exit 255).

Run: python3 test_wig_geometry.py   (no pytest, no torch, no model needed)
"""

from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path

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

# Terminal chunk: chr length 1000, chunk [900, 1000), flank 20.
# seq = 20 flanks + 100 chunk bases + 20 flanks (forward orientation, as chunker emits).
seq = "A" * 20 + "C" * 100 + "G" * 20


def check_wig(module, plus_header: str, minus_header: str) -> None:
    os.environ["SPLICETRANSFORMER_MOCK"] = "1"
    handles = (io.StringIO(), io.StringIO(), io.StringIO(), io.StringIO())
    kwargs = dict(
        seq=seq,
        models=[object()],
        round_to=4,
        min_prob=0.001,
        offset=20,
        wig_handles=handles,
        logger=LOGGER,
    )
    module.process_record(plus_header, **kwargs)
    module.process_record(minus_header, **kwargs)
    acc_plus, donor_plus, acc_minus, donor_minus = [h.getvalue() for h in handles]

    for name, wig in (
        ("acc_plus", acc_plus),
        ("donor_plus", donor_plus),
        ("acc_minus", acc_minus),
        ("donor_minus", donor_minus),
    ):
        lines = wig.splitlines()
        assert lines[0].startswith("fixedStep chrom=chr19 start=902 step=1 span=1"), (
            f"{name}: bad header: {lines[0]}"
        )
        count = len(lines) - 1
        assert count == 100 - 1, f"{name}: expected 99 values, got {count}"
        last_pos = 902 + count - 1
        assert last_pos == 1000, (
            f"{name}: last value at {last_pos}, expected 1000 (chunk end)"
        )


def main() -> None:
    spliceai = load_module("spliceai_predict", SPLICEAI_PY)
    splicetransformer = load_module("splicetransformer_predict", SPLICETRANSFORMER_PY)

    class FakeModel:
        def predict(self, x, verbose=0):
            import numpy as np

            return np.zeros((1, x.shape[1], 3))

    # spliceai: mock via fake keras model (no numpy import issues at module level)
    spliceai_handles = (io.StringIO(), io.StringIO(), io.StringIO(), io.StringIO())
    spliceai.process_record(
        "chr19:900-1000(+)", seq, [FakeModel()], 4, 0.001, 20, spliceai_handles, LOGGER
    )
    spliceai.process_record(
        "chr19:900-1000(-)", seq, [FakeModel()], 4, 0.001, 20, spliceai_handles, LOGGER
    )
    for h in spliceai_handles:
        lines = h.getvalue().splitlines()
        assert lines[0].startswith("fixedStep chrom=chr19 start=902"), lines[0]
        assert len(lines) - 1 == 99, (
            f"spliceai: expected 99 values, got {len(lines) - 1}"
        )
        assert 902 + (len(lines) - 1) - 1 == 1000, "spliceai: last value past chunk end"

    check_wig(splicetransformer, "chr19:900-1000(+)", "chr19:900-1000(-)")

    print(
        "OK: wigs end exactly at `end` (1000), 99 values, start=902 — no end+1 overrun"
    )


if __name__ == "__main__":
    main()
