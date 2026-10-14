#!/usr/bin/env python3
"""CPU-only image check; run inside the built image with this file mounted at /tests."""

import contextlib
import hashlib
import importlib.util
import io
import os
import subprocess
import tempfile
from pathlib import Path

RUNNER = Path("/usr/local/libexec/spliceai2-runner.py")
spec = importlib.util.spec_from_file_location("spliceai2_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def check_encoding_contract():
    import numpy as np
    from spliceai2.dataset import _one_hot_encode

    encoder = runner.Predictor([], "cpu", "fp32")
    plus_input, plus_start = runner.locus_sequence("NRACGT", 0, 4, True, 2)
    minus_input, minus_start = runner.locus_sequence("NRACGT", 0, 4, False, 2)
    assert plus_start == minus_start == 0
    assert runner.record_layout("toy:0-4(+)", "nRaC", 2)[4] == "NRAC"
    assert runner.record_layout("toy:0-4(-)", "nRaC", 2)[4] == "CARN"
    for plus, strand in ((True, "+"), (False, "-")):
        layout = runner.record_layout(f"toy:0-4({strand})", "NRACGT", 2)
        window = runner.window_sequence(layout[4], layout[5], *next(runner.windows(4, 16)))
        forward = window if plus else window[::-1]
        expected = _one_hot_encode(forward, "GRCh38")
        if not plus:
            expected[:4] = expected[:4][::-1, ::-1]
        actual = encoder.encode([window], "GRCh38", plus)[0].numpy()
        assert np.array_equal(actual, expected)
    plus_encoded = encoder.encode([plus_input], "GRCh38", True)[0, :4].numpy()
    minus_encoded = encoder.encode([minus_input], "GRCh38", False)[0, :4].numpy()
    expected_plus = _one_hot_encode(plus_input, "GRCh38")
    forward_minus = minus_input[::-1]
    expected_minus = _one_hot_encode(forward_minus, "GRCh38")
    expected_minus[:4] = expected_minus[:4][::-1, ::-1]
    assert np.array_equal(plus_encoded, expected_plus[:4])
    assert np.array_equal(minus_encoded, expected_minus[:4])
    plus_channels = plus_encoded.argmax(axis=0)
    minus_channels = minus_encoded.argmax(axis=0)
    assert all(
        plus_channels[position] == 0
        for position in (0, plus_input.index("R") - 1, plus_input.index("R"))
    )
    assert all(
        minus_channels[len(forward_minus) - 1 - position] == 3
        for position in (0, forward_minus.index("R") - 1, forward_minus.index("R"))
    )


def candidate_probe(torch):
    class ZeroJunctionModel:
        def forward_2d(self, features, idxs):
            return torch.zeros((idxs.shape[0], idxs.shape[1], idxs.shape[1]))

    return runner.Predictor([ZeroJunctionModel(), ZeroJunctionModel()], "cpu", "fp32")


def select_candidates(predictor, torch, plus, offset, max_candidates, outside_high):
    sequence, mask_start = runner.locus_sequence("A" * 100, 0, 100, plus, offset)
    output_length = len(sequence) - runner.CONTEXT
    assert output_length > mask_start + 100
    low, high, outside = mask_start + 49, mask_start + 50, mask_start + 100
    logits = torch.full((2, 1, 2, output_length), -20.0)
    logits[:, 0, 0, low] = -6.0
    logits[0, 0, 1, high], logits[1, 0, 1, high] = -10.0, -2.0
    if outside_high:
        logits[:, 0, :, outside] = 20.0
    predictor.encode = lambda *args, **kwargs: torch.empty((1, 4, 1))
    predictor.forward = lambda x: (logits, [None, None])
    try:
        result = predictor.transcripts(
            sequence, "GRCh38", 1, plus, mask_start, 100, max_candidates
        )
    except TypeError as error:
        if "were given" not in str(error):
            raise
        result = predictor.transcripts(sequence, "GRCh38", 1)
    return result, mask_start, output_length


def check_locus_bounds():
    import torch

    predictor = candidate_probe(torch)
    for strand, offset in ((True, 0), (False, 0), (True, 65_540), (False, 65_540)):
        (masks, _, idxs), mask_start, output_length = select_candidates(
            predictor, torch, strand, offset, 1, True
        )
        assert idxs.tolist() == [mask_start + 50], idxs.tolist()
        assert all(mask_start <= index < mask_start + 100 for index in idxs)
        assert len(masks[0]) == output_length
    (masks, _, idxs), mask_start, _ = select_candidates(
        predictor, torch, True, 0, 1024, True
    )
    assert idxs.tolist() == list(range(mask_start, mask_start + 100))
    core_mask = masks[0][mask_start:mask_start + 100]
    assert all(core_mask)
    single_exon = runner.bed12_row("toy", 0, 100, "unspliced", True, core_mask, 1, 500)
    assert single_exon.split("\t")[9:12] == ["1", "100", "0"]
    assert runner.parser().parse_args([
        "transcripts", "--genome", "/tmp/g.fa", "--loci", "/tmp/l.bed"
    ]).max_candidates == 1024


def check_member_reduction():
    import torch

    predictor = candidate_probe(torch)
    high_probability = (torch.sigmoid(torch.tensor(-10.0)) + torch.sigmoid(torch.tensor(-2.0))) / 2
    low_probability = torch.sigmoid(torch.tensor(-6.0))
    assert abs(high_probability.item() - 0.05962416) < 1e-7
    assert abs(low_probability.item() - 0.00247262) < 1e-7
    assert high_probability > low_probability
    (masks, _, idxs), mask_start, _ = select_candidates(
        predictor, torch, True, 0, 1, False
    )
    assert idxs is not None, "member-mean probability candidate was dropped"
    assert idxs.tolist() == [mask_start + 50], idxs.tolist()
    assert all(mask_start <= index < mask_start + 100 for index in idxs)


def check_oom_handling():
    import numpy as np

    class Genome:
        def keys(self):
            return ["toy"]

        def __getitem__(self, chrom):
            return "A" * 64

        def close(self):
            pass

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        genome_path = root / "genome.fa"
        loci_path = root / "loci.bed"
        genome_path.write_text(">toy\n" + "A" * 64 + "\n")
        loci_path.write_text(
            "toy\t0\t32\tinjected_oom\t0\t+\n"
            "toy\t32\t64\tcompleted\t0\t-\n"
            "toy\t0\t64\ttoo_long\t0\t+\n"
        )
        outdir = root / "out"
        original = (
            runner.open_genome, runner.model_paths, runner.resolve_runtime,
            runner.load_models, runner.Predictor.transcripts, runner.sys.argv,
        )
        calls = 0

        def inject_memory_error(self, sequence, species, k, *options):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise MemoryError("injected allocation failure")
            _, mask_start, locus_length, _ = options
            out_length = len(sequence) - runner.CONTEXT
            assert 0 <= mask_start < mask_start + locus_length <= out_length
            return [np.ones(out_length, dtype=np.int32)], np.zeros((1, 1)), np.array([0])

        try:
            runner.open_genome = lambda path: Genome()
            runner.model_paths = lambda path: []
            runner.resolve_runtime = lambda *args: ("cpu", "fp32")
            runner.load_models = lambda *args: []
            runner.Predictor.transcripts = inject_memory_error
            runner.sys.argv = [
                "spliceai2", "transcripts", "--genome", str(genome_path),
                "--loci", str(loci_path), "--max-length", "32", "--offset", "0",
                "--device", "cpu", "--precision", "fp32", "--no-compile",
                "-o", str(outdir),
            ]
            assert runner.main() is None
        finally:
            (runner.open_genome, runner.model_paths, runner.resolve_runtime,
             runner.load_models, runner.Predictor.transcripts, runner.sys.argv) = original
        assert (outdir / "loci.skipped.bed").read_text().splitlines() == [
            "toy\t0\t64\ttoo_long\t0\t+\tmax_length",
            "toy\t0\t32\tinjected_oom\t0\t+\tout_of_memory",
        ]
        rows = [line.split("\t") for line in (outdir / "loci.transcripts.bed").read_text().splitlines()]
        assert len(rows) == 1 and rows[0][3] == "completed.t1" and rows[0][5] == "-"


def check():
    import huggingface_hub
    import lightning
    import torch
    from spliceai2.model import SpliceAI2

    check_encoding_contract()
    check_locus_bounds()
    check_member_reduction()
    check_oom_handling()

    class FixedModel:
        def __init__(self, logit):
            self.logit = logit

        def __call__(self, x):
            return torch.full((x.shape[0], 2, x.shape[2]), self.logit)

        def forward_1d(self, features):
            return features

    predictor = runner.Predictor([FixedModel(0.0), FixedModel(2.0)], "cpu", "fp32")
    predictor.encode = lambda sequences, species, plus=True: torch.zeros((1, 14, 16))
    member_mean_probability = predictor.splice_sites(["A" * 16], "GRCh38")[0, 0, 0]
    expected = torch.sigmoid(torch.tensor([0.0, 2.0])).mean().item()
    assert abs(member_mean_probability - expected) < 1e-7

    old_models = os.environ.get("SPLICEAI2_MODELS")
    os.environ["SPLICEAI2_MODELS"] = "/custom/models"
    assert runner.default_models_dir() == Path("/custom/models")
    assert runner.parser().parse_args(["predict", "--sequence", "/tmp/toy.fa"]).models == Path("/custom/models")
    os.environ.pop("SPLICEAI2_MODELS")
    assert runner.default_models_dir() == Path("/models")
    assert runner.parser().parse_args(["predict", "--sequence", "/tmp/toy.fa"]).models == Path("/models")
    if old_models is not None:
        os.environ["SPLICEAI2_MODELS"] = old_models

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            try:
                runner.model_paths(root / "missing")
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError("missing weights were accepted")
        assert "docker run -v <host dir>:/models" in stderr.getvalue()
        assert "apptainer run --bind <host dir>:/models" in stderr.getvalue()
        assert "fetch-models --out DIR" in stderr.getvalue()

        real_download = huggingface_hub.snapshot_download
        captured = io.StringIO()

        def fake_download(**kwargs):
            assert kwargs["repo_id"] == runner.HF_REPO
            assert kwargs["revision"] == "test-revision"
            assert "token" not in kwargs
            for name in kwargs["allow_patterns"]:
                path = Path(kwargs["local_dir"]) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())

        try:
            huggingface_hub.snapshot_download = fake_download
            with contextlib.redirect_stdout(captured):
                runner.fetch_models(root / "download", "test-revision")
        finally:
            huggingface_hub.snapshot_download = real_download
        lines = captured.getvalue().splitlines()
        assert len(lines) == 2
        for folder, line in zip(runner.MODEL_FOLDERS, lines):
            path = root / "download" / folder / "model.ckpt"
            assert line == f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}"
        original_download = huggingface_hub.snapshot_download

        class GatedRepoError(Exception):
            pass

        try:
            huggingface_hub.snapshot_download = lambda **kwargs: (_ for _ in ()).throw(GatedRepoError())
            with contextlib.redirect_stderr(stderr := io.StringIO()):
                try:
                    runner.fetch_models(root / "gated", None)
                except SystemExit as error:
                    assert error.code == 2
                else:
                    raise AssertionError("gated fetch was accepted")
        finally:
            huggingface_hub.snapshot_download = original_download
        assert len(stderr.getvalue().splitlines()) == 1 and "accept the terms" in stderr.getvalue()

        models = root / "models"
        torch.manual_seed(17)
        model = SpliceAI2(
            num_features=4, num_blocks=1, kernel_size=3, dilation=(1,), pool=(1,),
            crop_len=runner.CONTEXT,
        )
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": lightning.__version__,
        }
        for folder in runner.MODEL_FOLDERS:
            path = models / folder / "model.ckpt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, path)

        fasta = root / "toy.fa"
        fasta.write_text(">toy\n" + "ACGT" * 16 + "\n")
        chunks = root / "chunked"
        subprocess.run(
            ["spliceai2", "chunk", "-s", str(fasta), "-c", "32", "-f", "8", "-m", "1", "-o", str(chunks)],
            check=True,
        )
        shards = sorted((chunks / "chunks").glob("*.fa"))
        records = [record for shard in shards for record in runner.iter_fasta(shard)]
        assert len(records) == 4 and {runner.parse_header(h)[3] for h, _ in records} == {True, False}
        for header, bases in records:
            chrom, start, end, plus, sense, core_start, _ = runner.record_layout(header, bases, 8)
            tile = next(runner.windows(end - start, 16))
            padded = runner.window_sequence(sense, core_start, *tile)
            assert len(padded) == runner.CONTEXT + tile[2]
            assert padded[runner.FLANK:runner.FLANK + tile[1]] == sense[core_start:core_start + tile[1]]
        tail = list(runner.windows(34, 16))[-1]
        padded_tail = runner.window_sequence("C" * 8 + "A" * 34 + "G" * 8, 8, *tail)
        assert tail == (32, 2, 16)
        assert padded_tail[runner.FLANK:runner.FLANK + 16] == "A" * 16
        shard = root / "chunks.fa"
        shard.write_text("\n".join(path.read_text().strip() for path in shards) + "\n")

        common = ["--sequence", str(shard), "--models", str(models), "--offset", "8",
                  "--device", "cpu", "--precision", "fp32", "--no-compile"]
        wigdir = root / "wig"
        subprocess.run(
            ["spliceai2", "predict", *common, "--window", "16", "--batch-size", "2", "-o", str(wigdir)],
            check=True,
        )
        wig_files = sorted(wigdir.glob("*.wig"))
        assert len(wig_files) == 4
        for path in wig_files:
            lines = path.read_text().splitlines()
            headers = [line for line in lines if line.startswith("fixedStep ")]
            values = [line for line in lines if not line.startswith("fixedStep ")]
            assert len(headers) == 2 and len(values) == 63
            assert headers[0] == "fixedStep chrom=toy start=2 step=1 span=1"
            assert headers[1] == "fixedStep chrom=toy start=34 step=1 span=1"

        assert runner.genomic_wig_values([1, 2, 3, 4, 5], True, False) == [1, 2, 3, 4, 5]
        assert runner.genomic_wig_values([1, 2, 3, 4, 5], False, False) == [3, 2, 1, 0.0, 0.0]
        assert runner.genomic_wig_values([1, 2, 3, 4, 5], False, True) == [3, 2, 0.0, 0.0]
        assert runner.WIG_HEADER.format("toy", 100 + 2) == "fixedStep chrom=toy start=102 step=1 span=1\n"
        assert [len(batch) for batch in runner.window_batches([(0, 16, 16), (16, 16, 16), (32, 8, 16)], 2)] == [2, 1]
        plus_input, plus_start = runner.locus_sequence("AACCGTGA", 0, 2, True, 2)
        minus_input, minus_start = runner.locus_sequence("AACCGTGA", 6, 8, False, 2)
        assert plus_start == minus_start == 0
        assert plus_input[runner.FLANK - 2:runner.FLANK + 4] == "NNAACC"
        minus_forward = minus_input[::-1]
        minus_core = minus_forward.index("GA")
        assert minus_forward[minus_core - 2:minus_core + 4] == "GTGANN"

        plus = runner.bed12_row("toy", 100, 106, "plus", True, [1, 1, 0, 1, 1, 1], 1, 500)
        minus = runner.bed12_row("toy", 100, 106, "minus", False, [1, 1, 0, 1, 1, 1], 1, 500)
        plus_fields, minus_fields = plus.split("\t"), minus.split("\t")
        assert plus_fields[1:3] == ["100", "106"]
        assert plus_fields[9:12] == ["2", "2,3", "0,3"]
        assert minus_fields[1:3] == ["100", "106"]
        assert minus_fields[9:12] == ["2", "3,2", "0,4"]
        terminal = runner.bed12_row("toy", 100, 106, "edge", True, [0, 1, 0, 1, 0, 0], 1, 500)
        assert terminal.split("\t")[9:12] == ["3", "2,1,1", "0,3,5"]

        for invalid, line in (
            ("missing\t0\t2\tx\t0\t+\n", 1),
            ("toy\t2\t2\tx\t0\t+\n", 1),
            ("toy\t0\t5\tx\t0\t+\n", 1),
            ("toy\t0\t2\tx\t0\t.\n", 1),
            ("toy\t0\t2\tx\t0\t+\ntoy\t2\t4\tx\t0\t-\n", 2),
        ):
            invalid_bed = root / "invalid.bed"
            invalid_bed.write_text(invalid)
            try:
                runner.read_loci(invalid_bed, {"toy": "ACGT"})
            except ValueError as error:
                assert f"line {line}" in str(error), error
            else:
                raise AssertionError(f"invalid BED was accepted: {invalid!r}")
        unindexed = root / "unindexed.fa"
        unindexed.write_text(">toy\nACGT\n")
        access = runner.os.access
        try:
            runner.os.access = lambda path, mode: False
            try:
                runner.open_genome(unindexed)
            except ValueError as error:
                assert "samtools faidx" in str(error)
            else:
                raise AssertionError("an unwritable missing FASTA index was accepted")
        finally:
            runner.os.access = access

        beddir = root / "bed"
        loci = root / "toy.bed"
        loci.write_text("toy\t0\t32\tplus_locus\t0\t+\ntoy\t32\t64\tminus_locus\t0\t-\ntoy\t0\t64\tskipped_locus\t0\t+\n")
        transcript_common = ["--genome", str(fasta), "--loci", str(loci), "--models", str(models),
                             "--offset", "8", "--max-length", "32", "--device", "cpu",
                             "--precision", "fp32", "--no-compile"]
        subprocess.run(
            ["spliceai2", "transcripts", *transcript_common, "-k", "3", "-o", str(beddir)],
            check=True,
        )
        rows = [line.split("\t") for line in (beddir / "toy.transcripts.bed").read_text().splitlines()]
        assert rows and {row[5] for row in rows} == {"+", "-"}
        assert {row[3].split(".t", 1)[0] for row in rows} == {"plus_locus", "minus_locus"}
        assert {(int(row[1]), int(row[2])) for row in rows} == {(0, 32), (32, 64)}
        assert [(int(row[1]), int(row[2])) for row in rows] == sorted((int(row[1]), int(row[2])) for row in rows)
        assert (beddir / "toy.skipped.bed").read_text().splitlines() == [
            "toy\t0\t64\tskipped_locus\t0\t+\tmax_length"
        ]
        assert all(len(row) == 12 for row in rows)

        print(f"PASS: defaults, BED refusals, locus extraction, both-strand BED12 blocks and WIG, CPU smoke ({len(rows)} BED12 rows)")


if __name__ == "__main__":
    check()
