#!/usr/bin/env python3
"""Wrap merge_annotations.py for Hiller scatter/gather.

merge_annotations.py emits GFF3 with gene IDs assigned by
(seqid, start, strand), not chunk order. This wrapper:

* runs the upstream merger
* restores FASTA record order from a sequence-order manifest
* writes GTF
* optionally extracts CDS/protein FASTAs from the merged annotation
  plus the original genome (never concatenates per-chunk FASTAs)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

STANDARD_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def parse_manifest(path: Path) -> list[str]:
    seqids = []
    with path.open(encoding="utf-8") as handle:
        for index, raw in enumerate(handle):
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if index == 0 and parts[0].lower() in {"order", "idx", "index"}:
                continue
            if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
                seqids.append(parts[1])
            else:
                seqids.append(parts[0])
    if not seqids:
        raise SystemExit(f"no sequence IDs in manifest {path}")
    return seqids


def parse_gff3_attributes(attr: str) -> dict[str, str]:
    out = {}
    if not attr or attr == ".":
        return out
    for part in attr.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key] = value
    return out


def attrs_to_gff3(attrs: dict[str, str]) -> str:
    if not attrs:
        return "."
    items = []
    for key in ("ID", "Parent"):
        if key in attrs:
            items.append(f"{key}={attrs[key]}")
    for key in sorted(k for k in attrs if k not in {"ID", "Parent"}):
        items.append(f"{key}={attrs[key]}")
    return ";".join(items)


def split_gff3(text: str) -> tuple[list[str], dict[str, list[str]]]:
    header: list[str] = []
    blocks: dict[str, list[str]] = defaultdict(list)
    order: list[str] = []
    for line in text.splitlines(keepends=True):
        if not line.strip() or line.startswith("#"):
            if not blocks:
                header.append(line)
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 1:
            continue
        seqid = cols[0]
        if seqid not in blocks:
            order.append(seqid)
        blocks[seqid].append(line if line.endswith("\n") else line + "\n")
    blocks["_order"] = order  # type: ignore[assignment]
    return header, blocks


def reorder_gff3(text: str, seqids: list[str]) -> str:
    header, blocks = split_gff3(text)
    original = list(blocks.get("_order", []))
    blocks.pop("_order", None)
    seen = set()
    out = []
    if not any(line.startswith("##gff-version") for line in header):
        out.append("##gff-version 3\n")
    out.extend(header)
    for seqid in seqids:
        if seqid in blocks:
            out.extend(blocks[seqid])
            seen.add(seqid)
    for seqid in original:
        if seqid not in seen and seqid in blocks:
            out.extend(blocks[seqid])
    return "".join(out)


def gff3_to_gtf(text: str) -> str:
    tx_to_gene: dict[str, str] = {}
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 9:
            continue
        rows.append(cols)
        attrs = parse_gff3_attributes(cols[8])
        ftype = cols[2]
        if ftype in {"transcript", "mRNA"} and "ID" in attrs:
            tx_to_gene[attrs["ID"]] = attrs.get("Parent", attrs["ID"])

    lines = []
    for cols in rows:
        seqid, source, ftype, start, end, score, strand, phase, attr_str = cols
        attrs = parse_gff3_attributes(attr_str)
        if ftype == "gene":
            gene_id = attrs.get("ID", "")
            gtf_attr = f'gene_id "{gene_id}";'
        elif ftype in {"transcript", "mRNA"}:
            tx_id = attrs.get("ID", "")
            gene_id = attrs.get("Parent", tx_to_gene.get(tx_id, tx_id))
            gtf_ftype = "transcript"
            gtf_attr = f'gene_id "{gene_id}"; transcript_id "{tx_id}";'
            ftype = gtf_ftype
        else:
            parent = attrs.get("Parent", "")
            tx_id = parent.split(",")[0]
            gene_id = tx_to_gene.get(tx_id, tx_id)
            gtf_attr = f'gene_id "{gene_id}"; transcript_id "{tx_id}";'
        lines.append(
            "\t".join([seqid, source, ftype, start, end, score, strand, phase, gtf_attr])
            + "\n"
        )
    return "".join(lines)


def load_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    name = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line.strip())
    return {key: "".join(val) for key, val in seqs.items()}


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def translate(cds: str) -> str:
    seq = cds.upper().replace("U", "T")
    aa = []
    for i in range(0, len(seq) - len(seq) % 3, 3):
        residue = STANDARD_TABLE.get(seq[i : i + 3], "X")
        if residue == "*":
            break
        aa.append(residue)
    return "".join(aa)


def collect_transcripts(text: str) -> dict[str, dict]:
    transcripts: dict[str, dict] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 9:
            continue
        seqid, _source, ftype, start, end, _score, strand, phase, attr_str = cols
        attrs = parse_gff3_attributes(attr_str)
        if ftype in {"transcript", "mRNA"}:
            tx_id = attrs.get("ID")
            if not tx_id:
                continue
            transcripts.setdefault(
                tx_id,
                {"seqid": seqid, "strand": strand, "cds": []},
            )
            transcripts[tx_id]["seqid"] = seqid
            transcripts[tx_id]["strand"] = strand
            continue
        if ftype != "CDS":
            continue
        parent = attrs.get("Parent", "").split(",")[0]
        if not parent:
            continue
        transcripts.setdefault(
            parent,
            {"seqid": seqid, "strand": strand, "cds": []},
        )
        transcripts[parent]["cds"].append(
            (int(start), int(end), phase if phase not in {".", ""} else "0")
        )
    return transcripts


def extract_sequences(text: str, fasta: dict[str, str]) -> tuple[str, str]:
    transcripts = collect_transcripts(text)
    cds_out = []
    prot_out = []
    for tx_id, rec in transcripts.items():
        if not rec["cds"]:
            continue
        seq = fasta.get(rec["seqid"])
        if seq is None:
            raise SystemExit(f"sequence {rec['seqid']!r} missing from genome FASTA")
        strand = rec["strand"]
        features = sorted(rec["cds"], key=lambda item: item[0])
        if strand == "-":
            features = list(reversed(features))
        pieces = []
        first_phase = int(features[0][2]) if features else 0
        for start, end, _phase in features:
            piece = seq[start - 1 : end]
            if strand == "-":
                piece = reverse_complement(piece)
            pieces.append(piece)
        cds = "".join(pieces)[first_phase:]
        cds_out.append(f">{tx_id}\n")
        for i in range(0, len(cds), 60):
            cds_out.append(cds[i : i + 60] + "\n")
        protein = translate(cds)
        prot_out.append(f">{tx_id}\n")
        for i in range(0, len(protein), 60):
            prot_out.append(protein[i : i + 60] + "\n")
    return "".join(cds_out), "".join(prot_out)


def find_merger() -> str:
    found = shutil.which("merge_annotations.py")
    if found:
        return found
    candidates = [
        Path("/opt/tiberius/tiberius/scripts/merge_annotations.py"),
    ]
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if directory:
            candidates.append(Path(directory) / "merge_annotations.py")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise SystemExit("merge_annotations.py not found on PATH or at /opt/tiberius")


def run_merger(inputs: list[Path]) -> str:
    cmd = [sys.executable, find_merger(), "--mode", "full", *[str(p) for p in inputs]]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--genome", type=Path)
    parser.add_argument("--protseq", action="store_true")
    parser.add_argument("--codingseq", action="store_true")
    parser.add_argument("annotations", nargs="+", type=Path)
    args = parser.parse_args(argv)

    missing = [path for path in args.annotations if not path.is_file()]
    if missing:
        raise SystemExit(f"missing annotation file(s): {missing}")
    if (args.protseq or args.codingseq) and args.genome is None:
        raise SystemExit("--genome is required when extracting protein/CDS FASTAs")

    merged = run_merger(args.annotations)
    ordered = reorder_gff3(merged, parse_manifest(args.manifest))
    gtf = gff3_to_gtf(ordered)

    gff_path = Path(f"{args.out_prefix}.gff3")
    gtf_path = Path(f"{args.out_prefix}.gtf")
    gff_path.write_text(ordered, encoding="utf-8")
    gtf_path.write_text(gtf, encoding="utf-8")

    if args.protseq or args.codingseq:
        fasta = load_fasta(args.genome)
        cds, prot = extract_sequences(ordered, fasta)
        if args.codingseq:
            Path(f"{args.out_prefix}.cds").write_text(cds, encoding="utf-8")
        if args.protseq:
            Path(f"{args.out_prefix}.prot").write_text(prot, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
