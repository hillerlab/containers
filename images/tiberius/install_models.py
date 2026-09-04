#!/usr/bin/env python3
"""Download, verify, and extract a subset of Tiberius weight archives."""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path


def parse_tsv(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        header = None
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) != len(header):
                raise SystemExit(f"malformed models.tsv row: {line!r}")
            rows.append(dict(zip(header, parts)))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def extract_archive(archive: Path, dest_dir: Path) -> None:
    print(f"extracting {archive.name}", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated hiller aliases, or 'all'",
    )
    parser.add_argument(
        "--cfg-dir",
        type=Path,
        default=None,
        help="If set, copy alias.yaml files next to upstream YAMLs",
    )
    args = parser.parse_args()

    rows = parse_tsv(args.tsv)
    by_alias = {row["hiller_alias"]: row for row in rows}
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    if selected == ["all"]:
        selected = list(by_alias)
    unknown = [name for name in selected if name not in by_alias]
    if unknown:
        known = ", ".join(by_alias)
        raise SystemExit(f"unknown model alias(es): {', '.join(unknown)}. Known: {known}")

    args.dest.mkdir(parents=True, exist_ok=True)
    seen_urls: set[str] = set()
    for alias in selected:
        row = by_alias[alias]
        url = row["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        filename = url.rsplit("/", 1)[-1]
        archive = args.dest / filename
        download(url, archive)
        actual = sha256_file(archive)
        expected = row["sha256"].lower()
        if actual != expected:
            raise SystemExit(
                f"SHA256 mismatch for {filename}: expected {expected}, got {actual}"
            )
        extract_archive(archive, args.dest)
        archive.unlink()
        stem = filename[:-7] if filename.endswith(".tar.gz") else Path(filename).stem
        extracted = args.dest / stem
        if not extracted.is_dir() or not any(extracted.iterdir()):
            raise SystemExit(f"expected extracted weights at {extracted}")
        print(f"installed {alias} -> {extracted}", flush=True)

    if args.cfg_dir:
        for row in rows:
            alias = row["hiller_alias"]
            src = args.cfg_dir / row["upstream_yaml"]
            dst = args.cfg_dir / f"{alias}.yaml"
            if src.is_file() and not dst.exists():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"alias {dst.name} -> {src.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
