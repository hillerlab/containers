#!/usr/bin/env python3
"""Download and verify a subset of OrionGeno lineage checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


HF_REVISION = "e1fd73c7e83ab6f16ed7850bba057facfba70b56"
HF_BASE = (
    "https://huggingface.co/BGI-Research/OrionGeno/resolve/"
    f"{HF_REVISION}/checkpoints"
)
REQUIRED_FILES = ("config.json", "pytorch_model.bin")


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
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "hillerlab-oriongeno"},
    )
    print(f"downloading {url}", flush=True)
    temporary = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        temporary.replace(dest)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated lineage aliases, or 'all'",
    )
    parser.add_argument(
        "--require",
        type=Path,
        default=None,
        help="Checkpoint directory that must be installed (the image default)",
    )
    args = parser.parse_args()

    rows = parse_tsv(args.tsv)
    by_alias: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_alias.setdefault(row["alias"], []).append(row)

    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    if selected == ["all"]:
        selected = list(by_alias)
    if not selected:
        raise SystemExit("no models selected")
    unknown = [name for name in selected if name not in by_alias]
    if unknown:
        known = ", ".join(by_alias)
        raise SystemExit(
            f"unknown model alias(es): {', '.join(unknown)}. Known: {known}"
        )

    args.dest.mkdir(parents=True, exist_ok=True)
    for alias in selected:
        lineage_dir = args.dest / f"oriongeno_{alias}"
        for row in by_alias[alias]:
            filename = row["filename"]
            url = f"{HF_BASE}/oriongeno_{alias}/{filename}"
            destination = lineage_dir / filename
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    download(url, destination)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(
                        f"download attempt {attempt} failed for {filename}: {exc}",
                        flush=True,
                    )
            if last_error is not None:
                raise SystemExit(
                    f"failed to download {url} after 3 attempts: {last_error}"
                )
            actual = sha256_file(destination)
            expected = row["sha256"].lower()
            if actual != expected:
                destination.unlink(missing_ok=True)
                raise SystemExit(
                    f"SHA256 mismatch for {alias}/{filename}: "
                    f"expected {expected}, got {actual}"
                )
        for filename in REQUIRED_FILES:
            path = lineage_dir / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"missing checkpoint file: {path}")
        print(f"installed {alias} -> {lineage_dir}", flush=True)

    if args.require is not None:
        if not args.require.is_dir():
            raise SystemExit(
                f"required checkpoint directory is missing: {args.require}. "
                "Include that lineage in ORIONGENO_MODELS."
            )
        for filename in REQUIRED_FILES:
            path = args.require / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"required checkpoint file is missing: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
