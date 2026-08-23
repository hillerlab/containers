# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

# ---------- Build Stage - Rust ----------
FROM rust:1.93-bookworm AS rust-builder
WORKDIR /build

COPY modules/splicetransformer/rs/chunk/Cargo.toml modules/splicetransformer/rs/chunk/Cargo.lock ./splicetransformer/chunk/
COPY modules/splicetransformer/rs/chunk/src ./splicetransformer/chunk/src

COPY modules/splicetransformer/rs/derive/Cargo.toml modules/splicetransformer/rs/derive/Cargo.lock ./splicetransformer/derive/
COPY modules/splicetransformer/rs/derive/src ./splicetransformer/derive/src

RUN cargo build --release --locked --manifest-path ./splicetransformer/chunk/Cargo.toml
RUN cargo build --release --locked --manifest-path ./splicetransformer/derive/Cargo.toml

# ---------- Build Stage - Python ----------
FROM python:3.11-slim-bookworm AS runtime
ARG MODULE_VERSION
ARG SPTRANSFORMER_MODEL_URL=https://media.githubusercontent.com/media/ShenLab-Genomics/SpliceTransformer/release1/model/weights/SpTransformer_pytorch.ckpt
ARG SPTRANSFORMER_MODEL_SHA256=bf9937c234139f850a770f84a26c65213a80d696d82426949a432ed1ccb9aa4e
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /work

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    "numpy<2" \
    "torch==2.4.1" \
    "sinkhorn-transformer" \
    "axial-positional-embedding"

COPY --from=rust-builder /build/splicetransformer/chunk/target/release/splicetransformer-chunk /usr/local/bin/splicetransformer-chunk
COPY --from=rust-builder /build/splicetransformer/derive/target/release/splicetransformer-derive /usr/local/bin/splicetransformer-derive
COPY modules/splicetransformer/py/splicetransformer.py /usr/local/bin/splicetransformer.py
COPY modules/splicetransformer/py/model /usr/local/bin/model

# Keep the 120 MB checkpoint out of this repository; fetch the pinned upstream GitHub LFS object.
RUN mkdir -p /models \
    && python - "${SPTRANSFORMER_MODEL_URL}" "${SPTRANSFORMER_MODEL_SHA256}" <<'PY'
import hashlib
import os
import sys
import tempfile
import urllib.request

url, expected_sha256 = sys.argv[1:]
destination = "/models/SpTransformer_pytorch.ckpt"
request = urllib.request.Request(url, headers={"User-Agent": "hillerlab-splicetransformer"})
with urllib.request.urlopen(request, timeout=300) as response:
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir="/models", delete=False) as output:
        temporary = output.name
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)

actual_sha256 = digest.hexdigest()
if actual_sha256 != expected_sha256:
    os.unlink(temporary)
    raise SystemExit(
        f"SpliceTransformer checkpoint checksum mismatch: expected {expected_sha256}, got {actual_sha256}"
    )
os.replace(temporary, destination)
PY

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN chmod 0755 /usr/local/bin/splicetransformer.py \
    && printf '%s\n' \
        '#!/bin/sh' \
        'set -eu' \
        '' \
        'subcommand="${1:-}"' \
        '' \
        'case "$subcommand" in' \
        '  chunk)' \
        '    shift' \
        '    exec /usr/local/bin/splicetransformer-chunk "$@"' \
        '    ;;' \
        '  predict)' \
        '    shift' \
        '    exec /usr/local/bin/splicetransformer.py "$@"' \
        '    ;;' \
        '  derive)' \
        '    shift' \
        '    exec /usr/local/bin/splicetransformer-derive "$@"' \
        '    ;;' \
        '  "")' \
        '    echo "Usage: splicetransformer {chunk|predict|derive} [args...]" >&2' \
        '    exit 2' \
        '    ;;' \
        '  *)' \
        '    echo "Unknown subcommand: $subcommand" >&2' \
        '    echo "Usage: splicetransformer {chunk|predict|derive} [args...]" >&2' \
        '    exit 2' \
        '    ;;' \
        'esac' \
        > /usr/local/bin/splicetransformer \
    && chmod 0755 /usr/local/bin/splicetransformer \
    && splicetransformer-chunk --help >/dev/null \
    && splicetransformer.py --help >/dev/null \
    && splicetransformer chunk --help >/dev/null \
    && splicetransformer predict --help >/dev/null \
    && splicetransformer derive --help >/dev/null
