# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

# ---------- Build Stage - Rust ----------
FROM rust:1.93-bookworm AS rust-builder
WORKDIR /build

COPY modules/aparent/rs/Cargo.toml modules/aparent/rs/Cargo.lock ./aparent/
COPY modules/aparent/rs/src ./aparent/src

RUN cargo build --release --locked --manifest-path ./aparent/Cargo.toml

# ---------- Build Stage - Python ----------
FROM python:3.11-slim-bookworm AS runtime
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /work

RUN pip install --no-cache-dir \
    "numpy >= 2" \
    "scipy >= 1.2.1" \
    "tensorflow >= 1.13.1" \
    "keras >= 2.2.4" 

COPY --from=rust-builder /build/aparent/target/release/aparent /usr/local/bin/aparent-chunk
COPY modules/aparent/py/aparent.py /usr/local/bin/aparent.py

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN chmod 0755 /usr/local/bin/aparent.py \
    && printf '%s\n' \
        '#!/bin/sh' \
        'set -eu' \
        '' \
        'subcommand="${1:-}"' \
        '' \
        'case "$subcommand" in' \
        '  chunk)' \
        '    shift' \
        '    exec /usr/local/bin/aparent-chunk "$@"' \
        '    ;;' \
        '  predict)' \
        '    shift' \
        '    exec /usr/local/bin/aparent.py "$@"' \
        '    ;;' \
        '  "")' \
        '    echo "Usage: aparent {chunk|predict} [args...]" >&2' \
        '    exit 2' \
        '    ;;' \
        '  *)' \
        '    echo "Unknown subcommand: $subcommand" >&2' \
        '    echo "Usage: aparent {chunk|predict} [args...]" >&2' \
        '    exit 2' \
        '    ;;' \
        'esac' \
        > /usr/local/bin/aparent \
    && chmod 0755 /usr/local/bin/aparent \
    && aparent chunk --help >/dev/null \
    && aparent predict --help >/dev/null
