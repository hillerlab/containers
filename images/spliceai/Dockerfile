# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

# ---------- Build Stage - Rust ----------
FROM rust:1.93-bookworm AS rust-builder
WORKDIR /build

COPY modules/spliceai/rs/chunk/Cargo.toml modules/spliceai/rs/chunk/Cargo.lock ./spliceai/chunk/
COPY modules/spliceai/rs/chunk/src ./spliceai/chunk/src

COPY modules/spliceai/rs/derive/Cargo.toml modules/spliceai/rs/derive/Cargo.lock ./spliceai/derive/
COPY modules/spliceai/rs/derive/src ./spliceai/derive/src

RUN cargo build --release --locked --manifest-path ./spliceai/chunk/Cargo.toml
RUN cargo build --release --locked --manifest-path ./spliceai/derive/Cargo.toml

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
    "numpy<2" \
    "tensorflow-cpu==2.15.1" \
    "keras==2.15.0" \
    "spliceai==1.3.1"

COPY --from=rust-builder /build/spliceai/chunk/target/release/spliceai-chunk /usr/local/bin/spliceai-chunk
COPY --from=rust-builder /build/spliceai/derive/target/release/spliceai-derive /usr/local/bin/spliceai-derive
COPY modules/spliceai/py/spliceai.py /usr/local/bin/spliceai.py

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN chmod 0755 /usr/local/bin/spliceai.py \
    && printf '%s\n' \
        '#!/bin/sh' \
        'set -eu' \
        '' \
        'subcommand="${1:-}"' \
        '' \
        'case "$subcommand" in' \
        '  chunk)' \
        '    shift' \
        '    exec /usr/local/bin/spliceai-chunk "$@"' \
        '    ;;' \
        '  predict)' \
        '    shift' \
        '    exec /usr/local/bin/spliceai.py "$@"' \
        '    ;;' \
        '  derive)' \
        '    shift' \
        '    exec /usr/local/bin/spliceai-derive "$@"' \
        '    ;;' \
        '  "")' \
        '    echo "Usage: spliceai {chunk|predict} [args...]" >&2' \
        '    exit 2' \
        '    ;;' \
        '  *)' \
        '    echo "Unknown subcommand: $subcommand" >&2' \
        '    echo "Usage: spliceai {chunk|predict} [args...]" >&2' \
        '    exit 2' \
        '    ;;' \
        'esac' \
        > /usr/local/bin/spliceai \
    && chmod 0755 /usr/local/bin/spliceai \
    && python -c "import importlib.metadata as m; d=m.distribution('spliceai'); needed={f'spliceai/models/spliceai{i}.h5' for i in range(1, 6)}; have={str(p) for p in (d.files or [])}; missing=needed-have; assert not missing, missing" \
    && spliceai-chunk --help >/dev/null \
    && spliceai.py --help >/dev/null \
    && spliceai chunk --help >/dev/null \
    && spliceai predict --help >/dev/null \
    && spliceai derive --help >/dev/null
