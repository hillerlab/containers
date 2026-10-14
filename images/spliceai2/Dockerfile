# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.
# This Dockerfile is Hiller Lab code (Apache-2.0). The installed Illumina SpliceAI2 2.0 package
# is subject to the SpliceAI2 Model Terms of Use, which restrict redistribution. Model weights
# are not included; mount them at /models.

# ---------- Build Stage - Rust ----------
FROM rust:1.93-bookworm AS rust-builder
WORKDIR /build

COPY modules/spliceai/rs/chunk/Cargo.toml modules/spliceai/rs/chunk/Cargo.lock ./spliceai/chunk/
COPY modules/spliceai/rs/chunk/src ./spliceai/chunk/src
RUN cargo build --release --locked --manifest-path ./spliceai/chunk/Cargo.toml

# ---------- Build Stage - Python ----------
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS runtime
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}" \
      org.opencontainers.image.title="SpliceAI2" \
      org.opencontainers.image.description="SpliceAI2 splice-site prediction and transcript decoding. Weights are not included and must be mounted at /models." \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-SpliceAI2-Model-Terms" \
      org.opencontainers.image.source="https://github.com/hillerlab/containers" \
      org.opencontainers.image.url="https://github.com/hillerlab/containers" \
      org.opencontainers.image.vendor="The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung"
RUN echo "Building version ${MODULE_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/usr/local/bin:/opt/conda/bin:${PATH}"

WORKDIR /work

RUN python -m pip install --no-cache-dir \
    "spliceai2==2.0" \
    "lightning==2.6.1" \
    "pytorch-lightning==2.6.1" \
    "numpy==2.4.3" \
    "pandas==3.0.1" \
    "pyfaidx==0.9.0.3" \
    "huggingface_hub==0.36.2" \
    && python -c "import lightning, pytorch_lightning, torch; assert (torch.__version__.startswith('2.6.0') and torch.version.cuda == '12.4' and lightning.__version__ == '2.6.1' and pytorch_lightning.__version__ == '2.6.1')"

COPY --from=rust-builder /build/spliceai/chunk/target/release/spliceai-chunk /usr/local/bin/spliceai-chunk
COPY modules/spliceai2/py/spliceai2.py /usr/local/libexec/spliceai2-runner.py

RUN chmod 0755 /usr/local/libexec/spliceai2-runner.py \
    && printf '%s\n' \
        '#!/bin/sh' \
        'set -eu' \
        'subcommand="${1:-}"' \
        'case "$subcommand" in' \
        '  chunk) shift; exec /usr/local/bin/spliceai-chunk "$@" ;;' \
        '  predict|transcripts|fetch-models) exec /usr/local/libexec/spliceai2-runner.py "$@" ;;' \
        '  --help|-h) exec /usr/local/libexec/spliceai2-runner.py --help ;;' \
        '  --version|-v) exec /usr/local/libexec/spliceai2-runner.py --version ;;' \
        '  *)' \
        '    echo "Usage: spliceai2 {chunk|predict|transcripts|fetch-models} [args...]" >&2' \
        '    exit 2 ;;' \
        'esac' \
        > /usr/local/bin/spliceai2 \
    && chmod 0755 /usr/local/bin/spliceai2 \
    && spliceai2 chunk --help >/dev/null \
    && spliceai2 predict --help >/dev/null \
    && spliceai2 transcripts --help >/dev/null \
    && spliceai2 fetch-models --help >/dev/null \
    && spliceai2 --version >/dev/null

VOLUME /models
ENTRYPOINT ["spliceai2"]
