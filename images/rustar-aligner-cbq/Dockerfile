# syntax=docker/dockerfile:1

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

FROM rust:1.93-bookworm AS builder
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

# AVX2 for the SIMD-tuned seed-search/DP loops. Upstream's default is baseline
# generic x86-64 (portable, slower). x86-64-v3 is safe: check_cpu_compat only
# bails for v3/v4 binaries on non-AVX2 CPUs, and then with a friendly error.
ENV RUSTFLAGS="-C target-cpu=x86-64-v3"

RUN git clone --depth 1 --branch main \
        https://github.com/alejandrogzi/rustar-aligner /build/rustar \
    && cd /build/rustar \
    && cargo build --release --locked \
    && cp target/release/rustar-aligner /opt/rustar-aligner

# ponytail: --branch main is a moving branch; bump VERSION when the fork changes

FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        bash \
        procps \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/rustar-aligner /usr/local/bin/rustar-aligner
RUN ln -s rustar-aligner /usr/local/bin/STAR \
    && ln -s rustar-aligner /usr/local/bin/ruSTAR \
    && rustar-aligner --help >/dev/null

CMD ["/bin/bash"]