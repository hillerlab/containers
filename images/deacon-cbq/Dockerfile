# syntax=docker/dockerfile:1

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

FROM rust:1.93-bookworm AS builder
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

# Fork's default .cargo/config.toml builds with -C target-cpu=native, whose
# AVX-512 binaries abort via ensure_simd on older CPUs. Upstream's release
# pipeline swaps in config-portable.toml (x86-64-v3 / AVX2) — do the same.
RUN git clone --depth 1 --branch v0.0.16-cbq-input \
        https://github.com/alejandrogzi/deacon /build/deacon \
    && cp /build/deacon/.cargo/config-portable.toml /build/deacon/.cargo/config.toml \
    && cd /build/deacon \
    && cargo build --release --locked -p deacon \
    && cp /build/deacon/target/release/deacon /opt/deacon


FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        bash \
        procps \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/deacon /usr/local/bin/deacon
RUN deacon --version

CMD ["/bin/bash"]
