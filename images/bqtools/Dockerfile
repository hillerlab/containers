# syntax=docker/dockerfile:1

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

FROM rust:1.93-bookworm AS builder
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

# rust-htslib's build.rs runs bindgen -> needs libclang (rust:bookworm has gcc
# but not clang); zlib-ng builds via the cmake crate -> needs cmake. htslib and
# zlib-ng are vendored/static; no system libs needed.
# AVX2 enables htslib's SIMD (bgzf) paths; portable to any x86-64-v3 CPU.
# ponytail: no bqtools runtime CPU guard - v3 assumes AVX2 hosts (2013+)
ENV RUSTFLAGS="-C target-cpu=x86-64-v3"
RUN apt-get update && apt-get install --yes --no-install-recommends \
        clang \
        cmake \
    && rm -rf /var/lib/apt/lists/* \
    && cargo install bqtools \
        --version "${MODULE_VERSION}" \
        --root /opt/bqtools

FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        bash \
        procps \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/bqtools/bin/bqtools /usr/local/bin/bqtools
RUN bqtools --version

CMD ["/bin/bash"]