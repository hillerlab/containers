# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

# ---------- Build Stage - Rust ----------------
FROM rust:1-slim-bookworm AS rustbuild
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    gcc \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY modules/longread/modules/longread/Cargo.toml modules/longread/modules/longread/Cargo.lock ./
COPY modules/longread/modules/longread/src ./src

RUN cargo build --release --locked && strip target/release/longread
RUN cargo install xloci --version 0.0.4 --root /opt/xloci --locked
RUN cargo install chromsize --version 0.0.34 --root /opt/chromsize --locked

# ---------- Build Stage - Mamba ----------------
FROM mambaorg/micromamba:1.5.8-bookworm-slim

RUN micromamba install -y -n base -c bioconda -c conda-forge \
        pbsim3=3.0.4 \
        pbccs=6.4.0 \
        isoseq=4.0.0 \
        lima samtools=1.21 \
        procps-ng \
    && micromamba clean --all --yes

ENV PATH=/opt/conda/bin:$PATH

COPY --from=rustbuild /src/target/release/longread /opt/conda/bin/longread
COPY --from=rustbuild /opt/xloci/bin/xloci /opt/conda/bin/xloci

COPY modules/longread/assets/pbsim_models /opt/pbsim_models
ENV PBSIM3_MODELS=/opt/pbsim_models

# create alias for pbsim3 to avoid conflict with pbsim
RUN ln -s /opt/conda/bin/pbsim /opt/conda/bin/pbsim3

RUN longread --help >/dev/null \
    && xloci --help >/dev/null \
    && chromsize --help >/dev/null

CMD ["bash"]
