# syntax=docker/dockerfile:1

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

FROM rust:1-slim-bookworm AS build
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

WORKDIR /src

# gcc: the `libdeflate-sys` crate (fast bgzf codec used by the PacBio BAM subcommands)
# compiles bundled C sources with the `cc` crate.
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config gcc \
    && rm -rf /var/lib/apt/lists/*

COPY modules/longread-rs/Cargo.toml modules/longread-rs/Cargo.lock ./
COPY modules/longread-rs/src ./src

RUN cargo build --release --locked && strip target/release/longread

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/target/release/longread /usr/local/bin/longread
USER 1000:1000

CMD ["bash"]
