# syntax=docker/dockerfile:1

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

FROM rust:1.90-bookworm AS builder
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

RUN cargo install bqc \
    --version "${MODULE_VERSION}" \
    --locked \
    --root /opt/bqc

# Minimal runtime stage
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        bash \
        procps \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/bqc/bin/bqc /usr/local/bin/bqc
RUN bqc --version

CMD ["/bin/bash"]
