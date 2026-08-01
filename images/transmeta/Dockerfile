# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

# ---------- Build Stage ----------
FROM debian:bookworm-slim AS builder

ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

ARG TRANSMETA_REPO=https://github.com/yutingsdu/TransMeta.git
ARG TRANSMETA_REF=7f0c82020d4ac9f6202d80cd6699f12a89e5ffdd

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        procps \
        git \
        g++ \
        gcc \
        make \
        libboost-all-dev \
        libbamtools-dev \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/* && \
    git clone --depth 1 "${TRANSMETA_REPO}" /opt/transmeta-src && \
    git -C /opt/transmeta-src checkout "${TRANSMETA_REF}" && \
    make -C /opt/transmeta-src/src/samtools-0.1.18 lib CFLAGS="-g -Wall -fsigned-char -O2 -fPIC" && \
    make -C /opt/transmeta-src/src all BOOST_PATH=/usr BAMTOOLS_PATH=/usr && \
    rm -f /opt/transmeta-src/src/bin/temp

# ---------- Runtime Stage ----------
FROM debian:bookworm-slim

ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

COPY --from=builder /opt/transmeta-src/TransMeta /opt/transmeta/TransMeta
COPY --from=builder /opt/transmeta-src/TransMetaMerge /opt/transmeta/TransMetaMerge
COPY --from=builder /opt/transmeta-src/src/bin /opt/transmeta/src/bin
COPY images/transmeta/VERSION /opt/transmeta/VERSION

ENV PATH="/opt/transmeta:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libbamtools2.5.2 \
        libboost-filesystem1.74.0 \
        zlib1g \
        procps && \
    rm -rf /var/lib/apt/lists/* && \
    TransMeta --help > /tmp/transmeta.help 2>&1 || true && \
    grep -qi "TransMeta" /tmp/transmeta.help && \
    rm /tmp/transmeta.help

WORKDIR /data

CMD ["bash"]
