# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

FROM debian:bookworm-slim AS downloader

ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

COPY images/prank/VERSION /tmp/prank.VERSION
COPY images/prank/SHA256SUMS /tmp/prank.SHA256SUMS

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl && \
    rm -rf /var/lib/apt/lists/* && \
    PRANK_VERSION="$(tr -d '[:space:]' < /tmp/prank.VERSION)" && \
    case "${PRANK_VERSION}" in ""|*[!0-9]*) echo "Invalid PRANK version: ${PRANK_VERSION}" >&2; exit 1 ;; esac && \
    PRANK_TARBALL="prank.linux64.${PRANK_VERSION}.tgz" && \
    if ! curl -fsSL "https://raw.githubusercontent.com/ariloytynoja/prank-msa/master/binaries/${PRANK_TARBALL}" \
        -o "/tmp/${PRANK_TARBALL}" 2>/dev/null; then \
        curl -fsSL "https://raw.githubusercontent.com/ariloytynoja/prank-msa/master/binaries/previous_version/${PRANK_TARBALL}" \
            -o "/tmp/${PRANK_TARBALL}"; \
    fi && \
    cd /tmp && \
    grep -F "  ${PRANK_TARBALL}" prank.SHA256SUMS > prank.tgz.sha256 || \
        { echo "Missing checksum for ${PRANK_TARBALL}" >&2; exit 1; } && \
    sha256sum -c prank.tgz.sha256 && \
    tar -xzf "/tmp/${PRANK_TARBALL}" -C /opt && \
    rm "/tmp/${PRANK_TARBALL}"

FROM debian:bookworm-slim

COPY --from=downloader /opt/prank /opt/prank
COPY images/prank/VERSION /opt/prank/VERSION

ENV PATH="/opt/prank/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libstdc++6 \
        procps && \
    rm -rf /var/lib/apt/lists/* && \
    PRANK_VERSION="$(tr -d '[:space:]' < /opt/prank/VERSION)" && \
    prank -help > /tmp/prank.help && \
    grep -q "prank v.${PRANK_VERSION}" /tmp/prank.help && \
    rm /tmp/prank.help

WORKDIR /data

CMD ["bash"]
