# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

FROM debian:bookworm-slim AS downloader

COPY images/macse2/VERSION /tmp/macse2.VERSION
COPY images/macse2/SHA256SUMS /tmp/macse2.SHA256SUMS

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl && \
    rm -rf /var/lib/apt/lists/* && \
    MACSE2_VERSION="$(tr -d '[:space:]' < /tmp/macse2.VERSION)" && \
    case "${MACSE2_VERSION}" in ""|*[!0-9.]*) echo "Invalid MACSE2 version: ${MACSE2_VERSION}" >&2; exit 1 ;; esac && \
    MACSE2_JAR="macse_v${MACSE2_VERSION}.jar" && \
    curl -fsSL \
        "https://www.agap-ge2pop.org/wp-content/uploads/macse/releases/${MACSE2_JAR}" \
        -o "/tmp/${MACSE2_JAR}" && \
    cd /tmp && \
    grep -F "  ${MACSE2_JAR}" macse2.SHA256SUMS > macse2.jar.sha256 || \
        { echo "Missing checksum for ${MACSE2_JAR}" >&2; exit 1; } && \
    sha256sum -c macse2.jar.sha256 && \
    mkdir -p /opt/macse2 && \
    mv "/tmp/${MACSE2_JAR}" /opt/macse2/macse2.jar

FROM debian:bookworm-slim
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

COPY --from=downloader /opt/macse2/macse2.jar /opt/macse2/macse2.jar
COPY images/macse2/VERSION /opt/macse2/VERSION

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        procps && \
    rm -rf /var/lib/apt/lists/* && \
    printf '%s\n' \
        '#!/bin/sh' \
        'exec java -Djava.awt.headless=true -jar /opt/macse2/macse2.jar "$@"' \
        > /usr/local/bin/macse2 && \
    chmod 0755 /usr/local/bin/macse2 && \
    MACSE2_VERSION="$(tr -d '[:space:]' < /opt/macse2/VERSION)" && \
    macse2 -help > /tmp/macse2.help && \
    grep -q "This is MACSE V${MACSE2_VERSION}" /tmp/macse2.help && \
    rm /tmp/macse2.help

WORKDIR /data

CMD ["bash"]
