# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

# ---------- Build Stage - Micromamba ----------

FROM mambaorg/micromamba:1.5.8
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

WORKDIR /work

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY modules/chaincleaner/NetFilterNonNested.perl /usr/local/bin/NetFilterNonNested.perl

RUN chmod 0755 /usr/local/bin/NetFilterNonNested.perl \
    && ln -sf /usr/local/bin/NetFilterNonNested.perl /usr/local/bin/NetFilterNonNested

USER $MAMBA_USER

RUN micromamba install -y -n base -c conda-forge -c bioconda \
        ucsc-chaincleaner=455 \
        ucsc-chainnet=482 \
        ucsc-chainscore=455 \
        perl \
        procps-ng \
    && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH="${MAMBA_ROOT_PREFIX}/bin:${PATH}"

CMD ["/bin/bash"]




