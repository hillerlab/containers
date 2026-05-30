# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

# ---------- Build Stage - Micromamba ----------
FROM mambaorg/micromamba:bookworm-slim AS runtime
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

WORKDIR /work

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY modules/repeat_filler/repeat_filler.py /usr/local/bin/repeat_filler.py

RUN chmod 0755 /usr/local/bin/repeat_filler.py \
    && ln -sf /usr/local/bin/repeat_filler.py /usr/local/bin/repeat_filler

USER $MAMBA_USER

RUN micromamba install -y -n base \
        -c conda-forge \
        -c bioconda \
        python=3.11 \
        lastz=1.0.4 \
        ucsc-axttopsl=482 \
        ucsc-axtchain=482 \
        ucsc-chainsort=482 \
        procps-ng \
    && micromamba clean --all --yes

ENV PATH="${MAMBA_ROOT_PREFIX}/bin:${PATH}"

RUN repeat_filler --help >/dev/null \
    && repeat_filler --version >/dev/null

CMD ["/bin/bash"]
