# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

# ---------- Build Stage - Micromamba ----------

FROM mambaorg/micromamba:1.5.8
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

RUN micromamba install -y -n base -c conda-forge -c bioconda \
        python=3.9 \
        lastz=1.0.4 \
        procps-ng \
    && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

CMD ["/bin/bash"]
