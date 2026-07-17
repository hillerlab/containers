# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

FROM mambaorg/micromamba:1.5.8-bookworm-slim
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

WORKDIR /work

USER root
USER $MAMBA_USER

# samtools is bundled so the module can normalise PBSIM3's SAM/BAM output to BAM in one step.
RUN micromamba install -y -n base -c bioconda -c conda-forge \
        pbsim3=${MODULE_VERSION} samtools=1.21 procps-ng \
    && micromamba clean --all --yes

ENV PATH=/opt/conda/bin:$PATH
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH="${MAMBA_ROOT_PREFIX}/bin:${PATH}"

# create alias for pbsim3 to avoid conflict with pbsim
RUN ln -s /opt/conda/bin/pbsim /opt/conda/bin/pbsim3

CMD ["/bin/bash"]

