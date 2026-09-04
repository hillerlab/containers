# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.
#
# This Dockerfile is Hiller Lab code (Apache-2.0). Bundled Tiberius, bricks2marble,
# and hidten remain under the MIT license.

# NGC TensorFlow 25.02 is the last NVIDIA TF image (Python 3.12, TF 2.17).
# Pin the linux/amd64 manifest, not the multi-arch list or a floating tag.
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3@sha256:d78d5b31b7e6cd9c902e98f73a977fcdce9d47d0bc1c685b3fa94f4187e6301c

USER root

ARG MODULE_VERSION=2.0.7
# Default subset: Mammalia both masking variants + Vertebrates.
# Pass TIBERIUS_MODELS=all to bundle every alias in models.tsv.
ARG TIBERIUS_MODELS=mammalia_nosoftmasking_v2,mammalia_softmasking_v2,vertebrates
LABEL module.version="${MODULE_VERSION}" \
      org.opencontainers.image.title="tiberius" \
      org.opencontainers.image.description="Tiberius ab initio gene prediction. CUDA-capable image that falls back to CPU when no GPU is exposed." \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.opencontainers.image.source="https://github.com/hillerlab/containers" \
      org.opencontainers.image.url="https://github.com/hillerlab/containers" \
      org.opencontainers.image.vendor="The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung"
RUN echo "Building version ${MODULE_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive \
    TF_USE_LEGACY_KERAS=0 \
    PYTHONPATH=/opt/tiberius \
    PATH="/opt/tiberius:/opt/tiberius/tiberius/scripts:${PATH}" \
    TIBERIUS_HOME=/opt/tiberius \
    TIBERIUS_VERSION=${MODULE_VERSION}

WORKDIR /work

RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Do not pip install tiberius[from_source]: that extra pulls PyPI
# tensorflow[and-cuda] and can replace NVIDIA's TF 2.17.
COPY modules/tiberius/pyproject.toml \
     modules/tiberius/README.md \
     modules/tiberius/LICENSE \
     modules/tiberius/tiberius.py \
     /opt/tiberius/
COPY modules/tiberius/tiberius /opt/tiberius/tiberius
COPY modules/tiberius/model_cfg /opt/tiberius/model_cfg
COPY images/tiberius/models.tsv /opt/tiberius/models.tsv
COPY images/tiberius/merge.py /usr/local/bin/tiberius-merge
COPY images/tiberius/install_models.py /tmp/install_models.py

RUN chmod 0755 /opt/tiberius/tiberius.py \
        /opt/tiberius/tiberius/scripts/*.py \
        /usr/local/bin/tiberius-merge \
        /tmp/install_models.py \
    && ln -sf /opt/tiberius/tiberius.py /usr/local/bin/tiberius.py \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        hatchling \
        "keras>=3,<4" \
        numpy \
        pydantic \
        biopython \
        requests \
        "packaging>=23.0" \
        pandas \
        pyyaml \
        rich \
    && python3 -m pip install --no-cache-dir --no-deps \
        hidten \
        bricks2marble \
    && python3 -m pip install --no-cache-dir --no-deps -e /opt/tiberius \
    && python3 /tmp/install_models.py \
        --tsv /opt/tiberius/models.tsv \
        --dest /opt/tiberius/model_weights \
        --cfg-dir /opt/tiberius/model_cfg \
        --models "${TIBERIUS_MODELS}" \
    && rm -f /tmp/install_models.py \
    && python3 -c 'import sys; assert sys.version_info >= (3, 12), sys.version; import tensorflow as tf; import keras; import tiberius; import bricks2marble; import hidten; print(sys.version); print("tensorflow", tf.__version__); print("keras", keras.__version__); assert tf.__version__.startswith("2.17"), tf.__version__' \
    && tiberius.py --help >/dev/null \
    && tiberius.py --list_cfg \
    && tiberius-merge --help >/dev/null \
    && test -n "$(ls -A /opt/tiberius/model_weights)" \
    && test -s /opt/tiberius/model_cfg/mammalia_nosoftmasking_v2.yaml \
    && test -s /opt/tiberius/LICENSE

CMD ["/bin/bash"]
