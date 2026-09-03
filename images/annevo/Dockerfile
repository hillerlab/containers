# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.
#
# This Dockerfile is Hiller Lab code (Apache-2.0). Bundled ANNEVO source, models,
# and /opt/annevo/LICENSE remain under the ANNEVO Non-Commercial License.
# ANNEVO is free for academic and non-profit research; commercial use requires
# a separate license from the ANNEVO copyright holders.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG MODULE_VERSION=2.3.3
LABEL module.version="${MODULE_VERSION}" \
      org.opencontainers.image.title="annevo" \
      org.opencontainers.image.description="ANNEVO ab initio gene annotation. CUDA-capable image that falls back to CPU when no GPU is exposed. Bundled ANNEVO is non-commercial." \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-ANNEVO-Non-Commercial" \
      org.opencontainers.image.source="https://github.com/hillerlab/containers" \
      org.opencontainers.image.url="https://github.com/hillerlab/containers" \
      org.opencontainers.image.vendor="The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung"
RUN echo "Building version ${MODULE_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/opt/annevo \
    ANNEVO_HOME=/opt/annevo \
    ANNEVO_VERSION=${MODULE_VERSION}

WORKDIR /work

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python-is-python3 \
        procps \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.1.0 \
        torchvision==0.16.0 \
        torchaudio==2.1.0 \
    && python -m pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.3.3 \
        h5py==3.14.0 \
        numba==0.65.1 \
        torchmetrics==0.8.2 \
        bcbio-gff==0.7.1 \
        tqdm==4.67.1 \
        biopython

COPY modules/annevo/annotation.py \
     modules/annevo/prediction.py \
     modules/annevo/decoding.py \
     modules/annevo/LICENSE \
     /opt/annevo/
COPY modules/annevo/src /opt/annevo/src
COPY modules/annevo/model /opt/annevo/model
COPY modules/annevo/datamodule /opt/annevo/datamodule
COPY modules/annevo/saved_model /opt/annevo/saved_model

RUN printf '%s\n' \
        '#!/bin/sh' \
        'set -eu' \
        '' \
        "ANNEVO_HOME=\"\${ANNEVO_HOME:-/opt/annevo}\"" \
        "ANNEVO_VERSION=\"\${ANNEVO_VERSION:-${MODULE_VERSION}}\"" \
        "export PYTHONPATH=\"\${ANNEVO_HOME}\${PYTHONPATH:+:\$PYTHONPATH}\"" \
        '' \
        'subcommand="${1:-}"' \
        '' \
        'case "$subcommand" in' \
        '  annotation|prediction|decoding)' \
        '    shift' \
        '    exec python "$ANNEVO_HOME/${subcommand}.py" "$@"' \
        '    ;;' \
        '  --version|-V)' \
        '    printf "annevo %s\n" "$ANNEVO_VERSION"' \
        '    exit 0' \
        '    ;;' \
        '  -h|--help)' \
        '    echo "Usage: annevo {annotation|prediction|decoding} [args...]"' \
        '    echo "       annevo --version"' \
        '    exit 0' \
        '    ;;' \
        '  "")' \
        '    echo "Usage: annevo {annotation|prediction|decoding} [args...]" >&2' \
        '    echo "       annevo --version" >&2' \
        '    exit 2' \
        '    ;;' \
        '  *)' \
        '    echo "Unknown subcommand: $subcommand" >&2' \
        '    echo "Usage: annevo {annotation|prediction|decoding} [args...]" >&2' \
        '    exit 2' \
        '    ;;' \
        'esac' \
        > /usr/local/bin/annevo \
    && chmod 0755 /usr/local/bin/annevo \
    && for lineage in Mammalia Aves Insecta Fungi Magnoliopsida Actinopteri; do \
           test -s "/opt/annevo/saved_model/ANNEVO_${lineage}.pt" \
           || { echo "missing model: ANNEVO_${lineage}.pt" >&2; exit 1; }; \
       done \
    && test -s /opt/annevo/LICENSE

RUN python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 10), sys.version

import torch
import numpy
import pandas
import h5py
import numba
import Bio

print(sys.version)
print(torch.__version__)
print(torch.version.cuda)
assert torch.__version__.startswith("2.1.0"), torch.__version__
assert str(torch.version.cuda).startswith("12.1"), torch.version.cuda
PY

RUN annevo --version \
    && annevo annotation --help >/dev/null \
    && annevo prediction --help >/dev/null \
    && annevo decoding --help >/dev/null

CMD ["/bin/bash"]
