# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.
#
# This Dockerfile is Hiller Lab code (Apache-2.0). Bundled OrionGeno source
# and /opt/oriongeno/LICENSE remain under the OrionGeno Non-Commercial License.
# Checkpoint weights are CC-BY-NC-4.0. OrionGeno is free for academic and
# non-profit research; commercial use requires a separate license from BGI.

# Pin the linux/amd64 manifest, not the multi-arch index or a floating tag.
# The torch 2.7.0+cu126 wheel already ships the CUDA 12.6 runtime, so this
# stays on the slim Python image instead of an NVIDIA CUDA base.
FROM python:3.10.18-slim-bookworm@sha256:b4d66d07136c546f1765eae2bfcce9a64fa95f37c717c02bedd06d0476d1dbbd

ARG MODULE_VERSION=2026.7.10
# Default subset: mammals. Pass ORIONGENO_MODELS=all to bundle every lineage
# in models.tsv, or a comma-separated list such as mammals,birds,fish.
ARG ORIONGENO_MODELS=mammals
LABEL module.version="${MODULE_VERSION}" \
      org.opencontainers.image.title="oriongeno" \
      org.opencontainers.image.description="OrionGeno ab initio gene annotation. CUDA image; inference requires an NVIDIA GPU (no CPU fallback). Default checkpoint: mammals. Bundled OrionGeno is non-commercial." \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-OrionGeno-Non-Commercial AND CC-BY-NC-4.0" \
      org.opencontainers.image.source="https://github.com/hillerlab/containers" \
      org.opencontainers.image.url="https://github.com/hillerlab/containers" \
      org.opencontainers.image.vendor="The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung"
RUN echo "Building version ${MODULE_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/opt/oriongeno \
    ORIONGENO_HOME=/opt/oriongeno \
    ORIONGENO_VERSION=${MODULE_VERSION} \
    ORIONGENO_CHECKPOINT=/opt/oriongeno/checkpoints/oriongeno_mammals \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRITON_CACHE_DIR=/tmp/triton \
    TORCH_HOME=/tmp/torch \
    XDG_CACHE_HOME=/tmp

WORKDIR /work

# Triton JIT-compiles a CUDA wrapper on first GPU use and needs libc headers,
# not just gcc. libgomp1 is the OpenMP runtime torch and numba load.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        procps \
        gcc \
        libc6-dev \
        make \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"

COPY modules/oriongeno/requirements.min.txt \
     modules/oriongeno/requirements.native-cu126.txt \
     /tmp/oriongeno-requirements/

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
        -r /tmp/oriongeno-requirements/requirements.min.txt \
    && python -m pip install --no-cache-dir --no-deps \
        -r /tmp/oriongeno-requirements/requirements.native-cu126.txt \
    && rm -rf /tmp/oriongeno-requirements

COPY images/oriongeno/models.tsv /opt/oriongeno/models.tsv
COPY images/oriongeno/install_models.py /tmp/install_models.py

RUN python /tmp/install_models.py \
        --tsv /opt/oriongeno/models.tsv \
        --dest /opt/oriongeno/checkpoints \
        --models "${ORIONGENO_MODELS}" \
        --require /opt/oriongeno/checkpoints/oriongeno_mammals \
    && rm -f /tmp/install_models.py

COPY modules/oriongeno/main.py \
     modules/oriongeno/LICENSE \
     /opt/oriongeno/
COPY modules/oriongeno/oriongeno /opt/oriongeno/oriongeno

RUN printf '%s\n' \
        '#!/bin/sh' \
        'set -eu' \
        '' \
        'ORIONGENO_HOME="${ORIONGENO_HOME:-/opt/oriongeno}"' \
        'ORIONGENO_VERSION="${ORIONGENO_VERSION:-unknown}"' \
        'export PYTHONPATH="${ORIONGENO_HOME}${PYTHONPATH:+:$PYTHONPATH}"' \
        '' \
        'subcommand="${1:-}"' \
        '' \
        'case "$subcommand" in' \
        '  --version|-V)' \
        '    printf "oriongeno %s\n" "$ORIONGENO_VERSION"' \
        '    exit 0' \
        '    ;;' \
        '  ""|-h|--help)' \
        '    exec python "$ORIONGENO_HOME/main.py" --help' \
        '    ;;' \
        '  *)' \
        '    exec python "$ORIONGENO_HOME/main.py" "$@"' \
        '    ;;' \
        'esac' \
        > /usr/local/bin/oriongeno \
    && chmod 0755 /usr/local/bin/oriongeno

# Importing mamba_ssm or causal_conv1d initializes Triton, which needs an
# NVIDIA driver. Image builds have none, so check the wheels are present
# instead of importing them.
RUN python - <<'PY'
import importlib.metadata
import pathlib
import sys

assert sys.version_info[:2] == (3, 10), sys.version

import torch
import numpy
import Bio
import einops
import transformers
import numba

print(sys.version)
print(torch.__version__)
print(torch.version.cuda)
assert torch.__version__.startswith("2.7.0"), torch.__version__
assert str(torch.version.cuda).startswith("12.6"), torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI is True

for dist in ("mamba-ssm", "causal-conv1d"):
    version = importlib.metadata.version(dist)
    print(dist, version)
assert importlib.metadata.version("mamba-ssm").startswith("2.2.5"), importlib.metadata.version("mamba-ssm")
assert importlib.metadata.version("causal-conv1d").startswith("1.5.2"), importlib.metadata.version("causal-conv1d")

site = pathlib.Path("/usr/local/lib/python3.10/site-packages")
native = [
    site / "selective_scan_cuda.cpython-310-x86_64-linux-gnu.so",
    site / "causal_conv1d_cuda.cpython-310-x86_64-linux-gnu.so",
]
for path in native:
    assert path.is_file() and path.stat().st_size > 0, path
    print(path.name, path.stat().st_size)
PY

RUN test -s /opt/oriongeno/LICENSE \
    && test -s /opt/oriongeno/oriongeno/model_packages/species_embeddings_pca_formatted_numpy_float32.pkl \
    && test -s /opt/oriongeno/checkpoints/oriongeno_mammals/pytorch_model.bin \
    && test -s /opt/oriongeno/checkpoints/oriongeno_mammals/config.json \
    && oriongeno --version \
    && oriongeno --help >/dev/null \
    && oriongeno multi --help >/dev/null

CMD ["/bin/bash"]
