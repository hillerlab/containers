# syntax=docker/dockerfile:1.7
# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.
#
# Docker translation of the working Apptainer/Singularity def file for TOGA2.
FROM continuumio/anaconda3
ARG MODULE_VERSION
LABEL Author="SGN Hillerlab" module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"
COPY images/toga2/VERSION /tmp/toga2.VERSION

SHELL ["/bin/bash", "-c"]

ENV PATH="/root/.cargo/bin:/opt/nextflow:/opt/conda/bin:/opt/TOGA2:/opt/TOGA2/bin:/opt/TOGA2/bin/prank:/opt/TOGA2/bin/iqtree2:/opt/TOGA2/bin/intronIC:${PATH}"
ENV SHELL=/bin/bash

RUN apt-get update && apt-get install -y \
        curl \
        openjdk-17-jdk \
        git \
        bsd-mailx \
        mailutils \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# cargo / rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Nextflow
RUN mkdir -p /opt/nextflow && \
    curl -sSL https://get.nextflow.io | bash && \
    mv nextflow /opt/nextflow/nextflow && \
    chmod +x /opt/nextflow/nextflow

# avoid known conflicting python packages
RUN python3 -m pip uninstall -y gensim numba datashader && \
    python3 -m pip install --upgrade --root-user-action=ignore \
        streamlit \
        pyarrow \
        bottleneck

# make bash the default shell in the container
RUN sed -i 's/required/sufficient/g' /etc/pam.d/chsh && \
    rm /bin/sh && ln -f /bin/bash /bin/sh

# clone and build TOGA2, pinned to the release tag in the VERSION file
RUN TOGA2_VERSION="$(tr -d '[:space:]' < /tmp/toga2.VERSION)" && \
    case "${TOGA2_VERSION}" in ""|*[!0-9.a-zA-Z]*) echo "Invalid TOGA2 version: ${TOGA2_VERSION}" >&2; exit 1 ;; esac && \
    git clone --recurse-submodules --depth 1 --branch "v${TOGA2_VERSION}" \
        https://github.com/hillerlab/TOGA2 /opt/TOGA2
COPY images/toga2/VERSION /opt/TOGA2/VERSION
WORKDIR /opt/TOGA2
RUN python3 -m venv toga2 && \
    . toga2/bin/activate && \
    make

# Permanently "activate" the venv for every future RUN/ENTRYPOINT/CMD and for
# the running container.
ENV VIRTUAL_ENV=/opt/TOGA2/toga2
ENV PATH="/opt/TOGA2/toga2/bin:${PATH}"

# dummy user for Slurm, as in the original %runscript
RUN adduser --disabled-password --gecos "" slurm

RUN cat <<'EOF' > /usr/local/bin/toga2-entrypoint.sh
#!/bin/bash
set -e
echo "slurmadmin:x:300:300::/opt/slurm/slurm:/bin/false" >> /etc/passwd
echo "slurmadmin:x:300:" >> /etc/group

cd /opt/TOGA2
export SLURM_CONF="${SLURM_CONF}"
export NXF_HOME="${TMPDIR}/.nextflow"
export EXEC_DIR="${EXEC_DIR}"
export PATH="${EXEC_DIR}:/root/.cargo/bin:/opt/nextflow:/opt/conda/bin:/opt/TOGA2:/opt/TOGA2/bin:/opt/TOGA2/bin/prank:/opt/TOGA2/bin/iqtree2:/opt/TOGA2/bin/intronIC:${PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${LIB_DIR}"

exec "$@"
EOF
RUN chmod +x /usr/local/bin/toga2-entrypoint.sh

WORKDIR /opt/TOGA2
ENTRYPOINT ["/usr/local/bin/toga2-entrypoint.sh"]
CMD ["bash"]