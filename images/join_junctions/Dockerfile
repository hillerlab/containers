# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

# ---------- Build Stage - Python ----------
FROM python:3.11-slim-bookworm AS runtime
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

WORKDIR /work

COPY modules/join_junctions/join_junctions.py /usr/local/bin/join_junctions.py

RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN chmod 0755 /usr/local/bin/join_junctions.py && \
    ln -s /usr/local/bin/join_junctions.py /usr/local/bin/join_junctions \
    && join_junctions --help >/dev/null && \
    join_junctions --version >/dev/null

CMD ["bash"]
