# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

# ---------- Build Stage - Python ----------
FROM python:3.11-slim-bookworm AS runtime
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

WORKDIR /work

COPY modules/repeat_filler/repeat_filler.py /usr/local/bin/repeat_filler.py

RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN chmod 0755 /usr/local/bin/repeat_filler.py && \
    ln -s /usr/local/bin/repeat_filler.py /usr/local/bin/repeat_filler \
    && repeat_filler --help >/dev/null && \
    repeat_filler --version >/dev/null

CMD ["bash"]
