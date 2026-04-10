# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

FROM debian:bookworm-slim
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sed \
        procps && \
    rm -rf /var/lib/apt/lists/*

CMD ["bash"]
