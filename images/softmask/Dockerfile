# syntax=docker/dockerfile:1.7

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

# ---------- Build Stage - Rust ----------------
FROM rust:1.93-bookworm AS rust-builder
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo install --locked --root /opt/rust-tools --version 0.0.5 fxsplit

# ---------- Build Stage - Conda ----------------
FROM mambaorg/micromamba:1.5.1 AS conda-env
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba install -y -n base -c conda-forge -c bioconda \
        gawk=5.3.0 \
        repeatmasker=4.1.5 \
        repeatmodeler=2.0.5 \
        ucsc-fatotwobit=482 \
        ucsc-twobittofa=482 \
        python=3.11 \
    && micromamba clean --all --yes

# ---------- Runtime Stage ---------------------
FROM mambaorg/micromamba:1.5.1 AS runtime
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

COPY --from=conda-env /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

COPY --from=rust-builder /opt/rust-tools/bin/fxsplit /usr/local/bin/fxsplit
COPY --chmod=0755 modules/softmask/src/bin/group_mask_chunks.py modules/softmask/src/bin/renameheaders.py modules/softmask/src/bin/merge_repeatmasker_repeats.sh /usr/local/bin/

RUN fxsplit --version \
    && RepeatMasker -v 

WORKDIR /work

CMD ["/bin/bash"]
