# syntax=docker/dockerfile:1.7

# ── Build stage ─────────────────────────────────────────────────────────────
FROM debian:12-slim AS builder

ARG BEAVER_REPO=https://github.com/alejandrogzi/beaver.git
ARG BEAVER_REF=master

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        autoconf \
        automake \
        ca-certificates \
        g++ \
        gcc \
        git \
        libboost-all-dev \
        libhts-dev \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 --branch "${BEAVER_REF}" "${BEAVER_REPO}" beaver-src

WORKDIR /opt/beaver-src
RUN aclocal \
    && autoconf \
    && autoheader \
    && automake -a \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && install -m 0755 src/beaver /usr/local/bin/beaver

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM debian:12-slim AS runtime

ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

COPY --from=builder /usr/local/bin/beaver /usr/local/bin/beaver

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libhts3 \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN beaver
WORKDIR /data
