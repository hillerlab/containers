# syntax=docker/dockerfile:1.7
# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.
FROM debian:bookworm-slim AS builder
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"
COPY images/hmmcleaner/VERSION /tmp/hmmcleaner.VERSION
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        perl \
        cpanminus \
        libssl-dev \
        zlib1g-dev \
        libxml2-dev \
        libexpat1-dev \
        libgsl-dev && \
    rm -rf /var/lib/apt/lists/* && \
    HMMCLEANER_VERSION="$(tr -d '[:space:]' < /tmp/hmmcleaner.VERSION)" && \
    case "${HMMCLEANER_VERSION}" in ""|*[!0-9.]*) echo "Invalid Bio::MUST::Apps::HmmCleaner version: ${HMMCLEANER_VERSION}" >&2; exit 1 ;; esac && \
    cpanm --local-lib-contained=/opt/hmmcleaner/perl5 --notest \
        Test2::Suite \
        Test::Files \
        Moose \
        Modern::Perl \
        Getopt::Euclid \
        IPC::System::Simple \
        List::AllUtils \
        Path::Class \
        Smart::Comments \
        Template \
        Test::Most \
        aliased \
        namespace::autoclean \
        Bio::FastParsers::Hmmer \
        Bio::MUST::Core \
        Bio::MUST::Drivers \
        "Bio::MUST::Apps::HmmCleaner@${HMMCLEANER_VERSION}"

FROM debian:bookworm-slim
COPY --from=builder /opt/hmmcleaner/perl5 /opt/hmmcleaner/perl5
COPY images/hmmcleaner/VERSION /opt/hmmcleaner/VERSION
ENV PERL5LIB="/opt/hmmcleaner/perl5/lib/perl5"
ENV PATH="/opt/hmmcleaner/perl5/bin:${PATH}"
# libxml2 / libexpat1 / libgsl27 + libgslcblas0 are the runtime counterparts
# of the -dev headers pulled in above to compile XS modules. Verified the
# -dev -> runtime package mapping against Ubuntu 24.04's apt metadata; bookworm
# is very likely identical but wasn't checked directly, so confirm with
# `apt-cache depends libgsl-dev` in your own build environment. The more
# reliable check is `ldd` on the .so files under /opt/hmmcleaner/perl5 after
# a build, to see exactly what's actually linked rather than guessing from
# the -dev package names.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        perl \
        hmmer \
        libxml2 \
        libexpat1 \
        libgsl27 \
        libgslcblas0 \
        procps && \
    rm -rf /var/lib/apt/lists/* && \
    (HmmCleaner.pl --help > /tmp/hmmcleaner.help 2>&1 || true) && \
    grep -qi "hmmcleaner" /tmp/hmmcleaner.help && \
    rm /tmp/hmmcleaner.help
WORKDIR /data
CMD ["bash"]