# syntax=docker/dockerfile:1.7
# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
# Distributed under the terms of the Apache License, Version 2.0.

FROM debian:bookworm-slim AS builder
ARG MODULE_VERSION
LABEL module.version="${MODULE_VERSION}"
RUN echo "Building version ${MODULE_VERSION}"

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
        Bio::MUST::Apps::HmmCleaner

FROM debian:bookworm-slim
COPY --from=builder /opt/hmmcleaner/perl5 /opt/hmmcleaner/perl5
ENV PERL5LIB="/opt/hmmcleaner/perl5/lib/perl5"
ENV PATH="/opt/hmmcleaner/perl5/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        perl \
        hmmer \
        procps && \
    rm -rf /var/lib/apt/lists/* && \
    (HmmCleaner.pl --help > /tmp/hmmcleaner.help 2>&1 || true) && \
    grep -qi "hmmcleaner" /tmp/hmmcleaner.help && \
    rm /tmp/hmmcleaner.help

WORKDIR /data
CMD ["bash"]