<p align="center">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="../figures/hillerlab-dark.png"
    >
    <source
      media="(prefers-color-scheme: light)"
      srcset="../figures/hillerlab-light.png"
    >
    <img
      width="200"
      alt="Hiller Lab"
      src="../figures/hillerlab-light.png"
    >
  </picture>
</p>

  <span>
    <h1 align="center">
        CONTAINER CATALOG
    </h1>
  </span>

  <p align="center">
    <a href="https://github.com/hillerlab/containers" target="_blank">
      <img alt="GitHub License" src="https://img.shields.io/github/license/hillerlab/containers?color=blue">
    </a>
  </p>

  <p align="center">
    <samp>
        <span> The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung </span>
        <br>
        <br>
        <a href="https://github.com/hillerlab/containers/blob/master./catalog/catalog.md">catalog</a> .
        <a href="https://en.wikipedia.org/wiki/Containerization_(computing)">container</a> .
        <a href="https://hillerlab.com/">us</a> 
    </samp>
  </p>

</p>

# Image catalog

All container images are published to `ghcr.io/hillerlab/<name>`.

## Alignment & genome comparison

- [pylastz](https://github.com/hillerlab/containers/pkgs/container/pylastz) — Python-driven lastz aligner with UCSC twoBit/axt toolchain
- [make_lastz_chains](https://github.com/hillerlab/containers/pkgs/container/make_lastz_chains) — pipeline generating pairwise genome alignment chains from lastz
- [cesar2](https://github.com/hillerlab/containers/pkgs/container/cesar2) — realign coding exons/genes to DNA using a hidden Markov model
- [chainc](https://github.com/hillerlab/containers/pkgs/container/chainc) — remove chain-breaking alignments using chain/net files
- [chaincleaner](https://github.com/hillerlab/containers/pkgs/container/chaincleaner) — UCSC chainCleaner/chainNet/chainScore/chainSort chain cleanup toolkit
- [repeat_filler](https://github.com/hillerlab/containers/pkgs/container/repeat_filler) — fill gaps in chain alignments via local lastz realignment
- [macse2](https://github.com/hillerlab/containers/pkgs/container/macse2) — align coding sequences accounting for frameshifts and stop codons
- [prank](https://github.com/hillerlab/containers/pkgs/container/prank) — phylogeny-aware multiple sequence alignment

## RNA-seq & transcriptomics

- [rustar-aligner-cbq](https://github.com/hillerlab/containers/pkgs/container/rustar-aligner-cbq) — Rust reimplementation of the STAR RNA-seq aligner with CBQ input
- [spliceai](https://github.com/hillerlab/containers/pkgs/container/spliceai) — deep-learning splice-site prediction for RNA-seq
- [splicetransformer](https://github.com/hillerlab/containers/pkgs/container/splicetransformer) — deep-learning splice-site prediction with SpliceTransformer (tissue-specific, genome-wide)
- [intronic](https://github.com/hillerlab/containers/pkgs/container/intronic) — classify U2- vs U12-type introns using an SVM
- [join_junctions](https://github.com/hillerlab/containers/pkgs/container/join_junctions) — merge splice-junction calls across files into consensus junctions
- [transmeta](https://github.com/hillerlab/containers/pkgs/container/transmeta) — multi-sample RNA-seq transcript assembler
- [beaver](https://github.com/hillerlab/containers/pkgs/container/beaver) — single-cell RNA-seq transcript assembly merged across cells
- [aparent](https://github.com/hillerlab/containers/pkgs/container/aparent) — estimate poly(A) tail length via the APARENT deep-learning model
- [desalt](https://github.com/hillerlab/containers/pkgs/container/desalt) — de Bruijn graph-based spliced aligner for long transcriptome reads

## Long-read sequencing

- [pbsim3](https://github.com/hillerlab/containers/pkgs/container/pbsim3) — simulate PacBio/ONT reads using coverage and error models
- [longread](https://github.com/hillerlab/containers/pkgs/container/longread) — simulate PacBio Iso-Seq reads at scale (pipeline)
- [longread-rs](https://github.com/hillerlab/containers/pkgs/container/longread-rs) — transcript inventory and expression engine behind the longread pipeline

## Quality control & coverage

- [bqc](https://github.com/hillerlab/containers/pkgs/container/bqc) — CBQ-native all-in-one sequencing quality control tool
- [pandepth](https://github.com/hillerlab/containers/pkgs/container/pandepth) — ultrafast sequencing depth/coverage calculation

## Sequence filtering & depletion

- [deacon-cbq](https://github.com/hillerlab/containers/pkgs/container/deacon-cbq) — fast DNA search and host depletion using minimizers with CBQ input

## Repeat masking & annotation

- [softmask](https://github.com/hillerlab/containers/pkgs/container/softmask) — lightweight RepeatMasker-based soft-masking workflow
- [psauron](https://github.com/hillerlab/containers/pkgs/container/psauron) — machine-learning assessment of protein-coding gene annotation
- [annevo](https://github.com/hillerlab/containers/pkgs/container/annevo) — ANNEVO ab initio gene annotation (CUDA image, CPU fallback). Bundled ANNEVO is **non-commercial** (academic/non-profit research only).

## Format & data handling

- [bqtools](https://github.com/hillerlab/containers/pkgs/container/bqtools) — CLI for interacting with BINSEQ file formats

## Development & shell utilities

- [git](https://github.com/hillerlab/containers/pkgs/container/git) — distributed version-control system
- [sed](https://github.com/hillerlab/containers/pkgs/container/sed) — GNU stream editor
- [choose](https://github.com/hillerlab/containers/pkgs/container/choose) — fast column selection (cut/awk alternative)
- [rsync_ssh](https://github.com/hillerlab/containers/pkgs/container/rsync_ssh) — rsync + OpenSSH client for secure file transfer
