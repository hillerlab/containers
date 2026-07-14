//! `longread` — a BED12-native transcript inventory and expression engine for simulating
//! PacBio Iso-Seq data (Part 1 of the `longread` pipeline; see `CLAUDE.md`).
//!
//! The engine reads a BED12 annotation plus a transcript→gene mapping, generates synthetic
//! splice/truncation isoforms and two-gene fusions, and assigns exact integer molecule counts at
//! the gene and isoform level. All randomness is deterministic in `(seed, gene_id)` so results
//! are byte-identical regardless of thread count.

#![forbid(unsafe_code)]

pub mod error;
pub mod events;
pub mod expression;
pub mod fusion;
pub mod generate;
pub mod io;
pub mod model;
pub mod pacbio;
pub mod pbsim;
pub mod prepare;
pub mod report;
pub mod rng;
pub mod split;
pub mod validate;

pub use error::{Error, Result};
