//! Error types for the `longread` engine.

use thiserror::Error;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level error type.
#[derive(Debug, Error)]
pub enum Error {
    /// An I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A failure while reading a BED file through `genepred`.
    #[error("BED read error: {0}")]
    BedRead(String),

    /// A failure while writing a BED file through `genepred`.
    #[error("BED write error: {0}")]
    BedWrite(String),

    /// Input validation failed (one or more problems).
    #[error("input validation failed:\n{0}")]
    Validation(String),

    /// A malformed line in a tab-separated input file.
    #[error("{path}:{line}: {message}")]
    Parse {
        /// File in which the problem occurred.
        path: String,
        /// 1-based line number.
        line: usize,
        /// Human-readable description.
        message: String,
    },

    /// A logical/configuration error (mutually exclusive flags, etc.).
    #[error("{0}")]
    Config(String),

    /// A failure while normalizing or validating PacBio BAM files.
    #[error("{0}")]
    Pacbio(String),

    /// Serialization failure for `stats.json`.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

impl Error {
    /// Convenience constructor for a configuration error.
    pub fn config(msg: impl Into<String>) -> Self {
        Error::Config(msg.into())
    }

    /// Convenience constructor for a PacBio BAM processing error.
    pub fn pacbio(msg: impl Into<String>) -> Self {
        Error::Pacbio(msg.into())
    }

    /// Convenience constructor for a parse error.
    pub fn parse(path: impl Into<String>, line: usize, msg: impl Into<String>) -> Self {
        Error::Parse {
            path: path.into(),
            line,
            message: msg.into(),
        }
    }
}
