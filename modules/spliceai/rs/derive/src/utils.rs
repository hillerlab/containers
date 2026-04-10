// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gessellschaft für Naturforschung
// Distributed under the terms of the Apache License, Version 2.0.

use std::fmt::Display;

/// Returns the complement of a DNA base.
///
/// # Arguments
/// * `base` - ASCII byte of the base (A, C, G, T, N)
///
/// # Example
/// ```rust,ignore
/// assert_eq!(complement_base(b'A'), b'T');
/// assert_eq!(complement_base(b'c'), b'G');
/// ```
pub fn complement_base(base: u8) -> u8 {
    match base.to_ascii_uppercase() {
        b'A' => b'T',
        b'C' => b'G',
        b'G' => b'C',
        b'T' => b'A',
        b'N' => b'N',
        other => other,
    }
}

/// Replaces each base in-place with its complement.
///
/// # Arguments
/// * `seq` - Mutable byte slice to modify
///
/// # Example
/// ```rust,ignore
/// let mut seq = b"ATGC".to_vec();
/// complement_in_place(&mut seq);
/// assert_eq!(&seq, b"TACG");
/// ```
pub fn complement_in_place(seq: &mut [u8]) {
    seq.iter_mut().for_each(|base| {
        *base = complement_base(*base);
    });
}

/// Reverses a sequence and replaces each base with its complement in-place.
///
/// # Arguments
/// * `seq` - Mutable byte vector to modify
///
/// # Example
/// ```rust,ignore
/// let mut seq = b"ATGC".to_vec();
/// reverse_complement_in_place(&mut seq);
/// assert_eq!(&seq, b"GCAT");
/// ```
pub fn reverse_complement_in_place(seq: &mut Vec<u8>) {
    seq.reverse();
    complement_in_place(seq);
}

/// Returns the reverse complement of a dinucleotide (2bp).
///
/// Does not modify the original, returns a new Vec.
///
/// # Arguments
/// * `pair` - 2-byte slice representing dinucleotide
///
/// # Example
/// ```rust,ignore
/// assert_eq!(reverse_complement_dinucleotide(b"AC"), b"GT".to_vec());
/// assert_eq!(reverse_complement_dinucleotide(b"GT"), b"AC".to_vec());
/// ```
pub fn reverse_complement_dinucleotide(pair: &[u8]) -> Vec<u8> {
    vec![complement_base(pair[1]), complement_base(pair[0])]
}

/// Converts ASCII letters to uppercase in-place.
///
/// # Arguments
/// * `seq` - Mutable byte slice to modify
///
/// # Example
/// ```rust,ignore
/// let mut seq = b"atgc".to_vec();
/// uppercase_in_place(&mut seq);
/// assert_eq!(&seq, b"ATGC");
/// ```
pub fn uppercase_in_place(seq: &mut [u8]) {
    seq.iter_mut().for_each(|base| {
        *base = base.to_ascii_uppercase();
    });
}

/// Builds a coordinate key string: chr:position(strand).
///
/// # Arguments
/// * `chr` - Chromosome name
/// * `position` - Genomic position
/// * `strand` - Strand (+ or -)
///
/// # Example
/// ```rust,ignore
/// let key = build_coordinate_key("chr1", 1000, '+');
/// assert_eq!(key, b"chr1:1000(+)".to_vec());
/// ```
pub fn build_coordinate_key<C, S>(chr: C, position: usize, strand: S) -> Vec<u8>
where
    C: AsRef<[u8]>,
    S: Display,
{
    let mut key = chr.as_ref().to_vec();
    key.push(b':');
    key.extend(position.to_string().as_bytes());
    key.push(b'(');
    key.extend(strand.to_string().as_bytes());
    key.push(b')');
    key
}
