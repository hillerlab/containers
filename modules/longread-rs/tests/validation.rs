//! Validation fixtures (spec §7 "Rust unit fixtures").

use std::io::Write;

use longread::validate::{load_and_validate, ValidatedInput};
use tempfile::TempDir;

/// Write `bed`, `mapping`, and optional `sizes` into a temp dir and validate them.
fn validate(bed: &str, mapping: &str, sizes: Option<&str>) -> longread::Result<ValidatedInput> {
    let dir = TempDir::new().unwrap();
    let bed_path = dir.path().join("in.bed");
    let map_path = dir.path().join("map.tsv");
    std::fs::File::create(&bed_path)
        .unwrap()
        .write_all(bed.as_bytes())
        .unwrap();
    std::fs::File::create(&map_path)
        .unwrap()
        .write_all(mapping.as_bytes())
        .unwrap();
    let sizes_path = sizes.map(|s| {
        let p = dir.path().join("chrom.sizes");
        std::fs::File::create(&p)
            .unwrap()
            .write_all(s.as_bytes())
            .unwrap();
        p
    });
    load_and_validate(&bed_path, &map_path, sizes_path.as_ref(), 0)
}

#[test]
fn single_exon_plus_transcript() {
    let bed = "chr1\t100\t200\ttx1\t0\t+\t100\t200\t0,0,0\t1\t100\t0\n";
    let map = "tx1\tgeneA\n";
    let v = validate(bed, map, Some("chr1\t1000\n")).unwrap();
    assert_eq!(v.input_transcript_count, 1);
    assert_eq!(v.genes.len(), 1);
    assert_eq!(v.genes[0].transcripts[0].exons, vec![(100, 200)]);
}

#[test]
fn single_exon_minus_transcript() {
    let bed = "chr1\t100\t200\ttx1\t0\t-\t100\t200\t0,0,0\t1\t100\t0\n";
    let map = "tx1\tgeneA\n";
    let v = validate(bed, map, None).unwrap();
    assert_eq!(v.genes[0].strand, longread::model::Strand::Minus);
}

#[test]
fn three_exon_plus_and_minus() {
    let bed = "chr1\t100\t700\tp\t0\t+\t100\t700\t0,0,0\t3\t100,100,100\t0,300,500\n\
               chr2\t100\t700\tm\t0\t-\t100\t700\t0,0,0\t3\t100,100,100\t0,300,500\n";
    let map = "p\tgP\nm\tgM\n";
    let v = validate(bed, map, None).unwrap();
    assert_eq!(v.genes.len(), 2);
    for g in &v.genes {
        assert_eq!(g.transcripts[0].exon_count(), 3);
    }
}

#[test]
fn two_isoforms_of_one_gene() {
    let bed = "chr1\t100\t700\tiso1\t0\t+\t100\t700\t0,0,0\t3\t100,100,100\t0,300,500\n\
               chr1\t100\t700\tiso2\t0\t+\t100\t700\t0,0,0\t2\t100,100\t0,500\n";
    let map = "iso1\tgeneA\niso2\tgeneA\n";
    let v = validate(bed, map, None).unwrap();
    assert_eq!(v.genes.len(), 1);
    assert_eq!(v.genes[0].transcripts.len(), 2);
}

#[test]
fn missing_transcript_to_gene_mapping_is_rejected() {
    let bed = "chr1\t100\t200\ttx1\t0\t+\t100\t200\t0,0,0\t1\t100\t0\n";
    let map = ""; // no mapping
    let err = validate(bed, map, None).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("no transcript→gene mapping"), "got: {msg}");
}

#[test]
fn mapping_entry_absent_from_bed_is_reported() {
    let bed = "chr1\t100\t200\ttx1\t0\t+\t100\t200\t0,0,0\t1\t100\t0\n";
    let map = "tx1\tgeneA\nghost\tgeneB\n";
    let err = validate(bed, map, None).unwrap_err();
    assert!(err.to_string().contains("ghost"), "got: {err}");
}

#[test]
fn transcript_mapped_to_multiple_genes_is_rejected() {
    let bed = "chr1\t100\t200\ttx1\t0\t+\t100\t200\t0,0,0\t1\t100\t0\n";
    let map = "tx1\tgeneA\ntx1\tgeneB\n";
    let err = validate(bed, map, None).unwrap_err();
    assert!(
        err.to_string().contains("multiple mapping entries"),
        "got: {err}"
    );
}

#[test]
fn gene_on_multiple_chromosomes_is_rejected() {
    let bed = "chr1\t100\t200\ttx1\t0\t+\t100\t200\t0,0,0\t1\t100\t0\n\
               chr2\t100\t200\ttx2\t0\t+\t100\t200\t0,0,0\t1\t100\t0\n";
    let map = "tx1\tgeneA\ntx2\tgeneA\n";
    let err = validate(bed, map, None).unwrap_err();
    assert!(
        err.to_string().contains("multiple chromosome/strand"),
        "got: {err}"
    );
}

#[test]
fn gene_on_multiple_strands_is_rejected() {
    let bed = "chr1\t100\t200\ttx1\t0\t+\t100\t200\t0,0,0\t1\t100\t0\n\
               chr1\t300\t400\ttx2\t0\t-\t300\t400\t0,0,0\t1\t100\t0\n";
    let map = "tx1\tgeneA\ntx2\tgeneA\n";
    let err = validate(bed, map, None).unwrap_err();
    assert!(
        err.to_string().contains("multiple chromosome/strand"),
        "got: {err}"
    );
}

#[test]
fn duplicate_transcript_structure_is_rejected() {
    let bed = "chr1\t100\t200\ttx1\t0\t+\t100\t200\t0,0,0\t1\t100\t0\n\
               chr1\t100\t200\ttx2\t0\t+\t100\t200\t0,0,0\t1\t100\t0\n";
    let map = "tx1\tgeneA\ntx2\tgeneB\n";
    let err = validate(bed, map, None).unwrap_err();
    assert!(
        err.to_string()
            .contains("duplicate genomic transcript structure"),
        "got: {err}"
    );
}

#[test]
fn duplicate_transcript_name_is_rejected() {
    let bed = "chr1\t100\t200\ttx1\t0\t+\t100\t200\t0,0,0\t1\t100\t0\n\
               chr1\t300\t400\ttx1\t0\t+\t300\t400\t0,0,0\t1\t100\t0\n";
    let map = "tx1\tgeneA\n";
    let err = validate(bed, map, None).unwrap_err();
    assert!(
        err.to_string().contains("duplicate transcript name"),
        "got: {err}"
    );
}

#[test]
fn unknown_strand_is_rejected() {
    let bed = "chr1\t100\t200\ttx1\t0\t.\t100\t200\t0,0,0\t1\t100\t0\n";
    let map = "tx1\tgeneA\n";
    let err = validate(bed, map, None).unwrap_err();
    assert!(err.to_string().contains("strand must be"), "got: {err}");
}

#[test]
fn coordinate_out_of_bounds_is_rejected() {
    let bed = "chr1\t100\t2000\ttx1\t0\t+\t100\t2000\t0,0,0\t1\t1900\t0\n";
    let map = "tx1\tgeneA\n";
    let err = validate(bed, map, Some("chr1\t1000\n")).unwrap_err();
    assert!(err.to_string().contains("exceeds chromosome"), "got: {err}");
}

#[test]
fn chromosome_absent_from_sizes_is_rejected() {
    let bed = "chrX\t100\t200\ttx1\t0\t+\t100\t200\t0,0,0\t1\t100\t0\n";
    let map = "tx1\tgeneA\n";
    let err = validate(bed, map, Some("chr1\t1000\n")).unwrap_err();
    assert!(
        err.to_string().contains("absent from chrom.sizes"),
        "got: {err}"
    );
}

#[test]
fn overlapping_blocks_are_rejected() {
    // Second block starts before the first ends.
    let bed = "chr1\t100\t400\ttx1\t0\t+\t100\t400\t0,0,0\t2\t150,100\t0,100\n";
    let map = "tx1\tgeneA\n";
    let err = validate(bed, map, None).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("overlap") || msg.contains("abuts"),
        "got: {msg}"
    );
}

#[test]
fn header_line_in_mapping_is_tolerated() {
    let bed = "chr1\t100\t200\ttx1\t0\t+\t100\t200\t0,0,0\t1\t100\t0\n";
    let map = "transcript_id\tgene_id\ntx1\tgeneA\n";
    let v = validate(bed, map, None).unwrap();
    assert_eq!(v.genes.len(), 1);
}
