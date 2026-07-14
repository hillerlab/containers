//! End-to-end `prepare` tests: exact expression conservation, determinism, original
//! preservation, and fusion partner-order correctness (spec §4.12 acceptance gate, §7 Test A).

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use longread::generate::{EventWeights, NewIsoformsMode};
use longread::prepare::{run, PrepareParams};
use tempfile::TempDir;

const BED: &str = "\
chr1\t1000\t1700\ttxA1\t0\t+\t1000\t1700\t0,0,0\t3\t100,100,100\t0,300,600
chr1\t1000\t1700\ttxA2\t0\t+\t1000\t1700\t0,0,0\t2\t100,100\t0,600
chr1\t5000\t5700\ttxB1\t0\t-\t5000\t5700\t0,0,0\t3\t100,100,100\t0,300,600
chr1\t8000\t8400\ttxC1\t0\t+\t8000\t8400\t0,0,0\t2\t100,100\t0,300
chr2\t2000\t2500\ttxD1\t0\t+\t2000\t2500\t0,0,0\t2\t100,100\t0,400
";
const MAP: &str = "txA1\tgeneA\ntxA2\tgeneA\ntxB1\tgeneB\ntxC1\tgeneC\ntxD1\tgeneD\n";

fn params(dir: &Path, prefix: &str, threads: usize) -> PrepareParams {
    let bed = dir.join("in.bed");
    let map = dir.join("map.tsv");
    std::fs::File::create(&bed)
        .unwrap()
        .write_all(BED.as_bytes())
        .unwrap();
    std::fs::File::create(&map)
        .unwrap()
        .write_all(MAP.as_bytes())
        .unwrap();
    PrepareParams {
        bed,
        transcript_gene: map,
        chrom_sizes: None,
        output_prefix: dir.join(prefix).to_string_lossy().into_owned(),
        seed: 42,
        threads,
        new_isoforms_mode: NewIsoformsMode::Fixed(3),
        event_weights: EventWeights::uniform(),
        max_event_attempts: 100,
        min_transcript_length: 0,
        fusion_count: 2,
        min_fusion_intron: 1,
        max_fusion_distance: None,
        fusion_expression_scale: 1.0,
        total_molecules: 100_000,
        alpha: 4.0,
        require_exact_event_count: false,
    }
}

fn read_tsv(path: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1) // header
        .filter(|l| !l.is_empty())
        .map(|l| l.split('\t').map(|s| s.to_string()).collect())
        .collect()
}

/// Parse a BED12 file into `name -> (strand, absolute exons)`.
fn read_bed(path: &Path) -> HashMap<String, (char, Vec<(u64, u64)>)> {
    let mut out = HashMap::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let start: u64 = f[1].parse().unwrap();
        let name = f[3].to_string();
        let strand = f[5].chars().next().unwrap();
        let sizes: Vec<u64> = f[10]
            .trim_end_matches(',')
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let starts: Vec<u64> = f[11]
            .trim_end_matches(',')
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let exons: Vec<(u64, u64)> = starts
            .iter()
            .zip(sizes.iter())
            .map(|(&rs, &sz)| (start + rs, start + rs + sz))
            .collect();
        out.insert(name, (strand, exons));
    }
    out
}

fn outputs(prefix: &str) -> HashMap<&'static str, PathBuf> {
    let mut m = HashMap::new();
    for suf in [
        "isoforms.bed",
        "gene_depth.tsv",
        "isoform_depth.tsv",
        "manifest.tsv",
    ] {
        m.insert(suf, PathBuf::from(format!("{prefix}.{suf}")));
    }
    m
}

#[test]
fn conservation_and_originals_preserved() {
    let dir = TempDir::new().unwrap();
    let p = params(dir.path(), "sim", 1);
    let (stats, out) = run(&p).unwrap();

    // Gene molecules sum to total.
    let gene = read_tsv(&out.gene_depth);
    let gene_sum: u64 = gene.iter().map(|r| r[1].parse::<u64>().unwrap()).sum();
    assert_eq!(gene_sum, 100_000);
    assert_eq!(stats.total_allocated_gene_molecules, 100_000);

    // Isoform molecules sum to total, and per gene sum to the gene depth.
    let iso = read_tsv(&out.isoform_depth);
    let iso_sum: u64 = iso.iter().map(|r| r[2].parse::<u64>().unwrap()).sum();
    assert_eq!(iso_sum, 100_000);

    let gene_depth: HashMap<String, u64> = gene
        .iter()
        .map(|r| (r[0].clone(), r[1].parse().unwrap()))
        .collect();
    let mut per_gene: HashMap<String, u64> = HashMap::new();
    for r in &iso {
        *per_gene.entry(r[1].clone()).or_insert(0) += r[2].parse::<u64>().unwrap();
    }
    for (g, d) in &gene_depth {
        assert_eq!(
            per_gene.get(g).copied().unwrap_or(0),
            *d,
            "gene {g} isoform sum != depth"
        );
    }

    // All original transcripts are preserved in the output.
    let bed = read_bed(&out.isoforms_bed);
    for original in ["txA1", "txA2", "txB1", "txC1", "txD1"] {
        assert!(
            bed.contains_key(original),
            "original {original} was dropped"
        );
    }
}

#[test]
fn fusions_contain_both_partners_in_correct_order() {
    let dir = TempDir::new().unwrap();
    let p = params(dir.path(), "sim", 1);
    let (_stats, out) = run(&p).unwrap();
    let bed = read_bed(&out.isoforms_bed);
    let manifest = read_tsv(&out.manifest);

    let mut fusion_count = 0;
    for row in &manifest {
        // manifest columns: id, gene, parent, event_type, is_original, is_generated, is_fusion, pg5, pt5, pg3, pt3, ...
        if row[6] != "true" {
            continue; // is_fusion
        }
        fusion_count += 1;
        let id = &row[0];
        let pt5 = &row[8];
        let pt3 = &row[10];
        let (strand, fexons) = &bed[id];
        let (_, e5) = &bed[pt5];
        let (_, e3) = &bed[pt3];

        // The fusion exons are exactly the union of both partners' exons, sorted.
        let mut expected: Vec<(u64, u64)> = e5.iter().chain(e3.iter()).copied().collect();
        expected.sort();
        assert_eq!(fexons, &expected, "fusion {id} exon set mismatch");

        // 5′ partner comes first in transcript orientation.
        let p5_max = e5.iter().map(|e| e.1).max().unwrap();
        let p3_min = e3.iter().map(|e| e.0).min().unwrap();
        let p3_max = e3.iter().map(|e| e.1).max().unwrap();
        let p5_min = e5.iter().map(|e| e.0).min().unwrap();
        match strand {
            '+' => assert!(p5_max <= p3_min, "+ fusion {id}: 5′ partner not upstream"),
            '-' => assert!(p3_max <= p5_min, "- fusion {id}: 5′ partner not downstream"),
            _ => panic!("unexpected strand"),
        }
    }
    assert_eq!(fusion_count, 2, "expected two fusions");
}

#[test]
fn deterministic_across_threads() {
    let dir = TempDir::new().unwrap();
    let p1 = params(dir.path(), "t1", 1);
    let p4 = params(dir.path(), "t4", 4);
    run(&p1).unwrap();
    run(&p4).unwrap();
    for suf in [
        "isoforms.bed",
        "gene_depth.tsv",
        "isoform_depth.tsv",
        "manifest.tsv",
    ] {
        let a =
            std::fs::read(outputs(&dir.path().join("t1").to_string_lossy())[suf].clone()).unwrap();
        let b =
            std::fs::read(outputs(&dir.path().join("t4").to_string_lossy())[suf].clone()).unwrap();
        assert_eq!(a, b, "{suf} differs across thread counts");
    }
}

#[test]
fn require_exact_event_count_succeeds_when_met() {
    let dir = TempDir::new().unwrap();
    let mut p = params(dir.path(), "exact", 1);
    p.new_isoforms_mode = NewIsoformsMode::Fixed(1);
    p.fusion_count = 1;
    p.require_exact_event_count = true;
    // Should succeed: each gene can produce at least one event and one fusion exists.
    assert!(run(&p).is_ok());
}
