#!/usr/bin/env python3
"""Unit tests for images/tiberius/merge.py (no TensorFlow, no merger binary)."""

from __future__ import annotations

import unittest
from pathlib import Path

import merge as tm

GFF3 = """##gff-version 3
chr10	Tiberius	gene	1	9	.	+	.	ID=gene_000001
chr10	Tiberius	transcript	1	9	.	+	.	ID=gene_000001.t1;Parent=gene_000001
chr10	Tiberius	exon	1	9	.	+	.	ID=gene_000001.t1.exon1;Parent=gene_000001.t1
chr10	Tiberius	CDS	1	9	.	+	0	ID=gene_000001.t1.cds1;Parent=gene_000001.t1
chr2	Tiberius	gene	1	9	.	+	.	ID=gene_000002
chr2	Tiberius	transcript	1	9	.	+	.	ID=gene_000002.t1;Parent=gene_000002
chr2	Tiberius	exon	1	9	.	+	.	ID=gene_000002.t1.exon1;Parent=gene_000002.t1
chr2	Tiberius	CDS	1	9	.	+	0	ID=gene_000002.t1.cds1;Parent=gene_000002.t1
"""


class MergeHelpers(unittest.TestCase):
    def test_parse_manifest_annevo_header(self):
        path = Path("/tmp/tiberius_manifest.tsv")
        path.write_text("order\tseqid\tlength\n0\tchr2\t9\n1\tchr10\t9\n", encoding="utf-8")
        self.assertEqual(tm.parse_manifest(path), ["chr2", "chr10"])

    def test_reorder_restores_fasta_order_not_seqid_sort(self):
        ordered = tm.reorder_gff3(GFF3, ["chr2", "chr10"])
        seqids = [
            line.split("\t", 1)[0]
            for line in ordered.splitlines()
            if line and not line.startswith("#")
        ]
        self.assertEqual(seqids[0], "chr2")
        self.assertIn("chr10", seqids)
        self.assertLess(seqids.index("chr2"), seqids.index("chr10"))

    def test_gff3_to_gtf_attributes(self):
        gtf = tm.gff3_to_gtf(GFF3)
        self.assertIn('gene_id "gene_000001"', gtf)
        self.assertIn('transcript_id "gene_000001.t1"', gtf)
        self.assertNotIn("ID=", gtf)

    def test_extract_uses_merged_ids(self):
        fasta = {"chr10": "ATGCATTAA", "chr2": "ATGCCCTAA"}
        cds, prot = tm.extract_sequences(GFF3, fasta)
        self.assertIn(">gene_000001.t1", cds)
        self.assertIn(">gene_000002.t1", prot)
        self.assertIn("MH", prot)
        self.assertNotIn("gene_old", cds)


if __name__ == "__main__":
    unittest.main()
