"""Tests for the validation pipeline: PDB parser, fetch, comparison."""

from __future__ import annotations

import os
import tempfile
import numpy as np

from cotransfold.validation.pdb_parser import parse_pdb, _THREE_TO_ONE
from cotransfold.validation.benchmark_set import (
    BENCHMARK_SET, get_by_category, get_by_name, get_all_categories, summary_table,
)
from cotransfold.structure.rmsd import superposed_rmsd, tm_score
from cotransfold.validation.metrics import gdt_ts


# --- PDB Parser tests ---

def _write_minimal_pdb(filepath: str) -> None:
    """Write a minimal PDB file for testing."""
    lines = [
        "HEADER    TEST PROTEIN                            01-JAN-00   1TST",
        "REMARK   2 RESOLUTION.    1.50 ANGSTROMS.",
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N",
        "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C",
        "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C",
        "ATOM      4  N   GLY A   2       4.000   5.000   6.000  1.00  0.00           N",
        "ATOM      5  CA  GLY A   2       5.000   6.000   7.000  1.00  0.00           C",
        "ATOM      6  C   GLY A   2       6.000   7.000   8.000  1.00  0.00           C",
        "ATOM      7  N   VAL A   3       7.000   8.000   9.000  1.00  0.00           N",
        "ATOM      8  CA  VAL A   3       8.000   9.000  10.000  1.00  0.00           C",
        "ATOM      9  C   VAL A   3       9.000  10.000  11.000  1.00  0.00           C",
        "END",
    ]
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def test_parse_pdb_basic():
    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as f:
        path = f.name
    _write_minimal_pdb(path)

    try:
        struct = parse_pdb(path, chain_id='A')
        assert struct.n_residues == 3
        assert struct.sequence == 'AGV'
        assert struct.resolution == 1.50
        assert struct.ca_coords.shape == (3, 3)
        np.testing.assert_allclose(struct.ca_coords[0], [2, 3, 4])
        np.testing.assert_allclose(struct.ca_coords[1], [5, 6, 7])
    finally:
        os.unlink(path)


def test_three_to_one_covers_standard():
    standard = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY',
                'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                'THR', 'TRP', 'TYR', 'VAL']
    for res in standard:
        assert res in _THREE_TO_ONE, f"Missing {res}"


# --- Benchmark set tests ---

def test_benchmark_set_not_empty():
    assert len(BENCHMARK_SET) >= 15


def test_all_entries_have_required_fields():
    for e in BENCHMARK_SET:
        assert len(e.name) > 0
        assert len(e.pdb_id) == 4
        assert len(e.chain_id) == 1
        assert len(e.sequence) > 0
        assert e.n_residues > 0
        assert len(e.category) > 0
        assert e.n_residues == len(e.sequence), (
            f"{e.name}: n_residues={e.n_residues} != len(sequence)={len(e.sequence)}")


def test_categories_diverse():
    cats = get_all_categories()
    assert len(cats) >= 6, f"Need at least 6 categories, got {len(cats)}: {cats}"


def test_get_by_name():
    entry = get_by_name("Ubiquitin")
    assert entry.pdb_id == "1UBQ"


def test_get_by_pdb_id():
    entry = get_by_name("1UBQ")
    assert entry.name == "Ubiquitin"


def test_get_by_category():
    small = get_by_category("small_domain")
    assert len(small) >= 2


def test_summary_table():
    table = summary_table()
    assert "Ubiquitin" in table
    assert "PDB" in table


def test_has_cotranslational_entries():
    ct_entries = [e for e in BENCHMARK_SET if e.cotranslational_data]
    assert len(ct_entries) >= 2, "Need proteins with co-translational folding data"


def test_has_fold_switching_entries():
    fs_entries = get_by_category("fold_switching")
    assert len(fs_entries) >= 1


def test_has_mutation_data():
    with_mutations = [e for e in BENCHMARK_SET if len(e.mutations) > 0]
    assert len(with_mutations) >= 2


# --- GDT-TS tests ---

def test_gdt_ts_identical():
    coords = np.random.RandomState(42).randn(20, 3) * 10
    score = gdt_ts(coords, coords)
    np.testing.assert_allclose(score, 1.0, atol=1e-6)


def test_gdt_ts_range():
    rng = np.random.RandomState(42)
    c1 = rng.randn(20, 3) * 10
    c2 = rng.randn(20, 3) * 10
    score = gdt_ts(c1, c2)
    assert 0.0 <= score <= 1.0
