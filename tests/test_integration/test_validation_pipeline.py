"""Tests for the validation pipeline: PDB parser, benchmark set, metrics."""

from __future__ import annotations

import os
import tempfile
import numpy as np

from cotransfold.validation.pdb_parser import parse_pdb, _THREE_TO_ONE
from cotransfold.validation.benchmark_set import (
    BENCHMARK_SET, get_by_category, get_by_name, get_all_categories,
)
from cotransfold.validation.metrics import gdt_ts


def test_parse_pdb():
    lines = [
        "HEADER    TEST PROTEIN                            01-JAN-00   1TST",
        "REMARK   2 RESOLUTION.    1.50 ANGSTROMS.",
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N",
        "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C",
        "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C",
        "ATOM      4  N   GLY A   2       4.000   5.000   6.000  1.00  0.00           N",
        "ATOM      5  CA  GLY A   2       5.000   6.000   7.000  1.00  0.00           C",
        "ATOM      6  C   GLY A   2       6.000   7.000   8.000  1.00  0.00           C",
        "END",
    ]
    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as f:
        f.write('\n'.join(lines) + '\n')
        path = f.name
    try:
        struct = parse_pdb(path, 'A')
        assert struct.n_residues == 2
        assert struct.sequence == 'AG'
        assert struct.resolution == 1.50
        np.testing.assert_allclose(struct.ca_coords[0], [2, 3, 4])
    finally:
        os.unlink(path)


def test_benchmark_set_integrity():
    """All benchmark entries have valid fields and consistent lengths."""
    assert len(BENCHMARK_SET) >= 15
    cats = get_all_categories()
    assert len(cats) >= 6

    for e in BENCHMARK_SET:
        assert len(e.pdb_id) == 4
        assert e.n_residues == len(e.sequence), (
            f"{e.name}: n_residues={e.n_residues} != len(sequence)={len(e.sequence)}")

    # Key categories exist
    assert len(get_by_category("fold_switching")) >= 1
    assert len(get_by_category("af_weakness")) >= 1
    assert len([e for e in BENCHMARK_SET if e.cotranslational_data]) >= 2

    # Lookup works
    assert get_by_name("Ubiquitin").pdb_id == "1UBQ"
    assert get_by_name("1UBQ").name == "Ubiquitin"


def test_gdt_ts():
    rng = np.random.RandomState(42)
    coords = rng.randn(20, 3) * 10
    assert abs(gdt_ts(coords, coords) - 1.0) < 1e-6
    assert 0.0 <= gdt_ts(rng.randn(20, 3), rng.randn(20, 3)) <= 1.0
