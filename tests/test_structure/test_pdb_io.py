"""Tests for PDB file I/O."""

import os
import tempfile
import numpy as np
from cotransfold.core.residue import AminoAcid, sequence_from_string
from cotransfold.core.conformation import BackboneState
from cotransfold.structure.coordinates import torsion_to_cartesian
from cotransfold.structure.pdb_io import write_pdb, write_trajectory_pdb


def test_write_pdb_creates_file():
    state = BackboneState.alpha_helix(5)
    coords = torsion_to_cartesian(state)
    seq = sequence_from_string("AAAAA")

    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as f:
        path = f.name

    try:
        write_pdb(path, coords, seq)
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()

        # Should have 15 ATOM lines (5 residues * 3 atoms)
        atom_lines = [l for l in content.split('\n') if l.startswith('ATOM')]
        assert len(atom_lines) == 15

        # Should end with END
        assert content.strip().endswith('END')
    finally:
        os.unlink(path)


def test_pdb_atom_names():
    state = BackboneState.extended(2)
    coords = torsion_to_cartesian(state)
    seq = sequence_from_string("AG")

    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as f:
        path = f.name

    try:
        write_pdb(path, coords, seq)
        with open(path) as f:
            lines = [l for l in f.readlines() if l.startswith('ATOM')]

        # Check atom names cycle: N, CA, C, N, CA, C
        expected_atoms = ['N', 'CA', 'C', 'N', 'CA', 'C']
        for line, expected in zip(lines, expected_atoms):
            atom_name = line[12:16].strip()
            assert atom_name == expected

        # Check residue names
        assert lines[0][17:20] == 'ALA'
        assert lines[3][17:20] == 'GLY'
    finally:
        os.unlink(path)


def test_write_trajectory_pdb():
    seq = sequence_from_string("AAA")

    # Simulate growing chain: 1 residue, 2 residues, 3 residues
    frames = []
    for n in range(1, 4):
        state = BackboneState.alpha_helix(n)
        frames.append(torsion_to_cartesian(state))

    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as f:
        path = f.name

    try:
        write_trajectory_pdb(path, frames, seq)
        with open(path) as f:
            content = f.read()

        # Should have 3 MODEL records
        model_count = content.count('MODEL')
        assert model_count == 3

        endmdl_count = content.count('ENDMDL')
        assert endmdl_count == 3
    finally:
        os.unlink(path)
