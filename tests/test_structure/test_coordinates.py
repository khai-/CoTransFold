"""Tests for NeRF torsion-to-Cartesian conversion."""

import numpy as np
import pytest
from cotransfold.core.conformation import BackboneState
from cotransfold.structure.coordinates import (
    torsion_to_cartesian, get_ca_coords,
    compute_rise_per_residue, compute_end_to_end,
)
from cotransfold.config import (
    BOND_LENGTH_N_CA, BOND_LENGTH_CA_C, BOND_LENGTH_C_N,
)


def test_single_residue():
    state = BackboneState.extended(1)
    coords = torsion_to_cartesian(state)
    assert coords.shape == (1, 3, 3)
    # N at origin
    np.testing.assert_allclose(coords[0, 0], [0, 0, 0])
    # CA at bond length from N
    ca_dist = np.linalg.norm(coords[0, 1] - coords[0, 0])
    np.testing.assert_allclose(ca_dist, BOND_LENGTH_N_CA, atol=1e-6)
    # C at bond length from CA
    c_dist = np.linalg.norm(coords[0, 2] - coords[0, 1])
    np.testing.assert_allclose(c_dist, BOND_LENGTH_CA_C, atol=1e-6)


def test_bond_lengths_preserved():
    """All bond lengths should match ideal values throughout the chain."""
    state = BackboneState.alpha_helix(10)
    coords = torsion_to_cartesian(state)
    flat = coords.reshape(-1, 3)  # All atoms in order: N0, CA0, C0, N1, ...

    for i in range(len(flat) - 1):
        dist = np.linalg.norm(flat[i + 1] - flat[i])
        atom_type = i % 3
        if atom_type == 0:  # N-CA bond
            np.testing.assert_allclose(dist, BOND_LENGTH_N_CA, atol=1e-5,
                err_msg=f"N-CA bond at atom {i}")
        elif atom_type == 1:  # CA-C bond
            np.testing.assert_allclose(dist, BOND_LENGTH_CA_C, atol=1e-5,
                err_msg=f"CA-C bond at atom {i}")
        else:  # C-N bond (peptide bond)
            np.testing.assert_allclose(dist, BOND_LENGTH_C_N, atol=1e-5,
                err_msg=f"C-N bond at atom {i}")


def test_alpha_helix_rise():
    """Alpha helix should have ~1.5Å rise per residue (CA-CA along helix axis)."""
    state = BackboneState.alpha_helix(20)
    coords = torsion_to_cartesian(state)

    # Measure end-to-end distance along helix axis
    ca = get_ca_coords(coords)
    # The end-to-end distance for 20 residues of helix should be ~28.5Å
    # (19 * 1.5Å rise per residue)
    e2e = np.linalg.norm(ca[-1] - ca[0])
    # Helix rise is ~1.5Å/residue, but CA-CA distance is ~3.8Å
    # The rise per residue along the axis is what we check
    rise = e2e / 19
    assert 1.2 < rise < 1.8, f"Helix rise per residue: {rise:.2f}Å (expected ~1.5Å)"


def test_alpha_helix_ca_ca_distance():
    """Consecutive CA-CA distance in alpha helix should be ~3.8Å."""
    state = BackboneState.alpha_helix(10)
    coords = torsion_to_cartesian(state)
    ca = get_ca_coords(coords)
    ca_distances = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    for i, d in enumerate(ca_distances):
        np.testing.assert_allclose(d, 3.80, atol=0.05,
            err_msg=f"CA-CA distance at residue {i}: {d:.3f}Å")


def test_extended_chain_ca_ca_distance():
    """Extended chain should have CA-CA distance ~3.8Å."""
    state = BackboneState.extended(10)
    coords = torsion_to_cartesian(state)
    ca = get_ca_coords(coords)
    ca_distances = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    for i, d in enumerate(ca_distances):
        np.testing.assert_allclose(d, 3.80, atol=0.05,
            err_msg=f"CA-CA distance at residue {i}: {d:.3f}Å")


def test_extended_chain_is_longer():
    """Extended chain should be longer end-to-end than helix of same length."""
    n = 20
    helix = torsion_to_cartesian(BackboneState.alpha_helix(n))
    extended = torsion_to_cartesian(BackboneState.extended(n))

    helix_e2e = compute_end_to_end(helix)
    ext_e2e = compute_end_to_end(extended)
    assert ext_e2e > helix_e2e, (
        f"Extended ({ext_e2e:.1f}Å) should be longer than helix ({helix_e2e:.1f}Å)")


def test_empty_state():
    state = BackboneState.empty()
    coords = torsion_to_cartesian(state)
    assert coords.shape == (0, 3, 3)


def test_no_nan_or_inf():
    """No NaN or Inf in coordinates for various conformations."""
    for n in [1, 5, 20, 50]:
        for factory in [BackboneState.alpha_helix, BackboneState.extended]:
            state = factory(n)
            coords = torsion_to_cartesian(state)
            assert not np.any(np.isnan(coords)), f"NaN found for {factory.__name__}({n})"
            assert not np.any(np.isinf(coords)), f"Inf found for {factory.__name__}({n})"
