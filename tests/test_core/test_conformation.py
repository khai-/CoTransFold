"""Tests for BackboneState."""

import numpy as np
from cotransfold.core.conformation import BackboneState


def test_empty_state():
    state = BackboneState.empty()
    assert state.num_residues == 0


def test_extended_conformation():
    state = BackboneState.extended(10)
    assert state.num_residues == 10
    np.testing.assert_allclose(state.phi, np.radians(-120.0))
    np.testing.assert_allclose(state.psi, np.radians(120.0))
    np.testing.assert_allclose(state.omega, np.pi)


def test_alpha_helix_conformation():
    state = BackboneState.alpha_helix(10)
    assert state.num_residues == 10
    np.testing.assert_allclose(state.phi, np.radians(-57.0))
    np.testing.assert_allclose(state.psi, np.radians(-47.0))


def test_extend():
    state = BackboneState.empty()
    state = state.extend(phi=-1.0, psi=1.0)
    assert state.num_residues == 1
    state = state.extend(phi=-2.0, psi=2.0)
    assert state.num_residues == 2
    assert state.phi[0] == -1.0
    assert state.phi[1] == -2.0


def test_copy_is_independent():
    original = BackboneState.alpha_helix(5)
    copy = original.copy()
    copy.phi[0] = 999.0
    assert original.phi[0] != 999.0


def test_set_angles():
    state = BackboneState.extended(5)
    state.set_angles(2, phi=-1.0, psi=-0.8)
    assert state.phi[2] == -1.0
    assert state.psi[2] == -0.8
    # Omega unchanged
    np.testing.assert_allclose(state.omega[2], np.pi)


def test_torsion_vector_roundtrip():
    state = BackboneState.alpha_helix(5)
    vec = state.get_torsion_vector()
    assert vec.shape == (10,)  # 5 residues * 2 (phi, psi)

    state2 = BackboneState.extended(5)
    state2.set_from_torsion_vector(vec)
    np.testing.assert_allclose(state2.phi, state.phi)
    np.testing.assert_allclose(state2.psi, state.psi)
