"""Tests for Ramachandran potential."""

import numpy as np
from cotransfold.core.residue import AminoAcid
from cotransfold.energy.ramachandran import RamachandranEnergy, rama_probability
from cotransfold.core.conformation import BackboneState


def test_alpha_helix_low_energy():
    """Alpha helix region should have high probability (low energy)."""
    phi = np.radians(-63)
    psi = np.radians(-43)
    p = rama_probability(phi, psi, AminoAcid.ALA)
    assert p > 0.1, f"Alpha helix probability too low: {p}"


def test_disallowed_region_high_energy():
    """Disallowed region (phi=0, psi=0) should have low probability."""
    p_disallowed = rama_probability(0.0, 0.0, AminoAcid.ALA)
    p_helix = rama_probability(np.radians(-63), np.radians(-43), AminoAcid.ALA)
    assert p_helix > p_disallowed * 5, "Helix should be much more probable than (0,0)"


def test_glycine_wider_distribution():
    """Glycine should have significant probability in left-handed helix region."""
    phi = np.radians(63)
    psi = np.radians(43)
    p_gly = rama_probability(phi, psi, AminoAcid.GLY)
    p_ala = rama_probability(phi, psi, AminoAcid.ALA)
    assert p_gly > p_ala * 2, "Glycine should be more permissive in left-handed region"


def test_proline_restricted_phi():
    """Proline should strongly prefer phi near -63°."""
    p_correct = rama_probability(np.radians(-63), np.radians(-35), AminoAcid.PRO)
    p_wrong = rama_probability(np.radians(-120), np.radians(130), AminoAcid.PRO)
    assert p_correct > p_wrong * 2, "Proline should prefer phi~-63°"


def test_energy_term_interface():
    """RamachandranEnergy should work with the EnergyTerm interface."""
    from cotransfold.structure.coordinates import torsion_to_cartesian

    rama = RamachandranEnergy()
    assert rama.name == "ramachandran"

    backbone = BackboneState.alpha_helix(10)
    coords = torsion_to_cartesian(backbone)
    seq = [AminoAcid.ALA] * 10

    e = rama.compute(coords, backbone, seq)
    assert isinstance(e, float)
    assert np.isfinite(e)


def test_helix_lower_than_random():
    """Alpha helix should have lower Ramachandran energy than random angles."""
    from cotransfold.structure.coordinates import torsion_to_cartesian

    rama = RamachandranEnergy()
    seq = [AminoAcid.ALA] * 10

    # Helix
    helix = BackboneState.alpha_helix(10)
    e_helix = rama.compute(torsion_to_cartesian(helix), helix, seq)

    # Random (but valid) angles
    rng = np.random.RandomState(42)
    random_state = BackboneState(
        phi=rng.uniform(-np.pi, np.pi, 10),
        psi=rng.uniform(-np.pi, np.pi, 10),
        omega=np.full(10, np.pi),
    )
    e_random = rama.compute(torsion_to_cartesian(random_state), random_state, seq)

    assert e_helix < e_random, (
        f"Helix energy ({e_helix:.2f}) should be lower than random ({e_random:.2f})")
