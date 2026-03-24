"""Tests for hydrogen bond energy."""

import numpy as np
from cotransfold.core.residue import AminoAcid
from cotransfold.core.conformation import BackboneState
from cotransfold.structure.coordinates import torsion_to_cartesian
from cotransfold.energy.hbond import HydrogenBondEnergy


def test_hbond_name():
    hb = HydrogenBondEnergy()
    assert hb.name == "hbond"


def test_helix_has_hbonds():
    """Alpha helix should have significant negative H-bond energy (i->i+4 pattern)."""
    hb = HydrogenBondEnergy()
    backbone = BackboneState.alpha_helix(15)
    coords = torsion_to_cartesian(backbone)
    seq = [AminoAcid.ALA] * 15

    e = hb.compute(coords, backbone, seq)
    assert e < 0, f"Helix H-bond energy should be negative, got {e:.2f}"
    # 15-residue helix should have ~11 H-bonds (residues 0-10 donate to 4-14)
    # Each ~-1 kcal/mol
    assert e < -3.0, f"Expected significant H-bond energy, got {e:.2f}"


def test_extended_fewer_hbonds():
    """Extended chain should have fewer/weaker H-bonds than helix."""
    hb = HydrogenBondEnergy()
    seq = [AminoAcid.ALA] * 15

    helix = BackboneState.alpha_helix(15)
    e_helix = hb.compute(torsion_to_cartesian(helix), helix, seq)

    extended = BackboneState.extended(15)
    e_ext = hb.compute(torsion_to_cartesian(extended), extended, seq)

    assert e_helix < e_ext, (
        f"Helix ({e_helix:.2f}) should have more H-bonds than extended ({e_ext:.2f})")


def test_short_chain_no_hbonds():
    """Chain shorter than 4 residues cannot form i->i+4 H-bonds."""
    hb = HydrogenBondEnergy()
    backbone = BackboneState.alpha_helix(3)
    coords = torsion_to_cartesian(backbone)
    seq = [AminoAcid.ALA] * 3

    e = hb.compute(coords, backbone, seq)
    assert e == 0.0, f"Short chain should have no H-bonds, got {e:.2f}"


def test_energy_is_nonpositive():
    """H-bond energy should never be positive (only attractive)."""
    hb = HydrogenBondEnergy()
    rng = np.random.RandomState(42)

    for _ in range(5):
        n = rng.randint(5, 20)
        backbone = BackboneState(
            phi=rng.uniform(-np.pi, np.pi, n),
            psi=rng.uniform(-np.pi, np.pi, n),
            omega=np.full(n, np.pi),
        )
        coords = torsion_to_cartesian(backbone)
        seq = [AminoAcid.ALA] * n
        e = hb.compute(coords, backbone, seq)
        assert e <= 0.0, f"H-bond energy should be <= 0, got {e:.2f}"
