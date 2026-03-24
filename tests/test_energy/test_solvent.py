"""Tests for implicit solvation energy."""

import numpy as np
from cotransfold.core.residue import AminoAcid
from cotransfold.core.conformation import BackboneState
from cotransfold.structure.coordinates import torsion_to_cartesian
from cotransfold.energy.solvent import SolventEnergy, SOLVATION_PARAMS


def test_solvent_name():
    sol = SolventEnergy()
    assert sol.name == "solvent"


def test_hydrophobic_residues_have_positive_solvation():
    """Ile, Leu, Val, Phe should have positive solvation cost (penalty for exposure)."""
    for aa in [AminoAcid.ILE, AminoAcid.LEU, AminoAcid.VAL, AminoAcid.PHE]:
        assert SOLVATION_PARAMS[aa] > 0, f"{aa.name} should be hydrophobic"


def test_charged_residues_negative_solvation():
    """Asp, Glu, Arg, Lys should have negative solvation (prefer solvent)."""
    for aa in [AminoAcid.ASP, AminoAcid.GLU, AminoAcid.ARG, AminoAcid.LYS]:
        assert SOLVATION_PARAMS[aa] < 0, f"{aa.name} should prefer solvent"


def test_in_tunnel_no_solvent_energy():
    """Residues inside the tunnel should have no solvent energy."""
    sol = SolventEnergy()
    backbone = BackboneState.extended(5)
    coords = torsion_to_cartesian(backbone)
    seq = [AminoAcid.LEU] * 5  # Very hydrophobic

    # All residues inside tunnel
    tunnel_positions = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    e = sol.compute(coords, backbone, seq,
                    tunnel_positions=tunnel_positions, tunnel_length=90.0)
    assert e == 0.0, f"In-tunnel residues should have no solvent energy, got {e}"


def test_exposed_hydrophobic_has_energy():
    """Exposed hydrophobic residues should incur solvation penalty."""
    sol = SolventEnergy()
    backbone = BackboneState.extended(5)
    coords = torsion_to_cartesian(backbone)
    seq = [AminoAcid.LEU] * 5

    # All residues outside tunnel
    tunnel_positions = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    e = sol.compute(coords, backbone, seq,
                    tunnel_positions=tunnel_positions, tunnel_length=90.0)
    assert e > 0, f"Exposed Leu should have positive solvation energy, got {e}"


def test_single_residue_returns_zero():
    sol = SolventEnergy()
    backbone = BackboneState.extended(1)
    coords = torsion_to_cartesian(backbone)
    seq = [AminoAcid.ALA]
    e = sol.compute(coords, backbone, seq)
    assert e == 0.0
