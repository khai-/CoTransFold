"""Tests for tunnel energy term."""

import numpy as np
from cotransfold.core.residue import AminoAcid, sequence_from_string
from cotransfold.core.conformation import BackboneState
from cotransfold.core.chain import NascentChain
from cotransfold.structure.coordinates import torsion_to_cartesian
from cotransfold.energy.tunnel_energy import TunnelEnergy
from cotransfold.energy.total import TotalEnergy
from cotransfold.tunnel.organisms import ecoli_tunnel


def _make_tunnel_chain(seq_str: str, helix: bool = False) -> NascentChain:
    """Create a chain with tunnel positions set."""
    chain = NascentChain.empty()
    phi = np.radians(-57.0) if helix else np.radians(-120.0)
    psi = np.radians(-47.0) if helix else np.radians(120.0)
    for aa in sequence_from_string(seq_str):
        chain.add_residue(aa, phi=phi, psi=psi)
    chain.update_exposure(90.0)
    return chain


def test_tunnel_energy_name():
    geom, elec = ecoli_tunnel()
    te = TunnelEnergy(geom, elec)
    assert te.name == "tunnel"


def test_no_tunnel_positions_returns_zero():
    """Without tunnel_positions kwarg, energy should be zero."""
    geom, elec = ecoli_tunnel()
    te = TunnelEnergy(geom, elec)
    backbone = BackboneState.extended(5)
    coords = torsion_to_cartesian(backbone)
    seq = [AminoAcid.ALA] * 5
    e = te.compute(coords, backbone, seq)
    assert e == 0.0


def test_exposed_residues_zero_tunnel_energy():
    """Residues fully outside the tunnel should contribute zero."""
    geom, elec = ecoli_tunnel()
    te = TunnelEnergy(geom, elec)

    backbone = BackboneState.extended(5)
    coords = torsion_to_cartesian(backbone)
    seq = [AminoAcid.ALA] * 5

    # All residues far outside tunnel
    tunnel_positions = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    e = te.compute(coords, backbone, seq, tunnel_positions=tunnel_positions)
    assert e == 0.0


def test_charged_residues_interact_with_tunnel():
    """Charged residues should have non-zero electrostatic energy in tunnel."""
    geom, elec = ecoli_tunnel()
    te = TunnelEnergy(geom, elec, wall_spring=0.0)  # Disable wall to isolate electrostatics

    backbone = BackboneState.extended(5)
    coords = torsion_to_cartesian(backbone)

    # Neutral residues
    seq_neutral = [AminoAcid.ALA] * 5
    tunnel_pos = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    e_neutral = te.compute(coords, backbone, seq_neutral, tunnel_positions=tunnel_pos)

    # Charged residues (Lys = +1)
    seq_charged = [AminoAcid.LYS] * 5
    e_charged = te.compute(coords, backbone, seq_charged, tunnel_positions=tunnel_pos)

    # Should be different
    assert e_charged != e_neutral, "Charged residues should interact with tunnel field"


def test_tunnel_energy_in_total():
    """TunnelEnergy should work as part of TotalEnergy."""
    geom, elec = ecoli_tunnel()
    te = TunnelEnergy(geom, elec)

    energy = TotalEnergy()
    energy.add_term(te, weight=1.0)

    chain = _make_tunnel_chain("AAAAAAAAAA")
    e = energy.compute(chain, tunnel_positions=chain.tunnel_position)
    assert isinstance(e, float)
    assert np.isfinite(e)


def test_negative_charges_destabilized_at_exit():
    """Negatively charged proteins should be destabilized at the tunnel exit.

    The ribosome surface is negatively charged, so negative residues
    near the exit experience repulsion (+8.4 kcal/mol per Streit 2025).
    """
    geom, elec = ecoli_tunnel()
    te = TunnelEnergy(geom, elec, wall_spring=0.0)

    backbone = BackboneState.extended(3)
    coords = torsion_to_cartesian(backbone)

    # Asp (charge -1) near exit (where potential is negative)
    seq_neg = [AminoAcid.ASP] * 3
    tunnel_pos = np.array([80.0, 85.0, 88.0])
    e_neg = te.compute(coords, backbone, seq_neg, tunnel_positions=tunnel_pos)

    # Lys (charge +1) near exit
    seq_pos = [AminoAcid.LYS] * 3
    e_pos = te.compute(coords, backbone, seq_pos, tunnel_positions=tunnel_pos)

    # Negative residues should have HIGHER energy at negative exit
    # (positive * negative = negative energy, negative * negative = positive energy)
    assert e_neg > e_pos, (
        f"Asp ({e_neg:.2f}) should be more destabilized than Lys ({e_pos:.2f}) at exit")
