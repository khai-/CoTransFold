"""Tests for NascentChain."""

import numpy as np
from cotransfold.core.residue import AminoAcid
from cotransfold.core.chain import NascentChain
from cotransfold.config import RESIDUE_ADVANCE_DISTANCE


def test_empty_chain():
    chain = NascentChain.empty()
    assert chain.chain_length == 0
    assert chain.num_exposed == 0


def test_add_single_residue():
    chain = NascentChain.empty()
    chain.add_residue(AminoAcid.ALA)
    assert chain.chain_length == 1
    assert chain.sequence[0] == AminoAcid.ALA
    assert chain.backbone.num_residues == 1
    # First residue is at PTC (position 0)
    assert chain.tunnel_position[0] == 0.0


def test_add_multiple_residues_advances_tunnel():
    chain = NascentChain.empty()
    chain.add_residue(AminoAcid.ALA)
    chain.add_residue(AminoAcid.GLY)
    chain.add_residue(AminoAcid.VAL)

    assert chain.chain_length == 3
    # First residue (ALA) has been advanced twice
    np.testing.assert_allclose(
        chain.tunnel_position[0], 2 * RESIDUE_ADVANCE_DISTANCE)
    # Second residue (GLY) has been advanced once
    np.testing.assert_allclose(
        chain.tunnel_position[1], 1 * RESIDUE_ADVANCE_DISTANCE)
    # Third residue (VAL) just added
    np.testing.assert_allclose(chain.tunnel_position[2], 0.0)


def test_update_exposure():
    chain = NascentChain.empty()
    # Add 30 residues to push first ones through a 90Å tunnel
    for _ in range(30):
        chain.add_residue(AminoAcid.ALA)

    # First residue at position 29 * 3.5 = 101.5Å
    chain.update_exposure(tunnel_length=90.0)

    # First few residues should be exposed (>90Å from PTC)
    assert chain.is_exposed[0] is True or chain.is_exposed[0] == True
    # Last residue at position 0 should not be exposed
    assert chain.is_exposed[-1] is False or chain.is_exposed[-1] == False


def test_frozen_mask():
    chain = NascentChain.empty()
    for _ in range(20):
        chain.add_residue(AminoAcid.ALA)

    mask = chain.get_frozen_mask(upper_end=30.0, constriction_end=50.0)
    assert mask.shape == (20,)

    # Last residue (position 0) should be frozen
    assert mask[-1] == 0.0
    # First residue (position 19*3.5=66.5) should be free
    assert mask[0] == 1.0


def test_copy_is_independent():
    chain = NascentChain.empty()
    chain.add_residue(AminoAcid.ALA)
    copy = chain.copy()
    copy.add_residue(AminoAcid.GLY)
    assert chain.chain_length == 1
    assert copy.chain_length == 2
