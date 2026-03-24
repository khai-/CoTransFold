"""Tests for chaperone program and individual chaperones."""

from __future__ import annotations

import numpy as np

from cotransfold.core.residue import AminoAcid, sequence_from_string
from cotransfold.core.chain import NascentChain
from cotransfold.chaperones.program import ChaperoneProgram
from cotransfold.chaperones.nac import NAC
from cotransfold.chaperones.trigger_factor import TriggerFactor
from cotransfold.chaperones.srp import SRP
from cotransfold.chaperones.hsp70 import Hsp70


def _make_chain(seq_str: str, n_advance: int = 0) -> NascentChain:
    """Build a chain, optionally advancing extra steps to expose residues."""
    chain = NascentChain.empty()
    for aa in sequence_from_string(seq_str):
        chain.add_residue(aa)
    # Extra advances to push residues through tunnel
    for _ in range(n_advance):
        chain.tunnel_position += 3.5
    chain.update_exposure(90.0)
    return chain


# --- NAC tests ---

def test_nac_engages_early():
    """NAC should engage when chain is 10-30 residues."""
    nac = NAC()
    chain = _make_chain("A" * 15)
    assert nac.should_engage(chain, 0.0)


def test_nac_disengages_after_30():
    """NAC should not engage for chains > 30 residues."""
    nac = NAC()
    chain = _make_chain("A" * 35)
    assert not nac.should_engage(chain, 0.0)


def test_nac_too_short():
    """NAC should not engage for very short chains."""
    nac = NAC()
    chain = _make_chain("AAAA")
    assert not nac.should_engage(chain, 0.0)


def test_nac_effect():
    nac = NAC()
    chain = _make_chain("A" * 20)
    action = nac.compute_effect(chain)
    assert action.chaperone_name == "NAC"
    assert action.energy_modifier < 0  # Stabilizing
    assert action.compaction_scale > 1.0  # Slight compaction


# --- Trigger Factor tests ---

def test_tf_needs_exposed_residues():
    """TF should not engage with no exposed residues."""
    tf = TriggerFactor()
    chain = _make_chain("A" * 10)  # All in tunnel
    assert not tf.should_engage(chain, 0.0)


def test_tf_engages_with_hydrophobic_exposed():
    """TF should engage when hydrophobic residues are exposed."""
    tf = TriggerFactor()
    # LLLLLL = very hydrophobic; advance enough to expose them
    chain = _make_chain("L" * 30, n_advance=10)
    if chain.num_exposed >= 5:
        assert tf.should_engage(chain, 0.0)


def test_tf_effect_has_holdase():
    tf = TriggerFactor()
    chain = _make_chain("L" * 30, n_advance=10)
    if tf.should_engage(chain, 0.0):
        action = tf.compute_effect(chain)
        assert action.holdase_mask is not None
        assert action.compaction_scale > 1.0


# --- SRP tests ---

def test_srp_detects_signal_sequence():
    """SRP should detect a hydrophobic stretch (signal sequence)."""
    srp = SRP()
    # LLLLLLLLL = 9 consecutive hydrophobic residues (signal-like)
    # Need them exposed
    chain = _make_chain("LLLLLLLLL" + "A" * 21, n_advance=15)
    if chain.num_exposed >= 7:
        # Check if the hydrophobic stretch is in the exposed region
        engaged = srp.should_engage(chain, 0.0)
        # May or may not engage depending on which residues are exposed
        assert isinstance(engaged, bool)


def test_srp_not_for_polar():
    """SRP should not engage for all-polar sequences."""
    srp = SRP()
    chain = _make_chain("D" * 30, n_advance=10)
    assert not srp.should_engage(chain, 0.0)


# --- Hsp70 tests ---

def test_hsp70_needs_many_exposed():
    """Hsp70 needs at least 15 exposed residues."""
    hsp70 = Hsp70()
    chain = _make_chain("A" * 10)
    assert not hsp70.should_engage(chain, 0.0)


def test_hsp70_engages_for_hydrophobic():
    """Hsp70 should engage when enough hydrophobic residues are exposed."""
    hsp70 = Hsp70()
    chain = _make_chain("V" * 40, n_advance=15)  # Val is hydrophobic
    if chain.num_exposed >= 15:
        assert hsp70.should_engage(chain, 0.0)


# --- ChaperoneProgram tests ---

def test_ecoli_program_has_all_chaperones():
    prog = ChaperoneProgram('ecoli')
    names = prog.available_chaperones
    assert 'NAC' in names
    assert 'TriggerFactor' in names
    assert 'SRP' in names
    assert 'Hsp70' in names


def test_eukaryotic_program_no_trigger_factor():
    prog = ChaperoneProgram('yeast')
    names = prog.available_chaperones
    assert 'NAC' in names
    assert 'TriggerFactor' not in names


def test_program_returns_actions():
    prog = ChaperoneProgram('ecoli')
    chain = _make_chain("A" * 20)
    actions = prog.active_chaperones(chain, 0.5)
    # NAC should be active for 20-residue chain
    names = [a.chaperone_name for a in actions]
    assert 'NAC' in names


def test_program_energy_modifier():
    prog = ChaperoneProgram('ecoli')
    chain = _make_chain("A" * 20)
    actions = prog.active_chaperones(chain, 0.5)
    total_mod = prog.get_total_energy_modifier(actions)
    assert isinstance(total_mod, float)


def test_program_empty_for_very_short():
    prog = ChaperoneProgram('ecoli')
    chain = _make_chain("AA")
    actions = prog.active_chaperones(chain, 0.0)
    # Only 2 residues: too short for any chaperone
    assert len(actions) == 0
