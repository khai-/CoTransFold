"""Tests for the co-translational folding simulation engine."""

from __future__ import annotations

import os
import tempfile
import numpy as np

from cotransfold.simulator.engine import SimulationEngine, SimulationConfig
from cotransfold.core.residue import AminoAcid


def _fast_config(**overrides) -> SimulationConfig:
    """Minimal config for fast tests."""
    defaults = dict(
        max_steps_per_residue=5,
        min_steps_per_residue=3,
        equilibration_steps=20,
        use_kinetics=False,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def test_simulate_and_trajectory():
    """Core simulation produces correct trajectory."""
    engine = SimulationEngine(_fast_config())
    traj = engine.simulate("AAAAAAAAAA")
    # 10 translation + 1 equilibration
    assert traj.num_steps == 11
    assert traj.final_backbone is not None
    assert traj.final_backbone.num_residues == 10
    # Energy decomposition present
    assert "ramachandran" in traj.snapshots[-1].energy_decomposed


def test_exposed_count_increases():
    """Exposed residues increase as chain grows through tunnel."""
    engine = SimulationEngine(_fast_config())
    traj = engine.simulate("A" * 30)
    exposed = traj.exposed_trace
    # Last translation step should have some exposed (30*3.5=105 > 90Å tunnel)
    assert exposed[-2] > 0  # -2 because -1 is equilibration (all exposed)
    assert exposed[-1] == 30  # equilibration: all exposed


def test_deterministic():
    """Same input = identical output."""
    config = _fast_config()
    t1 = SimulationEngine(config).simulate("AAGPV")
    t2 = SimulationEngine(config).simulate("AAGPV")
    for s1, s2 in zip(t1.snapshots, t2.snapshots):
        np.testing.assert_allclose(s1.energy_total, s2.energy_total, rtol=1e-8)


def test_string_and_list_equivalent():
    """String and list[AminoAcid] input produce identical results."""
    config = _fast_config()
    t1 = SimulationEngine(config).simulate("ACD")
    t2 = SimulationEngine(config).simulate([AminoAcid.ALA, AminoAcid.CYS, AminoAcid.ASP])
    for s1, s2 in zip(t1.snapshots, t2.snapshots):
        np.testing.assert_allclose(s1.energy_total, s2.energy_total, rtol=1e-8)


def test_simulate_and_export():
    """PDB export works."""
    engine = SimulationEngine(_fast_config())
    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as f:
        path = f.name
    try:
        traj = engine.simulate_and_export("AAAAA", path)
        assert os.path.exists(path)
        with open(path) as f:
            assert 'ATOM' in f.read()
    finally:
        os.unlink(path)


def test_with_kinetics():
    """Translation kinetics mode works."""
    config = _fast_config(use_kinetics=True)
    traj = SimulationEngine(config).simulate("AAAAAA")
    assert traj.num_steps == 7  # 6 + equilibration
