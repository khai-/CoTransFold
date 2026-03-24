"""Tests for the co-translational folding simulation engine."""

from __future__ import annotations

import os
import tempfile
import numpy as np

from cotransfold.simulator.engine import SimulationEngine, SimulationConfig
from cotransfold.core.residue import AminoAcid


def test_simulate_short_sequence():
    """Simulate a 10-residue poly-alanine."""
    config = SimulationConfig(
        max_steps_per_residue=20,  # Fast for testing
        use_kinetics=False,
    )
    engine = SimulationEngine(config)
    trajectory = engine.simulate("AAAAAAAAAA")

    assert trajectory.num_steps == 10
    assert trajectory.final_backbone is not None
    assert trajectory.final_backbone.num_residues == 10


def test_trajectory_has_correct_snapshots():
    """Each step should record a snapshot."""
    config = SimulationConfig(max_steps_per_residue=10, use_kinetics=False)
    engine = SimulationEngine(config)
    trajectory = engine.simulate("AAGPV")

    assert trajectory.num_steps == 5
    for i, snap in enumerate(trajectory.snapshots):
        assert snap.step == i
        assert snap.chain_length == i + 1


def test_exposed_count_increases():
    """As chain grows, more residues should become exposed."""
    config = SimulationConfig(max_steps_per_residue=10, use_kinetics=False)
    engine = SimulationEngine(config)
    # 30 residues: first ones should start exiting ~26-residue E. coli tunnel
    trajectory = engine.simulate("A" * 30)

    exposed = trajectory.exposed_trace
    # Last snapshot should have some exposed residues
    # 30 * 3.5 = 105Å from PTC for first residue, tunnel is 90Å
    assert exposed[-1] > 0, "Some residues should be exposed after 30 steps"
    # Exposed count should be non-decreasing
    for i in range(1, len(exposed)):
        assert exposed[i] >= exposed[i - 1], (
            f"Exposed count should not decrease: step {i-1}={exposed[i-1]}, step {i}={exposed[i]}")


def test_deterministic():
    """Same input should produce identical output."""
    config = SimulationConfig(max_steps_per_residue=15, use_kinetics=False)

    engine1 = SimulationEngine(config)
    traj1 = engine1.simulate("AAAAAGAAAA")

    engine2 = SimulationEngine(config)
    traj2 = engine2.simulate("AAAAAGAAAA")

    for s1, s2 in zip(traj1.snapshots, traj2.snapshots):
        np.testing.assert_allclose(s1.energy_total, s2.energy_total, rtol=1e-8)
        np.testing.assert_allclose(s1.backbone.phi, s2.backbone.phi, atol=1e-8)


def test_no_tunnel_mode():
    """Simulation should work without tunnel constraints."""
    config = SimulationConfig(
        use_tunnel=False,
        max_steps_per_residue=10,
        use_kinetics=False,
    )
    engine = SimulationEngine(config)
    trajectory = engine.simulate("AAAAA")
    assert trajectory.num_steps == 5


def test_energy_decomposition():
    """Each snapshot should have energy decomposition."""
    config = SimulationConfig(max_steps_per_residue=10, use_kinetics=False)
    engine = SimulationEngine(config)
    trajectory = engine.simulate("AAAAAAAAAA")

    last = trajectory.snapshots[-1]
    assert "ramachandran" in last.energy_decomposed
    assert "hbond" in last.energy_decomposed


def test_simulate_and_export():
    """simulate_and_export should produce a PDB file."""
    config = SimulationConfig(max_steps_per_residue=10, use_kinetics=False)
    engine = SimulationEngine(config)

    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as f:
        path = f.name

    try:
        trajectory = engine.simulate_and_export("AAAAA", path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert 'ATOM' in content
        assert trajectory.num_steps == 5
    finally:
        os.unlink(path)


def test_string_and_list_sequence_equivalent():
    """String and list[AminoAcid] input should produce identical results."""
    config = SimulationConfig(max_steps_per_residue=10, use_kinetics=False)

    engine1 = SimulationEngine(config)
    traj1 = engine1.simulate("ACDE")

    engine2 = SimulationEngine(config)
    traj2 = engine2.simulate([AminoAcid.ALA, AminoAcid.CYS, AminoAcid.ASP, AminoAcid.GLU])

    for s1, s2 in zip(traj1.snapshots, traj2.snapshots):
        np.testing.assert_allclose(s1.energy_total, s2.energy_total, rtol=1e-8)


def test_trajectory_summary():
    """Summary should be a readable string."""
    config = SimulationConfig(max_steps_per_residue=10, use_kinetics=False)
    engine = SimulationEngine(config)
    trajectory = engine.simulate("AAAAA")
    summary = trajectory.summary()
    assert "CoTransFold" in summary
    assert "5 residues" in summary


def test_with_kinetics():
    """Simulation with translation kinetics should work."""
    config = SimulationConfig(max_steps_per_residue=15)
    engine = SimulationEngine(config)
    trajectory = engine.simulate("AAAAAA")
    assert trajectory.num_steps == 6
