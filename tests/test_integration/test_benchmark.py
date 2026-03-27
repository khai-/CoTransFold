"""Integration tests — verify the full pipeline works end-to-end.

Kept lean: one small protein simulation + validation pipeline checks.
Full benchmarks are run via scripts/run_benchmark.py, not in tests.
"""

from __future__ import annotations

import numpy as np

from cotransfold.simulator.engine import SimulationEngine, SimulationConfig
from cotransfold.validation.benchmarks import (
    get_benchmark, get_reference_ca, CHIGNOLIN,
)
from cotransfold.validation.metrics import validate_structure
from cotransfold.structure.secondary import assign_secondary_structure, helix_fraction


def test_end_to_end_simulation():
    """Full pipeline: simulate chignolin (10 res) and validate."""
    config = SimulationConfig(
        max_steps_per_residue=10,
        min_steps_per_residue=5,
        equilibration_steps=50,
        use_kinetics=False,
    )
    engine = SimulationEngine(config)
    traj = engine.simulate(CHIGNOLIN.sequence)

    # 10 translation + 1 equilibration
    assert traj.num_steps == 11
    assert traj.final_backbone is not None
    assert traj.final_backbone.num_residues == 10

    # Validate against reference
    ref_ca = get_reference_ca(CHIGNOLIN)
    result = validate_structure(traj.final_backbone, ref_ca)
    assert result.ca_rmsd >= 0
    assert 0 <= result.tm_score <= 1
    assert 0 <= result.gdt_ts <= 1
    assert "CA RMSD" in result.summary()


def test_no_tunnel_mode():
    """Simulation works without tunnel."""
    config = SimulationConfig(
        use_tunnel=False, max_steps_per_residue=5,
        equilibration_steps=20, use_kinetics=False,
    )
    engine = SimulationEngine(config)
    traj = engine.simulate("AAAAA")
    assert traj.num_steps == 6  # 5 + eq
    assert traj.final_backbone.num_residues == 5


def test_secondary_structure_assignment():
    """SS assignment works on simulation output."""
    config = SimulationConfig(
        max_steps_per_residue=5, equilibration_steps=20, use_kinetics=False,
    )
    engine = SimulationEngine(config)
    traj = engine.simulate("AAAAAAAAAA")
    ss = assign_secondary_structure(traj.final_backbone)
    hf = helix_fraction(ss)
    assert len(ss) == 10
    assert 0.0 <= hf <= 1.0
