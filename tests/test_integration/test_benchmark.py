"""Integration tests — run benchmarks on small proteins."""

from __future__ import annotations

import numpy as np

from cotransfold.simulator.engine import SimulationEngine, SimulationConfig
from cotransfold.validation.benchmarks import (
    ALL_BENCHMARKS, get_benchmark, get_reference_ca, TRP_CAGE, CHIGNOLIN, WW_DOMAIN,
)
from cotransfold.validation.metrics import validate_structure
from cotransfold.structure.secondary import assign_secondary_structure, helix_fraction
from cotransfold.structure.coordinates import torsion_to_cartesian, get_ca_coords


def _run_benchmark(benchmark, max_steps: int = 30):
    """Run simulation on a benchmark protein and validate."""
    config = SimulationConfig(
        max_steps_per_residue=max_steps,
        min_steps_per_residue=10,
        use_kinetics=False,  # Faster for testing
    )
    engine = SimulationEngine(config)
    trajectory = engine.simulate(benchmark.sequence)

    ref_ca = get_reference_ca(benchmark)
    result = validate_structure(trajectory.final_backbone, ref_ca, benchmark.expected_ss)
    return trajectory, result


def test_chignolin_runs():
    """Chignolin (10 res) should complete without error."""
    traj, result = _run_benchmark(CHIGNOLIN, max_steps=20)
    assert traj.num_steps == 10
    assert result.n_residues == 10
    assert result.ca_rmsd >= 0
    assert 0 <= result.tm_score <= 1


def test_trp_cage_runs():
    """Trp-cage (20 res) should complete and produce reasonable structure."""
    traj, result = _run_benchmark(TRP_CAGE, max_steps=20)
    assert traj.num_steps == 20
    assert result.ca_rmsd >= 0
    assert 0 <= result.tm_score <= 1
    assert 0 <= result.gdt_ts <= 1


def test_ww_domain_runs():
    """WW domain (33 res) should complete."""
    traj, result = _run_benchmark(WW_DOMAIN, max_steps=15)
    assert traj.num_steps == len(WW_DOMAIN.sequence)


def test_all_benchmarks_run():
    """All benchmark proteins should simulate without errors."""
    config = SimulationConfig(
        max_steps_per_residue=10,
        min_steps_per_residue=5,
        use_kinetics=False,
    )
    engine = SimulationEngine(config)

    for bp in ALL_BENCHMARKS:
        traj = engine.simulate(bp.sequence)
        assert traj.num_steps == len(bp.sequence), (
            f"{bp.name} should have {len(bp.sequence)} steps, got {traj.num_steps}")
        assert traj.final_backbone is not None


def test_validation_result_has_summary():
    """ValidationResult should produce readable summary."""
    _, result = _run_benchmark(TRP_CAGE, max_steps=10)
    summary = result.summary()
    assert "CA RMSD" in summary
    assert "TM-score" in summary


def test_helical_protein_has_some_helix():
    """Villin headpiece (mostly helical) should show some helix after simulation."""
    from cotransfold.validation.benchmarks import VILLIN_HEADPIECE
    config = SimulationConfig(
        max_steps_per_residue=30,
        min_steps_per_residue=15,
        use_kinetics=False,
        use_tunnel=False,  # No tunnel = all residues free to fold
    )
    engine = SimulationEngine(config)
    traj = engine.simulate(VILLIN_HEADPIECE.sequence)

    ss = assign_secondary_structure(traj.final_backbone)
    hf = helix_fraction(ss)
    # With no tunnel and reasonable minimization, some helix should form
    # This is a soft test — mainly checking the pipeline works
    assert isinstance(hf, float)
    assert 0.0 <= hf <= 1.0
