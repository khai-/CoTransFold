#!/usr/bin/env python3
"""Run validation benchmarks on all test proteins."""

from __future__ import annotations

import time

from cotransfold.simulator.engine import SimulationEngine, SimulationConfig
from cotransfold.validation.benchmarks import ALL_BENCHMARKS, get_reference_ca
from cotransfold.validation.metrics import validate_structure
from cotransfold.structure.secondary import assign_secondary_structure, ss_summary


def main() -> None:
    print("CoTransFold Benchmark Suite")
    print("=" * 60)
    print()

    config = SimulationConfig(
        max_steps_per_residue=50,
        min_steps_per_residue=20,
        use_kinetics=False,
    )
    engine = SimulationEngine(config)

    results = []
    for bp in ALL_BENCHMARKS:
        print(f"Running: {bp.name} ({bp.pdb_id}, {len(bp.sequence)} residues)")
        print(f"  {bp.description}")

        t0 = time.time()
        traj = engine.simulate(bp.sequence)
        elapsed = time.time() - t0

        ref_ca = get_reference_ca(bp)
        result = validate_structure(traj.final_backbone, ref_ca, bp.expected_ss)
        ss = assign_secondary_structure(traj.final_backbone)

        print(f"  Time:      {elapsed:.1f}s")
        print(f"  CA RMSD:   {result.ca_rmsd:.2f} Å")
        print(f"  TM-score:  {result.tm_score:.3f}")
        print(f"  GDT-TS:    {result.gdt_ts:.3f}")
        print(f"  Expected:  {bp.expected_ss}")
        print(f"  Predicted: {ss}")
        print(f"  Helix:     {result.helix_fraction_pred*100:.1f}%")
        print(f"  Strand:    {result.strand_fraction_pred*100:.1f}%")
        print()

        results.append((bp, result, elapsed))

    # Summary table
    print("=" * 60)
    print(f"{'Protein':<20} {'Res':>4} {'RMSD':>7} {'TM':>6} {'GDT':>6} {'Time':>6}")
    print("-" * 60)
    for bp, result, elapsed in results:
        print(f"{bp.name:<20} {len(bp.sequence):>4} "
              f"{result.ca_rmsd:>6.2f}Å {result.tm_score:>6.3f} "
              f"{result.gdt_ts:>6.3f} {elapsed:>5.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
