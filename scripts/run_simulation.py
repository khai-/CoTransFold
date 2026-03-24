#!/usr/bin/env python3
"""CLI entry point for CoTransFold simulations.

Usage:
    python scripts/run_simulation.py SEQUENCE [options]

Examples:
    python scripts/run_simulation.py AAAAAAAAAAAAAAAAAAAA
    python scripts/run_simulation.py AAAAAAAAAAAAAAAAAAAA --organism ecoli --output result.pdb
    python scripts/run_simulation.py --fasta input.fasta --output result.pdb
"""

from __future__ import annotations

import argparse
import sys
import time

from cotransfold.simulator.engine import SimulationEngine, SimulationConfig
from cotransfold.core.trajectory import StepSnapshot
from cotransfold.structure.coordinates import torsion_to_cartesian
from cotransfold.structure.pdb_io import write_pdb
from cotransfold.core.residue import sequence_from_string


def progress_callback(snapshot: StepSnapshot) -> None:
    """Print progress during simulation."""
    marker = " [RARE CODON]" if snapshot.is_rare_codon else ""
    print(
        f"  Step {snapshot.step + 1:4d} | "
        f"length={snapshot.chain_length:3d} | "
        f"exposed={snapshot.num_exposed:3d} | "
        f"E={snapshot.energy_total:8.2f} | "
        f"iters={snapshot.minimization_iterations:3d}"
        f"{marker}"
    )


def read_fasta(path: str) -> str:
    """Read first sequence from a FASTA file."""
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq_lines:
                    break  # Only read first sequence
                continue
            seq_lines.append(line)
    return ''.join(seq_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='CoTransFold: Ribosome-aware co-translational protein folding simulator')
    parser.add_argument('sequence', nargs='?', help='Amino acid sequence (one-letter code)')
    parser.add_argument('--fasta', help='Input FASTA file')
    parser.add_argument('--organism', default='ecoli', choices=['ecoli', 'yeast', 'human'])
    parser.add_argument('--output', '-o', default='output.pdb', help='Output PDB file')
    parser.add_argument('--no-tunnel', action='store_true', help='Disable tunnel constraints')
    parser.add_argument('--no-kinetics', action='store_true', help='Disable translation kinetics')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    # Get sequence
    if args.fasta:
        seq = read_fasta(args.fasta)
    elif args.sequence:
        seq = args.sequence
    else:
        parser.error('Provide a sequence or --fasta file')
        return

    print(f"CoTransFold Simulator")
    print(f"{'=' * 50}")
    print(f"Sequence: {seq[:60]}{'...' if len(seq) > 60 else ''}")
    print(f"Length:   {len(seq)} residues")
    print(f"Organism: {args.organism}")
    print(f"Tunnel:   {'ON' if not args.no_tunnel else 'OFF'}")
    print(f"Kinetics: {'ON' if not args.no_kinetics else 'OFF'}")
    print()

    config = SimulationConfig(
        organism=args.organism,
        use_tunnel=not args.no_tunnel,
        use_kinetics=not args.no_kinetics,
    )
    engine = SimulationEngine(config)

    callback = None if args.quiet else progress_callback

    print("Running simulation...")
    t0 = time.time()
    trajectory = engine.simulate(seq, callback=callback)
    elapsed = time.time() - t0

    print()
    print(trajectory.summary())
    print(f"\nWall time: {elapsed:.1f}s")

    # Export PDB
    if trajectory.final_backbone:
        aa_seq = sequence_from_string(seq)
        coords = torsion_to_cartesian(trajectory.final_backbone)
        write_pdb(args.output, coords, aa_seq)
        print(f"Structure written to: {args.output}")


if __name__ == '__main__':
    main()
