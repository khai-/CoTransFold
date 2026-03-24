#!/usr/bin/env python3
"""Run the full benchmark suite: CoTransFold vs AlphaFold vs Experimental.

Usage:
    python scripts/run_benchmark.py                    # Run all
    python scripts/run_benchmark.py --category ultra_small  # One category
    python scripts/run_benchmark.py --skip-alphafold   # Skip AF downloads
    python scripts/run_benchmark.py --list             # List benchmark set
"""

from __future__ import annotations

import argparse
import sys

from cotransfold.simulator.engine import SimulationConfig
from cotransfold.validation.benchmark_set import BENCHMARK_SET, summary_table, get_all_categories
from cotransfold.validation.compare import (
    run_full_benchmark, save_results, print_summary_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='CoTransFold Benchmark Suite — Compare against experimental structures and AlphaFold')
    parser.add_argument('--category', '-c', help='Run only this category')
    parser.add_argument('--skip-alphafold', action='store_true',
                        help='Skip AlphaFold downloads')
    parser.add_argument('--list', action='store_true',
                        help='List all benchmark proteins and exit')
    parser.add_argument('--max-steps', type=int, default=50,
                        help='Max minimization steps per residue (default: 50)')
    parser.add_argument('--save', '-s', default='',
                        help='Label for saving results (e.g., v0.1.0)')
    parser.add_argument('--no-tunnel', action='store_true')
    parser.add_argument('--no-chaperones', action='store_true')

    args = parser.parse_args()

    if args.list:
        print("CoTransFold Benchmark Set")
        print("=" * 75)
        print(summary_table())
        print(f"\nTotal: {len(BENCHMARK_SET)} proteins")
        print(f"Categories: {', '.join(get_all_categories())}")
        return

    print("CoTransFold Benchmark Suite")
    print("=" * 60)

    config = SimulationConfig(
        max_steps_per_residue=args.max_steps,
        min_steps_per_residue=min(20, args.max_steps),
        use_kinetics=False,
        use_tunnel=not args.no_tunnel,
        use_chaperones=not args.no_chaperones,
    )

    categories = [args.category] if args.category else None

    results = run_full_benchmark(
        config=config,
        categories=categories,
        skip_alphafold=args.skip_alphafold,
    )

    print_summary_table(results)

    if args.save:
        save_results(results, label=args.save)


if __name__ == '__main__':
    main()
