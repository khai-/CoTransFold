#!/usr/bin/env python3
"""Self-improving optimization loop for CoTransFold.

Iteratively uses Claude Code CLI to analyze benchmark results, make targeted
code improvements, test, benchmark, and commit if improved.

All subprocess output is streamed live to stdout for full visibility.

Usage:
    python scripts/optimize_loop.py                     # 5 iterations
    python scripts/optimize_loop.py --iterations 3      # Custom count
    python scripts/optimize_loop.py --dry-run           # Show prompts only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'data' / 'benchmark_results'
LOG_FILE = PROJECT_ROOT / 'data' / 'optimization_log.json'

FOCUS_AREAS = [
    "Energy weight tuning — adjust w_ramachandran, w_hbond, w_vanderwaals, w_solvent, w_bonded "
    "in SimulationConfig defaults or in how they interact. Small changes (±0.1-0.3) can shift "
    "the balance between compaction and secondary structure.",

    "H-bond model improvements — the H-bond energy in src/cotransfold/energy/hbond.py uses a "
    "DSSP electrostatic model. Consider adjusting distance cutoffs, partial charges, or adding "
    "a distance-dependent weighting that better rewards alpha-helix i→i+4 patterns.",

    "VdW parameter tuning — adjust CA_EPSILON, CA_SIGMA, CB_EPSILON, LJ_CUTOFF in "
    "src/cotransfold/energy/vanderwaals.py and src/cotransfold/minimizer/analytical_gradient.py. "
    "The LJ well depth and equilibrium distances control compaction vs expansion balance.",

    "Solvent model refinement — adjust SOLVATION_PARAMS values, BURIAL_CUTOFF, or MAX_NEIGHBORS "
    "in src/cotransfold/energy/solvent.py. Stronger hydrophobic burial penalties drive compaction. "
    "Consider scaling solvation params by residue size.",

    "Minimizer convergence — adjust ftol, gtol, max_iterations, or the equilibration_steps "
    "in SimulationConfig. The minimizer may not be converging deeply enough. Also consider "
    "adjusting the helix_propensity_threshold for helix seeding.",
]


def run_streaming(cmd: list[str], timeout: int = 600,
                  cwd: Path | None = None, prefix: str = "  ") -> int:
    """Run a command with live streaming output. Returns exit code."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=cwd or PROJECT_ROOT, bufsize=1,
    )
    output_lines = []
    try:
        for line in proc.stdout:
            print(f"{prefix}{line}", end='', flush=True)
            output_lines.append(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"{prefix}[TIMEOUT after {timeout}s]", flush=True)
        return -1
    return proc.returncode


def run_quiet(cmd: list[str], timeout: int = 600,
              cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command quietly, capturing output."""
    return subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=timeout, cwd=cwd or PROJECT_ROOT,
    )


def run_benchmark(minimizer: str = 'fast', n_proteins: int = 6,
                  max_steps: int = 30) -> dict | None:
    """Run benchmark with live output, return results dict."""
    label = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cmd = [
        sys.executable, 'scripts/run_benchmark.py',
        '--save', label,
        '--skip-alphafold',
        '--max-steps', str(max_steps),
        '--minimizer', minimizer,
        '--n-proteins', str(n_proteins),
    ]

    rc = run_streaming(cmd, timeout=1200, prefix="  │ ")

    if rc != 0:
        print(f"  Benchmark failed (exit {rc})", flush=True)
        return None

    files = sorted(RESULTS_DIR.glob(f'benchmark_{label}_*.json'))
    if not files:
        print(f"  No results file found for label {label}", flush=True)
        return None

    with open(files[-1]) as f:
        return json.load(f)


def analyze_results(results: dict) -> dict:
    """Extract key metrics from benchmark results."""
    proteins = results.get('results', [])
    if not proteins:
        return {'avg_tm': 0, 'avg_rmsd': 0, 'weakest': None, 'best': None}

    tms = [p['ctf_tm'] for p in proteins]
    rmsds = [p['ctf_rmsd'] for p in proteins]

    weakest = min(proteins, key=lambda p: p['ctf_tm'])
    best = max(proteins, key=lambda p: p['ctf_tm'])

    return {
        'avg_tm': sum(tms) / len(tms),
        'avg_rmsd': sum(rmsds) / len(rmsds),
        'weakest': weakest,
        'best': best,
        'proteins': proteins,
    }


def print_results_table(analysis: dict):
    """Print a compact results table."""
    print(f"  ┌{'─' * 70}┐", flush=True)
    print(f"  │ {'Protein':<25} {'RMSD':>6} {'TM':>8} {'Helix':>6} {'Strand':>6} │", flush=True)
    print(f"  ├{'─' * 70}┤", flush=True)
    for p in analysis['proteins']:
        print(f"  │ {p['name']:<25} {p['ctf_rmsd']:5.1f}Å {p['ctf_tm']:8.4f} "
              f"{p.get('ctf_helix_frac', 0):5.0%} {p.get('ctf_strand_frac', 0):5.0%}  │", flush=True)
    print(f"  ├{'─' * 70}┤", flush=True)
    print(f"  │ {'Average':<25} {analysis['avg_rmsd']:5.1f}Å {analysis['avg_tm']:8.4f}"
          f"{'':>14} │", flush=True)
    print(f"  └{'─' * 70}┘", flush=True)


def build_prompt(results: dict, analysis: dict, iteration: int,
                 prev_diff: str) -> str:
    """Build the improvement prompt for Claude."""
    focus = FOCUS_AREAS[iteration % len(FOCUS_AREAS)]

    results_summary = ""
    for p in analysis['proteins']:
        results_summary += (
            f"  {p['name']:<25} RMSD={p['ctf_rmsd']:5.1f}Å  "
            f"TM={p['ctf_tm']:.4f}  Helix={p.get('ctf_helix_frac', 0):.0%}  "
            f"Strand={p.get('ctf_strand_frac', 0):.0%}  "
            f"SS={p.get('ctf_ss', 'N/A')}\n"
        )

    weakest = analysis['weakest']
    best = analysis['best']

    return f"""You are improving CoTransFold, a co-translational protein folding simulator.
The goal is to increase the average TM-score across the benchmark proteins.

Current benchmark results (iteration {iteration + 1}):
{results_summary}
Average TM-score: {analysis['avg_tm']:.4f}
Average RMSD: {analysis['avg_rmsd']:.1f}Å
Weakest protein: {weakest['name']} (TM={weakest['ctf_tm']:.4f}, RMSD={weakest['ctf_rmsd']:.1f}Å)
Best protein: {best['name']} (TM={best['ctf_tm']:.4f}, RMSD={best['ctf_rmsd']:.1f}Å)

Previous iteration changes: {prev_diff if prev_diff else "None (first iteration)"}

Focus area for this iteration: {focus}

Make ONE targeted code change to improve the average TM-score.

Rules:
- Edit existing files in src/cotransfold/ only
- Do not create new files
- Do not modify files in tests/ or scripts/
- Keep changes small and focused — one conceptual change
- Do not change the coordinate system, backbone representation, or benchmark infrastructure
- Make sure parameter changes are consistent across energy files AND analytical_gradient.py

After making changes, briefly explain what you changed and why in a single paragraph."""


def run_tests() -> bool:
    """Run pytest with live output, return True if all pass."""
    rc = run_streaming(
        [sys.executable, '-m', 'pytest', 'tests/', '-x', '-q', '--tb=short'],
        timeout=300, prefix="  │ ",
    )
    return rc == 0


def call_claude(prompt: str, dry_run: bool = False) -> tuple[str, int]:
    """Call Claude Code CLI with live streaming output.

    Returns (output_text, exit_code).
    """
    if dry_run:
        print(f"  [DRY RUN] Would send prompt ({len(prompt)} chars)", flush=True)
        return "[dry run]", 0

    print(f"  ┌─ Claude Code ─────────────────────────────────────────┐", flush=True)
    cmd = ['claude', '--print', '--dangerously-skip-permissions', '-p', prompt]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=PROJECT_ROOT, bufsize=1,
    )
    output_lines = []
    try:
        for line in proc.stdout:
            print(f"  │ {line}", end='', flush=True)
            output_lines.append(line)
        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  │ [TIMEOUT]", flush=True)
        return ''.join(output_lines), -1

    print(f"  └─────────────────────────────────────────────────────────┘", flush=True)
    return ''.join(output_lines), proc.returncode


def get_git_diff() -> str:
    """Get current uncommitted diff summary."""
    result = run_quiet(['git', 'diff', '--stat'])
    return result.stdout.strip()


def get_git_diff_full() -> str:
    """Get full diff for logging."""
    result = run_quiet(['git', 'diff'])
    return result.stdout[:3000]


def git_revert():
    """Revert all uncommitted changes."""
    run_quiet(['git', 'checkout', '--', '.'])
    print("  ✗ Reverted all changes.", flush=True)


def git_commit(iteration: int, avg_tm_before: float, avg_tm_after: float):
    """Commit the current changes."""
    msg = (
        f"optimize: iteration {iteration + 1} — "
        f"avg TM {avg_tm_before:.4f} → {avg_tm_after:.4f} "
        f"(+{(avg_tm_after - avg_tm_before) / max(avg_tm_before, 1e-10) * 100:.1f}%)"
    )
    run_quiet(['git', 'add', 'src/'])
    run_quiet(['git', 'commit', '-m', msg])
    print(f"  ✓ Committed: {msg}", flush=True)


def load_log() -> list:
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            return json.load(f)
    return []


def save_log(log: list):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='CoTransFold self-improving optimization loop')
    parser.add_argument('--iterations', '-n', type=int, default=5)
    parser.add_argument('--minimizer', default='fast', choices=['numpy', 'fast', 'jax'])
    parser.add_argument('--n-proteins', type=int, default=6)
    parser.add_argument('--max-steps', type=int, default=30)
    parser.add_argument('--dry-run', action='store_true', help='Show prompts without running Claude')
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  CoTransFold Self-Improving Optimization Loop")
    print(f"  Iterations: {args.iterations} | Minimizer: {args.minimizer}")
    print(f"  Proteins: {args.n_proteins} | Max steps: {args.max_steps}")
    print(f"{'═' * 60}\n", flush=True)

    log = load_log()
    prev_diff = ""

    for i in range(args.iterations):
        print(f"\n{'━' * 60}")
        print(f"  ITERATION {i + 1}/{args.iterations}")
        print(f"  Focus: {FOCUS_AREAS[i % len(FOCUS_AREAS)][:60]}...")
        print(f"{'━' * 60}\n", flush=True)
        t0 = time.time()

        # 1. Baseline benchmark
        print("  [1/5] Baseline benchmark", flush=True)
        baseline = run_benchmark(args.minimizer, args.n_proteins, args.max_steps)
        if baseline is None:
            print("  Skipping iteration — benchmark failed.", flush=True)
            continue
        baseline_analysis = analyze_results(baseline)
        print_results_table(baseline_analysis)

        # 2. Call Claude to make improvements
        print(f"\n  [2/5] Claude Code making improvements", flush=True)
        prompt = build_prompt(baseline, baseline_analysis, i, prev_diff)

        if args.dry_run:
            print(f"\n--- PROMPT ({len(prompt)} chars) ---\n{prompt}\n--- END ---", flush=True)
            continue

        claude_output, claude_rc = call_claude(prompt)
        if claude_rc != 0 and claude_rc != -1:
            print(f"  Claude exited with code {claude_rc}, skipping.", flush=True)
            continue

        # 3. Check changes
        diff = get_git_diff()
        if not diff:
            print("  No file changes detected, skipping.", flush=True)
            continue
        print(f"\n  Changes made:", flush=True)
        for line in diff.split('\n'):
            print(f"    {line}", flush=True)

        # 4. Run tests
        print(f"\n  [3/5] Running tests", flush=True)
        if not run_tests():
            print(f"\n  Tests FAILED — reverting.", flush=True)
            full_diff = get_git_diff_full()
            git_revert()
            prev_diff = f"(reverted: tests failed)\nDiff was:\n{full_diff[:500]}"
            log.append({
                'iteration': i + 1,
                'timestamp': datetime.now().isoformat(),
                'status': 'reverted_tests_failed',
                'baseline_avg_tm': baseline_analysis['avg_tm'],
                'diff_summary': diff,
            })
            save_log(log)
            continue

        # 5. Post-change benchmark
        print(f"\n  [4/5] Post-change benchmark", flush=True)
        after = run_benchmark(args.minimizer, args.n_proteins, args.max_steps)
        if after is None:
            print("  Post-change benchmark failed — reverting.", flush=True)
            git_revert()
            continue
        after_analysis = analyze_results(after)
        print_results_table(after_analysis)

        # 6. Compare and decide
        improved = after_analysis['avg_tm'] > baseline_analysis['avg_tm']
        delta = after_analysis['avg_tm'] - baseline_analysis['avg_tm']
        delta_pct = delta / max(baseline_analysis['avg_tm'], 1e-10) * 100

        print(f"\n  [5/5] Decision", flush=True)
        print(f"  Before: avg TM = {baseline_analysis['avg_tm']:.4f}  avg RMSD = {baseline_analysis['avg_rmsd']:.1f}Å", flush=True)
        print(f"  After:  avg TM = {after_analysis['avg_tm']:.4f}  avg RMSD = {after_analysis['avg_rmsd']:.1f}Å", flush=True)
        print(f"  Delta:  TM = {delta:+.4f} ({delta_pct:+.1f}%)", flush=True)

        full_diff = get_git_diff_full()

        if improved:
            git_commit(i, baseline_analysis['avg_tm'], after_analysis['avg_tm'])
            prev_diff = full_diff
            status = 'committed'
        else:
            git_revert()
            prev_diff = f"(reverted: TM regressed {delta:+.4f})\nDiff was:\n{full_diff[:500]}"
            status = 'reverted_regression'

        elapsed = time.time() - t0

        log.append({
            'iteration': i + 1,
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'baseline_avg_tm': baseline_analysis['avg_tm'],
            'after_avg_tm': after_analysis['avg_tm'],
            'delta_tm': delta,
            'delta_pct': delta_pct,
            'elapsed_s': elapsed,
            'focus': FOCUS_AREAS[i % len(FOCUS_AREAS)][:80],
            'diff_summary': diff,
            'claude_output': claude_output[:1000],
        })
        save_log(log)

        print(f"\n  Iteration time: {elapsed:.0f}s", flush=True)

    # Final summary
    print(f"\n{'═' * 60}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'═' * 60}")
    committed = [e for e in log if e.get('status') == 'committed']
    reverted = [e for e in log if 'reverted' in e.get('status', '')]
    print(f"  Committed: {len(committed)} | Reverted: {len(reverted)}")
    if committed:
        total_delta = sum(e['delta_tm'] for e in committed)
        print(f"  Total TM improvement: {total_delta:+.4f}")
    print(f"  Log: {LOG_FILE}")
    print(f"{'═' * 60}\n")


if __name__ == '__main__':
    main()
