"""Three-way comparison pipeline: CoTransFold vs AlphaFold vs Experimental.

For each benchmark protein:
1. Fetch experimental PDB structure (ground truth)
2. Fetch AlphaFold prediction (competitor)
3. Run CoTransFold simulation (our method)
4. Compare all three using RMSD, TM-score, GDT-TS
5. Store results for tracking over time
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from cotransfold.validation.benchmark_set import BenchmarkEntry, BENCHMARK_SET
from cotransfold.validation.pdb_parser import parse_pdb, PDBStructure
from cotransfold.validation.fetch import fetch_pdb, fetch_alphafold
from cotransfold.structure.rmsd import superposed_rmsd, tm_score
from cotransfold.validation.metrics import gdt_ts
from cotransfold.structure.coordinates import torsion_to_cartesian, get_ca_coords
from cotransfold.structure.secondary import assign_secondary_structure, helix_fraction, strand_fraction
from cotransfold.simulator.engine import SimulationEngine, SimulationConfig

RESULTS_DIR = Path(__file__).parents[3] / 'data' / 'benchmark_results'


@dataclass
class ComparisonResult:
    """Result of comparing CoTransFold vs AlphaFold vs experimental."""
    # Protein info
    name: str
    pdb_id: str
    category: str
    n_residues: int

    # CoTransFold vs Experimental
    ctf_rmsd: float
    ctf_tm: float
    ctf_gdt: float
    ctf_ss: str
    ctf_helix_frac: float
    ctf_strand_frac: float
    ctf_time_seconds: float

    # AlphaFold vs Experimental
    af_rmsd: float
    af_tm: float
    af_gdt: float
    af_ss: str
    af_helix_frac: float
    af_strand_frac: float

    # Experimental
    exp_ss: str
    exp_resolution: float

    # Meta
    timestamp: str = ''
    version: str = 'v0.1.0'

    def summary_line(self) -> str:
        return (
            f"{self.name:<25} "
            f"CTF: RMSD={self.ctf_rmsd:5.1f} TM={self.ctf_tm:.3f} | "
            f"AF:  RMSD={self.af_rmsd:5.1f} TM={self.af_tm:.3f} | "
            f"Δ TM={self.ctf_tm - self.af_tm:+.3f}"
        )


def compare_one(entry: BenchmarkEntry,
                engine: SimulationEngine,
                skip_alphafold: bool = False) -> ComparisonResult:
    """Run full comparison for one benchmark protein.

    Args:
        entry: benchmark entry
        engine: configured SimulationEngine
        skip_alphafold: if True, skip AlphaFold download (use zeros)

    Returns:
        ComparisonResult with all metrics
    """
    # 1. Fetch and parse experimental structure
    try:
        pdb_path = fetch_pdb(entry.pdb_id)
        exp_struct = parse_pdb(pdb_path, entry.chain_id)
        exp_ca = exp_struct.ca_coords
        exp_seq = exp_struct.sequence
    except Exception as e:
        print(f"  WARNING: Could not fetch PDB {entry.pdb_id}: {e}", flush=True)
        print(f"  Using sequence from benchmark entry instead", flush=True)
        exp_ca = None
        exp_seq = entry.sequence

    # 2. Fetch and parse AlphaFold prediction
    af_rmsd = af_tm = af_gdt = 0.0
    af_ss = ''
    af_helix = af_strand = 0.0

    if not skip_alphafold and entry.uniprot_id:
        try:
            af_path = fetch_alphafold(entry.uniprot_id)
            af_struct = parse_pdb(af_path, 'A')

            if exp_ca is not None and len(af_struct.ca_coords) > 0:
                # Align lengths
                n = min(len(exp_ca), len(af_struct.ca_coords))
                af_ca_trimmed = af_struct.ca_coords[:n]
                exp_ca_trimmed = exp_ca[:n]

                af_rmsd = superposed_rmsd(af_ca_trimmed, exp_ca_trimmed)
                af_tm = tm_score(af_ca_trimmed, exp_ca_trimmed)
                af_gdt = gdt_ts(af_ca_trimmed, exp_ca_trimmed)
        except Exception as e:
            print(f"  WARNING: Could not fetch AlphaFold for {entry.uniprot_id}: {e}", flush=True)

    # 3. Run CoTransFold simulation
    seq = exp_seq if exp_seq else entry.sequence
    t0 = time.time()
    trajectory = engine.simulate(seq)
    ctf_time = time.time() - t0

    # 4. Compare CoTransFold vs experimental
    ctf_rmsd = ctf_tm = ctf_gdt_val = 0.0
    ctf_ss = ''
    ctf_helix = ctf_strand = 0.0

    if trajectory.final_backbone:
        ctf_ss = assign_secondary_structure(trajectory.final_backbone)
        ctf_helix = helix_fraction(ctf_ss)
        ctf_strand = strand_fraction(ctf_ss)

        if exp_ca is not None and len(exp_ca) > 0:
            pred_coords = torsion_to_cartesian(trajectory.final_backbone)
            pred_ca = get_ca_coords(pred_coords)
            n = min(len(pred_ca), len(exp_ca))
            pred_ca_trimmed = pred_ca[:n]
            exp_ca_trimmed = exp_ca[:n]

            ctf_rmsd = superposed_rmsd(pred_ca_trimmed, exp_ca_trimmed)
            ctf_tm = tm_score(pred_ca_trimmed, exp_ca_trimmed)
            ctf_gdt_val = gdt_ts(pred_ca_trimmed, exp_ca_trimmed)

    # 5. Experimental SS (rough estimate from expected_ss or blank)
    exp_ss = entry.expected_ss

    return ComparisonResult(
        name=entry.name,
        pdb_id=entry.pdb_id,
        category=entry.category,
        n_residues=len(seq),
        ctf_rmsd=ctf_rmsd, ctf_tm=ctf_tm, ctf_gdt=ctf_gdt_val,
        ctf_ss=ctf_ss, ctf_helix_frac=ctf_helix, ctf_strand_frac=ctf_strand,
        ctf_time_seconds=ctf_time,
        af_rmsd=af_rmsd, af_tm=af_tm, af_gdt=af_gdt,
        af_ss=af_ss, af_helix_frac=af_helix, af_strand_frac=af_strand,
        exp_ss=exp_ss, exp_resolution=entry.resolution,
        timestamp=datetime.now().isoformat(),
    )


def run_full_benchmark(config: SimulationConfig | None = None,
                       categories: list[str] | None = None,
                       skip_alphafold: bool = False,
                       n_proteins: int = 0,
                       ) -> list[ComparisonResult]:
    """Run comparison on all (or selected) benchmark proteins.

    Args:
        config: SimulationConfig (default settings if None)
        categories: only run these categories (None = all)
        skip_alphafold: skip AlphaFold downloads

    Returns:
        List of ComparisonResult
    """
    if config is None:
        config = SimulationConfig(
            max_steps_per_residue=50,
            min_steps_per_residue=20,
            use_kinetics=False,
        )
    engine = SimulationEngine(config)

    entries = BENCHMARK_SET
    if categories:
        entries = [e for e in entries if e.category in categories]
    if n_proteins > 0:
        entries = entries[:n_proteins]

    results = []
    for i, entry in enumerate(entries, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(entries)}] {entry.name} ({entry.pdb_id}, {entry.n_residues} res)")
        print(f"Category: {entry.category} | Fold: {entry.fold_class}")
        print(f"{'='*60}", flush=True)

        result = compare_one(entry, engine, skip_alphafold=skip_alphafold)
        results.append(result)
        print(f"  {result.summary_line()}", flush=True)

    return results


def save_results(results: list[ComparisonResult], label: str = '') -> str:
    """Save benchmark results to JSON for tracking over time.

    Args:
        results: list of ComparisonResult
        label: optional label (e.g., version number)

    Returns:
        Path to saved file
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"benchmark_{label}_{timestamp}.json" if label else f"benchmark_{timestamp}.json"
    filepath = RESULTS_DIR / filename

    data = {
        'timestamp': datetime.now().isoformat(),
        'label': label,
        'n_proteins': len(results),
        'results': [asdict(r) for r in results],
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return str(filepath)


def print_summary_table(results: list[ComparisonResult]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"{'Protein':<22} {'Res':>4} {'CTF RMSD':>8} {'CTF TM':>7} {'AF RMSD':>8} {'AF TM':>7} {'Δ TM':>7} {'Time':>6}")
    print(f"{'-'*90}")

    for r in results:
        delta = r.ctf_tm - r.af_tm if r.af_tm > 0 else 0.0
        marker = " ★" if delta > 0 else ""
        print(
            f"{r.name:<22} {r.n_residues:>4} "
            f"{r.ctf_rmsd:>7.1f}Å {r.ctf_tm:>7.3f} "
            f"{r.af_rmsd:>7.1f}Å {r.af_tm:>7.3f} "
            f"{delta:>+7.3f} {r.ctf_time_seconds:>5.1f}s{marker}"
        )

    print(f"{'='*90}")

    # Category averages
    categories = sorted(set(r.category for r in results))
    print(f"\nCategory Averages:")
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        avg_ctf_tm = np.mean([r.ctf_tm for r in cat_results])
        avg_af_tm = np.mean([r.af_tm for r in cat_results])
        print(f"  {cat:<25} CTF TM={avg_ctf_tm:.3f}  AF TM={avg_af_tm:.3f}")
