"""Replica Exchange Monte Carlo (REMC) with staged scoring and decoy clustering.

Runs multiple replicas at different temperatures. High-temperature replicas
explore broadly; periodic swaps allow low-temperature replicas to escape
local minima.

Key improvements over basic REMC:
1. Staged scoring: progressively introduce energy terms (Rosetta-style)
2. Decoy clustering: return centroid of largest cluster, not lowest energy
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.chain import NascentChain
from cotransfold.core.conformation import BackboneState
from cotransfold.energy.total import TotalEnergy
from cotransfold.sampling.fragments import FragmentLibrary
from cotransfold.sampling.mc_moves import random_mc_move


# Staged scoring weights (Rosetta-inspired)
# Each stage is a dict of weight multipliers applied to the base weights
SCORING_STAGES = [
    {   # Stage 1: smooth landscape — steric + solvation + Rg + dipole
        'ramachandran': 0.0, 'hbond': 0.0, 'vanderwaals': 1.0,
        'solvent': 1.0, 'rg_restraint': 1.0, 'pair_potential': 0.5,
        'sheet_pairing': 0.0, 'torsion_coupling': 0.0, 'bonded': 0.5,
        'backbone_dipole': 0.5,
    },
    {   # Stage 2: add pair potential + sheet pairing + dipole
        'ramachandran': 0.0, 'hbond': 0.3, 'vanderwaals': 1.0,
        'solvent': 1.0, 'rg_restraint': 1.0, 'pair_potential': 1.0,
        'sheet_pairing': 0.5, 'torsion_coupling': 0.3, 'bonded': 0.5,
        'backbone_dipole': 0.8,
    },
    {   # Stage 3: add H-bonds + full dipole + torsion coupling
        'ramachandran': 0.2, 'hbond': 0.7, 'vanderwaals': 1.0,
        'solvent': 1.0, 'rg_restraint': 1.0, 'pair_potential': 1.0,
        'sheet_pairing': 1.0, 'torsion_coupling': 0.7, 'bonded': 1.0,
        'backbone_dipole': 1.0,
    },
    {   # Stage 4: full energy
        'ramachandran': 1.0, 'hbond': 1.0, 'vanderwaals': 1.0,
        'solvent': 1.0, 'rg_restraint': 1.0, 'pair_potential': 1.0,
        'sheet_pairing': 1.0, 'torsion_coupling': 1.0, 'bonded': 1.0,
        'backbone_dipole': 1.0,
    },
]


def _geometric_temperatures(t_low: float, t_high: float,
                            n_replicas: int) -> np.ndarray:
    return np.geomspace(t_low, t_high, n_replicas)


def _apply_staged_weights(energy_fn: TotalEnergy, stage_multipliers: dict,
                          base_weights: dict) -> None:
    """Temporarily rescale energy term weights for staged scoring."""
    for term, weight in energy_fn._terms:
        name = term.name
        if name in stage_multipliers and name in base_weights:
            # Modify the weight in-place (will be restored later)
            pass  # We handle this via energy_with_stage instead


def _compute_staged_energy(energy_fn: TotalEnergy, chain: NascentChain,
                           stage_mult: dict, **kwargs) -> float:
    """Compute energy with staged weight multipliers."""
    from cotransfold.structure.coordinates import torsion_to_cartesian

    if chain.chain_length == 0:
        return 0.0
    coords = torsion_to_cartesian(chain.backbone)
    total = 0.0
    for term, base_weight in energy_fn._terms:
        mult = stage_mult.get(term.name, 1.0)
        if mult == 0.0:
            continue
        e = term.compute(coords, chain.backbone, chain.sequence, **kwargs)
        total += base_weight * mult * e
    return total


def _torsion_distance(bb1: BackboneState, bb2: BackboneState) -> float:
    """Compute torsion-space RMSD between two backbone states."""
    d_phi = bb1.phi - bb2.phi
    d_phi = d_phi - 2 * np.pi * np.round(d_phi / (2 * np.pi))
    d_psi = bb1.psi - bb2.psi
    d_psi = d_psi - 2 * np.pi * np.round(d_psi / (2 * np.pi))
    return float(np.sqrt(np.mean(d_phi**2 + d_psi**2)))


def _cluster_decoys(decoys: list[BackboneState],
                    energies: list[float]) -> BackboneState:
    """SPICKER-like clustering: return centroid of largest cluster.

    Falls back to lowest energy if clustering fails.
    """
    n = len(decoys)
    if n <= 3:
        best_idx = int(np.argmin(energies))
        return decoys[best_idx]

    # Compute pairwise torsion distances
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _torsion_distance(decoys[i], decoys[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Find cutoff for largest cluster containing 15-70% of decoys
    best_cutoff = np.radians(60.0)  # default
    for cutoff in np.arange(np.radians(20), np.radians(90), np.radians(5)):
        neighbors = np.sum(dist_matrix < cutoff, axis=1)
        max_frac = np.max(neighbors) / n
        if 0.15 <= max_frac <= 0.70:
            best_cutoff = cutoff
            break

    # Find densest point
    neighbors = np.sum(dist_matrix < best_cutoff, axis=1)
    center_idx = int(np.argmax(neighbors))

    # Cluster members
    cluster = np.where(dist_matrix[center_idx] < best_cutoff)[0]

    if len(cluster) < 2:
        best_idx = int(np.argmin(energies))
        return decoys[best_idx]

    # Centroid: circular mean of torsion angles
    n_res = decoys[0].num_residues
    sin_phi = np.zeros(n_res)
    cos_phi = np.zeros(n_res)
    sin_psi = np.zeros(n_res)
    cos_psi = np.zeros(n_res)

    for idx in cluster:
        sin_phi += np.sin(decoys[idx].phi)
        cos_phi += np.cos(decoys[idx].phi)
        sin_psi += np.sin(decoys[idx].psi)
        cos_psi += np.cos(decoys[idx].psi)

    centroid = decoys[center_idx].copy()
    centroid.phi = np.arctan2(sin_phi, cos_phi)
    centroid.psi = np.arctan2(sin_psi, cos_psi)

    return centroid


def run_remc(chain: NascentChain,
             energy_fn: TotalEnergy,
             frag_lib: FragmentLibrary,
             n_replicas: int = 8,
             n_cycles: int = 100,
             mc_steps_per_cycle: int = 30,
             t_low: float = 0.5,
             t_high: float = 5.0,
             energy_kwargs: dict | None = None,
             seed: int = 42,
             ) -> BackboneState:
    """Run REMC with staged scoring and decoy clustering.

    Returns:
        Best backbone state (centroid of largest decoy cluster)
    """
    if energy_kwargs is None:
        energy_kwargs = {}

    rng = np.random.RandomState(seed)
    temps = _geometric_temperatures(t_low, t_high, n_replicas)
    n = chain.chain_length

    # Stage boundaries (4 stages)
    stage_boundaries = [
        n_cycles // 4,
        n_cycles // 2,
        3 * n_cycles // 4,
        n_cycles,
    ]

    # Initialize replicas
    replicas = []
    energies = np.zeros(n_replicas)
    for k in range(n_replicas):
        bb = chain.backbone.copy()
        if k > 0:
            bb.phi += rng.randn(n) * np.radians(10 * k)
            bb.psi += rng.randn(n) * np.radians(10 * k)
            bb.phi = np.arctan2(np.sin(bb.phi), np.cos(bb.phi))
            bb.psi = np.arctan2(np.sin(bb.psi), np.cos(bb.psi))
        replicas.append(bb)

        chain.backbone.phi[:] = bb.phi
        chain.backbone.psi[:] = bb.psi
        chain.backbone.omega[:] = bb.omega
        energies[k] = _compute_staged_energy(
            energy_fn, chain, SCORING_STAGES[0], **energy_kwargs)

    # Decoy collection (from lowest-T replica)
    decoys = []
    decoy_energies = []
    decoy_interval = max(1, n_cycles // 30)  # Collect ~30 decoys

    best_energy = energies[0]
    best_backbone = replicas[0].copy()

    # Pre-compute stage index for each cycle
    stage_for_cycle = np.zeros(n_cycles, dtype=int)
    for c in range(n_cycles):
        for si, boundary in enumerate(stage_boundaries):
            if c < boundary:
                stage_for_cycle[c] = si
                break

    for cycle in range(n_cycles):
        stage_idx = stage_for_cycle[cycle]
        stage_mult = SCORING_STAGES[stage_idx]
        move_stage = 'early' if stage_idx < 2 else 'late'

        # Run independent MC on each replica
        for k in range(n_replicas):
            T = temps[k]

            for _ in range(mc_steps_per_cycle):
                # Save only the angles (avoid full copy)
                old_phi = replicas[k].phi.copy()
                old_psi = replicas[k].psi.copy()

                # Apply move directly to replica
                chain.backbone.phi[:] = replicas[k].phi
                chain.backbone.psi[:] = replicas[k].psi
                chain.backbone.omega[:] = replicas[k].omega

                random_mc_move(chain.backbone, frag_lib, stage=move_stage, rng=rng)

                replicas[k].phi[:] = chain.backbone.phi
                replicas[k].psi[:] = chain.backbone.psi

                # Staged energy
                new_energy = _compute_staged_energy(
                    energy_fn, chain, stage_mult, **energy_kwargs)

                # Metropolis criterion
                delta_e = new_energy - energies[k]
                if delta_e < 0 or rng.random() < np.exp(-min(delta_e / max(T, 1e-10), 500)):
                    energies[k] = new_energy
                else:
                    replicas[k].phi[:] = old_phi
                    replicas[k].psi[:] = old_psi

        # Replica swaps
        if cycle % 2 == 0:
            pairs = range(0, n_replicas - 1, 2)
        else:
            pairs = range(1, n_replicas - 1, 2)

        for i in pairs:
            j = i + 1
            beta_i = 1.0 / max(temps[i], 1e-10)
            beta_j = 1.0 / max(temps[j], 1e-10)
            delta = (beta_i - beta_j) * (energies[j] - energies[i])

            if delta <= 0 or rng.random() < np.exp(-delta):
                replicas[i], replicas[j] = replicas[j], replicas[i]
                energies[i], energies[j] = energies[j], energies[i]

        # Track best and collect decoys
        if energies[0] < best_energy:
            best_energy = energies[0]
            best_backbone = replicas[0].copy()

        # Collect decoys from lowest-T replica (in later stages)
        if stage_idx >= 2 and cycle % decoy_interval == 0:
            decoys.append(replicas[0].copy())
            decoy_energies.append(energies[0])

    # Cluster decoys and return centroid
    if len(decoys) >= 5:
        return _cluster_decoys(decoys, decoy_energies)
    else:
        return best_backbone
