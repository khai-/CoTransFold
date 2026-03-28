"""Replica Exchange Monte Carlo (REMC) for protein structure sampling.

Runs multiple replicas at different temperatures in parallel (serial in this
implementation). High-temperature replicas explore broadly; periodic swaps
between adjacent replicas allow low-temperature replicas to escape local minima.

This is the proven method for ab initio protein structure prediction,
used by QUARK, I-TASSER, and Rosetta.
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.chain import NascentChain
from cotransfold.core.conformation import BackboneState
from cotransfold.energy.total import TotalEnergy
from cotransfold.sampling.fragments import FragmentLibrary
from cotransfold.sampling.mc_moves import random_mc_move


def _geometric_temperatures(t_low: float, t_high: float,
                            n_replicas: int) -> np.ndarray:
    """Generate geometrically spaced temperature ladder."""
    return np.geomspace(t_low, t_high, n_replicas)


def run_remc(chain: NascentChain,
             energy_fn: TotalEnergy,
             frag_lib: FragmentLibrary,
             n_replicas: int = 8,
             n_cycles: int = 50,
             mc_steps_per_cycle: int = 20,
             t_low: float = 0.5,
             t_high: float = 5.0,
             energy_kwargs: dict | None = None,
             seed: int = 42,
             ) -> BackboneState:
    """Run Replica Exchange Monte Carlo.

    Args:
        chain: nascent chain (backbone will be modified)
        energy_fn: total energy function
        frag_lib: fragment library for MC moves
        n_replicas: number of temperature replicas
        n_cycles: number of REMC cycles (each has MC steps + swap attempts)
        mc_steps_per_cycle: MC steps per replica per cycle
        t_low: lowest temperature
        t_high: highest temperature
        energy_kwargs: extra kwargs for energy function
        seed: random seed

    Returns:
        Best backbone state found (lowest energy at lowest temperature)
    """
    if energy_kwargs is None:
        energy_kwargs = {}

    rng = np.random.RandomState(seed)
    temps = _geometric_temperatures(t_low, t_high, n_replicas)

    n = chain.chain_length

    # Initialize replicas: copies of current backbone
    replicas = []
    energies = np.zeros(n_replicas)
    for k in range(n_replicas):
        bb = chain.backbone.copy()
        # Perturb non-lowest replicas for diversity
        if k > 0:
            bb.phi += rng.randn(n) * np.radians(10 * k)
            bb.psi += rng.randn(n) * np.radians(10 * k)
            bb.phi = np.arctan2(np.sin(bb.phi), np.cos(bb.phi))
            bb.psi = np.arctan2(np.sin(bb.psi), np.cos(bb.psi))
        replicas.append(bb)

        # Compute initial energy
        chain.backbone.phi[:] = bb.phi
        chain.backbone.psi[:] = bb.psi
        chain.backbone.omega[:] = bb.omega
        energies[k] = energy_fn.compute(chain, **energy_kwargs)

    best_energy = energies[0]
    best_backbone = replicas[0].copy()

    # Determine stage transitions
    early_cutoff = n_cycles * 2 // 3

    for cycle in range(n_cycles):
        stage = 'early' if cycle < early_cutoff else 'late'

        # Run independent MC on each replica
        for k in range(n_replicas):
            T = temps[k]

            for _ in range(mc_steps_per_cycle):
                # Save state
                old_phi = replicas[k].phi.copy()
                old_psi = replicas[k].psi.copy()
                old_energy = energies[k]

                # Apply move to replica
                chain.backbone.phi[:] = replicas[k].phi
                chain.backbone.psi[:] = replicas[k].psi
                chain.backbone.omega[:] = replicas[k].omega

                random_mc_move(chain.backbone, frag_lib, stage=stage, rng=rng)

                # Copy back
                replicas[k].phi[:] = chain.backbone.phi
                replicas[k].psi[:] = chain.backbone.psi

                # Compute new energy
                new_energy = energy_fn.compute(chain, **energy_kwargs)

                # Metropolis acceptance
                delta_e = new_energy - old_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
                    energies[k] = new_energy
                else:
                    # Reject: restore
                    replicas[k].phi[:] = old_phi
                    replicas[k].psi[:] = old_psi

        # Attempt swaps between adjacent replicas
        # Alternate even-odd pairs for detailed balance
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
                # Swap replicas
                replicas[i], replicas[j] = replicas[j], replicas[i]
                energies[i], energies[j] = energies[j], energies[i]

        # Track best structure (from lowest temperature replica)
        if energies[0] < best_energy:
            best_energy = energies[0]
            best_backbone = replicas[0].copy()

    return best_backbone
