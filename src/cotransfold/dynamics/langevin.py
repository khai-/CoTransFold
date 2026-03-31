"""Overdamped Langevin dynamics in torsion angle space.

The most physically faithful approach to protein folding simulation.
Each torsion angle evolves according to:

    dφ/dt = -∇E(φ)/γ + √(2kT/γ) · η(t)

where:
- ∇E(φ) = gradient of energy w.r.t. torsion angle
- γ = friction coefficient (models solvent viscosity)
- kT = thermal energy
- η(t) = Gaussian white noise

In discrete time (Euler-Maruyama integrator):

    φ(t+dt) = φ(t) - (dt/γ) · ∇E + √(2kT·dt/γ) · N(0,1)

This produces a trajectory that samples the Boltzmann distribution
P(φ) ∝ exp(-E(φ)/kT) in the long-time limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cotransfold.core.chain import NascentChain
from cotransfold.core.conformation import BackboneState
from cotransfold.energy.total import TotalEnergy


@dataclass
class LangevinConfig:
    """Configuration for Langevin dynamics."""
    dt: float = 0.005           # Time step (reduced units)
    friction: float = 10.0      # Friction coefficient γ
    temperature: float = 310.15 # K
    gradient_step: float = 1e-4 # Finite difference step for gradient
    kT_kcal: float = 0.616     # kT at 310K in kcal/mol


def _numerical_gradient(energy_fn: TotalEnergy, chain: NascentChain,
                        frozen_mask: np.ndarray, step: float = 1e-4,
                        **kwargs) -> np.ndarray:
    """Compute dE/d(phi,psi) via central finite differences.

    Returns gradient of shape (2*N,) with layout [phi_0, psi_0, phi_1, psi_1, ...]
    """
    n = chain.chain_length
    grad = np.zeros(2 * n)

    e0 = energy_fn.compute(chain, **kwargs)

    for i in range(n):
        if frozen_mask[i] == 0:
            continue

        # d/d(phi_i)
        old_phi = chain.backbone.phi[i]
        chain.backbone.phi[i] = old_phi + step
        ep = energy_fn.compute(chain, **kwargs)
        chain.backbone.phi[i] = old_phi - step
        em = energy_fn.compute(chain, **kwargs)
        chain.backbone.phi[i] = old_phi
        grad[2 * i] = (ep - em) / (2 * step)

        # d/d(psi_i)
        old_psi = chain.backbone.psi[i]
        chain.backbone.psi[i] = old_psi + step
        ep = energy_fn.compute(chain, **kwargs)
        chain.backbone.psi[i] = old_psi - step
        em = energy_fn.compute(chain, **kwargs)
        chain.backbone.psi[i] = old_psi
        grad[2 * i + 1] = (ep - em) / (2 * step)

    return grad


def run_langevin(chain: NascentChain,
                 energy_fn: TotalEnergy,
                 frozen_mask: np.ndarray,
                 n_steps: int = 500,
                 config: LangevinConfig | None = None,
                 energy_kwargs: dict | None = None,
                 rng: np.random.RandomState | None = None,
                 ) -> float:
    """Run overdamped Langevin dynamics on torsion angles.

    Args:
        chain: nascent chain (backbone modified in place)
        energy_fn: total energy function
        frozen_mask: per-residue mask (0=frozen, 1=free)
        n_steps: number of dynamics steps
        config: Langevin parameters
        energy_kwargs: extra kwargs for energy
        rng: random state

    Returns:
        Final energy after dynamics
    """
    if config is None:
        config = LangevinConfig()
    if energy_kwargs is None:
        energy_kwargs = {}
    if rng is None:
        rng = np.random.RandomState()

    n = chain.chain_length
    if n == 0:
        return 0.0

    dt = config.dt
    gamma = config.friction
    kT = config.kT_kcal

    # Noise amplitude: sqrt(2*kT*dt/γ)
    noise_amp = np.sqrt(2 * kT * dt / gamma)

    # Expand frozen mask to per-variable
    var_mask = np.repeat(frozen_mask, 2)  # (2*N,)

    best_energy = energy_fn.compute(chain, **energy_kwargs)
    best_phi = chain.backbone.phi.copy()
    best_psi = chain.backbone.psi.copy()

    for step in range(n_steps):
        # Compute gradient
        grad = _numerical_gradient(
            energy_fn, chain, frozen_mask,
            step=config.gradient_step, **energy_kwargs)

        # Langevin update: φ(t+dt) = φ(t) - (dt/γ)·∇E + noise
        noise = noise_amp * rng.randn(2 * n)

        # Apply update (only to free variables)
        delta = -(dt / gamma) * grad + noise
        delta *= var_mask

        # Update phi/psi
        chain.backbone.phi += delta[0::2]
        chain.backbone.psi += delta[1::2]

        # Wrap to [-π, π]
        chain.backbone.phi = np.arctan2(
            np.sin(chain.backbone.phi), np.cos(chain.backbone.phi))
        chain.backbone.psi = np.arctan2(
            np.sin(chain.backbone.psi), np.cos(chain.backbone.psi))

        # Track best structure (lowest energy seen)
        if step % 10 == 0 or step == n_steps - 1:
            e = energy_fn.compute(chain, **energy_kwargs)
            if e < best_energy:
                best_energy = e
                best_phi[:] = chain.backbone.phi
                best_psi[:] = chain.backbone.psi

    # Restore best structure seen during dynamics
    chain.backbone.phi[:] = best_phi
    chain.backbone.psi[:] = best_psi

    return best_energy


def run_annealed_langevin(chain: NascentChain,
                          energy_fn: TotalEnergy,
                          frozen_mask: np.ndarray,
                          n_steps: int = 2000,
                          t_start: float = 2.0,
                          t_end: float = 0.3,
                          n_stages: int = 5,
                          energy_kwargs: dict | None = None,
                          rng: np.random.RandomState | None = None,
                          ) -> float:
    """Run Langevin dynamics with temperature annealing.

    Starts hot (broad exploration) and cools down (refinement).
    This is the physically motivated version of simulated annealing —
    it's what happens when a protein cools from denatured to native.

    Returns: final energy
    """
    if energy_kwargs is None:
        energy_kwargs = {}
    if rng is None:
        rng = np.random.RandomState()

    temps = np.geomspace(t_start, t_end, n_stages)
    steps_per_stage = n_steps // n_stages

    best_energy = float('inf')
    best_phi = chain.backbone.phi.copy()
    best_psi = chain.backbone.psi.copy()

    for temp_factor in temps:
        cfg = LangevinConfig(
            dt=0.005 * min(temp_factor, 2.0),  # Larger steps at high T
            friction=10.0 / temp_factor,         # Less friction at high T
            temperature=310.15,
            kT_kcal=0.616 * temp_factor,         # Scale effective temperature
        )

        e = run_langevin(
            chain, energy_fn, frozen_mask,
            n_steps=steps_per_stage,
            config=cfg,
            energy_kwargs=energy_kwargs,
            rng=rng,
        )

        if e < best_energy:
            best_energy = e
            best_phi[:] = chain.backbone.phi
            best_psi[:] = chain.backbone.psi

    # Restore best
    chain.backbone.phi[:] = best_phi
    chain.backbone.psi[:] = best_psi

    return best_energy
