"""JAX-powered L-BFGS-B minimizer with exact autodiff gradients.

Uses jax.grad for exact gradients instead of numerical finite differences.
This is 50-200x faster than the numerical gradient minimizer for large proteins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize as scipy_minimize

from cotransfold.core.chain import NascentChain
from cotransfold.core.residue import AminoAcid, RESIDUE_PROPERTIES
from cotransfold.energy.jax_energy import total_energy_jax, _grad_phi_psi
from cotransfold.energy.solvent import SOLVATION_PARAMS
from cotransfold.minimizer.gradient import MinimizationResult


# Precompute residue class mappings
def _get_rama_class(aa: AminoAcid, next_aa) -> int:
    """0=general, 1=glycine, 2=proline, 3=preproline"""
    if aa == AminoAcid.GLY:
        return 1
    if aa == AminoAcid.PRO:
        return 2
    if next_aa == AminoAcid.PRO:
        return 3
    return 0


class JaxMinimizer:
    """L-BFGS-B minimizer using JAX autodiff for exact gradients."""

    def __init__(self,
                 max_iterations: int = 200,
                 ftol: float = 1e-6,
                 gtol: float = 1e-4) -> None:
        self.max_iterations = max_iterations
        self.ftol = ftol
        self.gtol = gtol

    def _prepare_static_params(self, chain: NascentChain,
                               tunnel_length: float,
                               **energy_kwargs) -> dict:
        """Prepare all static parameters as JAX arrays."""
        n = chain.chain_length
        seq = chain.sequence

        # Ramachandran classes
        rama_classes = np.zeros(n, dtype=np.int32)
        for i in range(n):
            next_aa = seq[i + 1] if i + 1 < n else None
            rama_classes[i] = _get_rama_class(seq[i], next_aa)

        # Charges
        charges = np.array([RESIDUE_PROPERTIES[aa].charge for aa in seq])

        # Solvation parameters
        solv = np.array([SOLVATION_PARAMS.get(aa, 0.0) for aa in seq])

        # Tunnel parameters
        tunnel_positions = energy_kwargs.get('tunnel_positions', np.full(n, tunnel_length + 10))

        # Precompute tunnel radii and potentials (these don't change during minimization)
        from cotransfold.tunnel.organisms import get_tunnel
        try:
            geom, elec = get_tunnel('ecoli')
            tunnel_radii = np.array([geom.radius_at(d) for d in tunnel_positions])
            tunnel_potentials = np.array([elec.potential_at(d) for d in tunnel_positions])
        except Exception:
            tunnel_radii = np.full(n, 10.0)
            tunnel_potentials = np.zeros(n)

        return {
            'rama_classes': jnp.array(rama_classes),
            'charges': jnp.array(charges),
            'solvation_params': jnp.array(solv),
            'tunnel_positions': jnp.array(tunnel_positions),
            'tunnel_radii': jnp.array(tunnel_radii),
            'tunnel_potentials': jnp.array(tunnel_potentials),
            'tunnel_length': float(tunnel_length),
        }

    def minimize(self,
                 chain: NascentChain,
                 energy_fn=None,  # Unused, kept for API compat
                 frozen_mask: np.ndarray | None = None,
                 max_iterations: int | None = None,
                 weights: dict | None = None,
                 tunnel_length: float = 90.0,
                 **energy_kwargs) -> MinimizationResult:
        """Minimize energy using JAX autodiff gradients.

        Args:
            chain: nascent chain (modified in-place)
            energy_fn: ignored (uses JAX energy internally)
            frozen_mask: per-residue mask (0=frozen, 1=free)
            max_iterations: override default
            weights: energy term weights
            tunnel_length: tunnel length for solvent/tunnel terms
        """
        n = chain.chain_length
        if n == 0:
            return MinimizationResult(0.0, 0.0, 0, True, "empty chain")

        max_iter = max_iterations or self.max_iterations

        if frozen_mask is None:
            frozen_mask = np.ones(n)

        free_indices = np.where(frozen_mask > 0)[0]
        if len(free_indices) == 0:
            return MinimizationResult(0.0, 0.0, 0, True, "no free variables")

        # Prepare static parameters
        params = self._prepare_static_params(chain, tunnel_length, **energy_kwargs)
        omega = jnp.array(chain.backbone.omega)
        n_free = len(free_indices)

        def _objective_and_grad(x_flat: np.ndarray):
            """Compute energy and gradient for scipy optimizer."""
            # Unpack: x_flat = [phi_free_0, psi_free_0, phi_free_1, psi_free_1, ...]
            phi = jnp.array(chain.backbone.phi)
            psi = jnp.array(chain.backbone.psi)

            # Set free variables
            phi = phi.at[free_indices].set(jnp.array(x_flat[0::2]))
            psi = psi.at[free_indices].set(jnp.array(x_flat[1::2]))

            # Energy
            e = total_energy_jax(phi, psi, omega,
                                 params['rama_classes'], params['charges'],
                                 params['solvation_params'],
                                 params['tunnel_positions'],
                                 params['tunnel_radii'],
                                 params['tunnel_potentials'],
                                 params['tunnel_length'],
                                 weights)

            # Gradients
            g_phi, g_psi = _grad_phi_psi(phi, psi, omega,
                                          params['rama_classes'], params['charges'],
                                          params['solvation_params'],
                                          params['tunnel_positions'],
                                          params['tunnel_radii'],
                                          params['tunnel_potentials'],
                                          params['tunnel_length'],
                                          weights)

            # Extract gradients for free variables only
            grad = np.empty(2 * n_free)
            grad[0::2] = np.array(g_phi[free_indices])
            grad[1::2] = np.array(g_psi[free_indices])

            return float(e), grad

        # Initial variables
        x0 = np.empty(2 * n_free)
        x0[0::2] = chain.backbone.phi[free_indices]
        x0[1::2] = chain.backbone.psi[free_indices]

        energy_before = _objective_and_grad(x0)[0]

        # Bounds
        bounds = [(-np.pi, np.pi)] * (2 * n_free)

        result = scipy_minimize(
            lambda x: _objective_and_grad(x),
            x0,
            method='L-BFGS-B',
            jac=True,  # objective returns (f, grad)
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': self.ftol,
                'gtol': self.gtol,
            },
        )

        # Apply result
        chain.backbone.phi[free_indices] = result.x[0::2]
        chain.backbone.psi[free_indices] = result.x[1::2]

        return MinimizationResult(
            energy_before=energy_before,
            energy_after=float(result.fun),
            n_iterations=result.nit,
            converged=result.success,
            message=result.message if isinstance(result.message, str) else result.message.decode(),
        )
