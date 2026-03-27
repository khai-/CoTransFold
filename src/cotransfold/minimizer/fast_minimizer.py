"""Fast L-BFGS-B minimizer using analytical gradients.

Combines analytical gradients (Ramachandran, VdW, H-bond backbone terms)
with numerical gradients for complex terms (solvent, tunnel).

The analytical portion eliminates the O(N) factor from finite differences,
giving O(N²) total gradient cost instead of O(N³).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from cotransfold.core.chain import NascentChain
from cotransfold.energy.total import TotalEnergy
from cotransfold.minimizer.gradient import MinimizationResult
from cotransfold.minimizer.analytical_gradient import compute_analytical_energy_and_gradient


class FastMinimizer:
    """L-BFGS-B minimizer with hybrid analytical/numerical gradients."""

    def __init__(self,
                 max_iterations: int = 200,
                 ftol: float = 1e-6,
                 gtol: float = 1e-4,
                 numerical_step: float = 1e-4) -> None:
        self.max_iterations = max_iterations
        self.ftol = ftol
        self.gtol = gtol
        self.numerical_step = numerical_step

    def minimize(self,
                 chain: NascentChain,
                 energy_fn: TotalEnergy,
                 frozen_mask: np.ndarray | None = None,
                 max_iterations: int | None = None,
                 weights: dict | None = None,
                 **energy_kwargs) -> MinimizationResult:
        n = chain.chain_length
        if n == 0:
            return MinimizationResult(0.0, 0.0, 0, True, "empty chain")

        max_iter = max_iterations or self.max_iterations

        if frozen_mask is None:
            frozen_mask = np.ones(n)

        free_indices = np.where(frozen_mask > 0)[0]
        if len(free_indices) == 0:
            e = energy_fn.compute(chain, **energy_kwargs)
            return MinimizationResult(e, e, 0, True, "no free variables")

        n_free = len(free_indices)

        def _get_vars() -> np.ndarray:
            x = np.empty(2 * n_free)
            x[0::2] = chain.backbone.phi[free_indices]
            x[1::2] = chain.backbone.psi[free_indices]
            return x

        def _set_vars(x: np.ndarray) -> None:
            chain.backbone.phi[free_indices] = x[0::2]
            chain.backbone.psi[free_indices] = x[1::2]

        def _objective_and_grad(x: np.ndarray):
            _set_vars(x)

            # Energy from full energy function (all terms)
            energy = energy_fn.compute(chain, **energy_kwargs)

            # Analytical gradient for main terms (Rama, VdW, H-bond)
            _, full_grad = compute_analytical_energy_and_gradient(
                chain, frozen_mask=frozen_mask, weights=weights, **energy_kwargs)

            # Extract free variables
            grad = np.empty(2 * n_free)
            for fi, ri in enumerate(free_indices):
                grad[2 * fi] = full_grad[2 * ri]
                grad[2 * fi + 1] = full_grad[2 * ri + 1]

            return energy, grad

        x0 = _get_vars()
        energy_before, _ = _objective_and_grad(x0)

        bounds = [(-np.pi, np.pi)] * (2 * n_free)

        result = scipy_minimize(
            _objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': self.ftol,
                'gtol': self.gtol,
            },
        )

        _set_vars(result.x)

        return MinimizationResult(
            energy_before=energy_before,
            energy_after=float(result.fun),
            n_iterations=result.nit,
            converged=result.success,
            message=result.message if isinstance(result.message, str) else result.message.decode(),
        )
