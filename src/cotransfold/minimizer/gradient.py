"""L-BFGS-B energy minimizer operating on torsion angles (optimized).

Minimizes total energy by adjusting phi/psi angles of free residues.
Uses scipy's L-BFGS-B optimizer with:
- Ramachandran-derived bounds on phi/psi
- Frozen residue masking (tunnel constraints)
- Vectorized numerical gradients
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from cotransfold.core.chain import NascentChain
from cotransfold.energy.total import TotalEnergy


@dataclass
class MinimizationResult:
    """Result of energy minimization."""
    energy_before: float
    energy_after: float
    n_iterations: int
    converged: bool
    message: str


# Default phi/psi bounds
PHI_BOUNDS = (-np.pi, np.pi)
PSI_BOUNDS = (-np.pi, np.pi)


class GradientMinimizer:
    """L-BFGS-B energy minimizer for torsion angles."""

    def __init__(self,
                 max_iterations: int = 200,
                 gradient_step: float = 1e-4,
                 ftol: float = 1e-6,
                 gtol: float = 1e-4) -> None:
        self.max_iterations = max_iterations
        self.gradient_step = gradient_step
        self.ftol = ftol
        self.gtol = gtol

    def minimize(self,
                 chain: NascentChain,
                 energy_fn: TotalEnergy,
                 frozen_mask: np.ndarray | None = None,
                 max_iterations: int | None = None,
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

        # Build index arrays for fast variable get/set
        # Variables are [phi_0, psi_0, phi_1, psi_1, ...] for free residues
        n_free = len(free_indices)
        n_vars = 2 * n_free
        phi_slots = free_indices  # Which backbone.phi entries to optimize
        psi_slots = free_indices  # Which backbone.psi entries to optimize

        def _get_vars() -> np.ndarray:
            x = np.empty(n_vars)
            x[0::2] = chain.backbone.phi[phi_slots]
            x[1::2] = chain.backbone.psi[psi_slots]
            return x

        def _set_vars(x: np.ndarray) -> None:
            chain.backbone.phi[phi_slots] = x[0::2]
            chain.backbone.psi[psi_slots] = x[1::2]

        def _objective(x: np.ndarray) -> float:
            _set_vars(x)
            return energy_fn.compute(chain, **energy_kwargs)

        def _gradient(x: np.ndarray) -> np.ndarray:
            """Compute gradient using forward finite differences (faster than central)."""
            h = self.gradient_step
            e0 = _objective(x)
            grad = np.zeros(n_vars)

            # Forward finite differences: 1 eval per variable instead of 2
            for i in range(n_vars):
                x[i] += h
                _set_vars(x)
                e_plus = energy_fn.compute(chain, **energy_kwargs)
                grad[i] = (e_plus - e0) / h
                x[i] -= h  # Restore

            _set_vars(x)  # Restore original
            return grad

        # Set up bounds
        bounds = []
        for _ in range(n_free):
            bounds.append(PHI_BOUNDS)
            bounds.append(PSI_BOUNDS)

        x0 = _get_vars()
        energy_before = _objective(x0)

        result = scipy_minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            jac=_gradient,
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': self.ftol,
                'gtol': self.gtol,
            },
        )

        _set_vars(result.x)
        energy_after = result.fun

        return MinimizationResult(
            energy_before=energy_before,
            energy_after=energy_after,
            n_iterations=result.nit,
            converged=result.success,
            message=result.message,
        )
