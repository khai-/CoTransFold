"""L-BFGS-B energy minimizer operating on torsion angles.

Minimizes total energy by adjusting phi/psi angles of free residues.
Uses scipy's L-BFGS-B optimizer with:
- Ramachandran-derived bounds on phi/psi
- Frozen residue masking (tunnel constraints)
- Numerical gradients via central finite differences
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


# Default phi/psi bounds (generous Ramachandran)
PHI_BOUNDS = (-np.pi, np.pi)
PSI_BOUNDS = (-np.pi, np.pi)


class GradientMinimizer:
    """L-BFGS-B energy minimizer for torsion angles."""

    def __init__(self,
                 max_iterations: int = 200,
                 gradient_step: float = 1e-4,
                 ftol: float = 1e-6,
                 gtol: float = 1e-4) -> None:
        """
        Args:
            max_iterations: maximum L-BFGS-B iterations
            gradient_step: step size for numerical gradient (radians)
            ftol: function tolerance for convergence
            gtol: gradient tolerance for convergence
        """
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
        """Minimize energy by adjusting phi/psi of free residues.

        Args:
            chain: the nascent chain (modified in-place)
            energy_fn: total energy function
            frozen_mask: per-residue mask (0=frozen, 0.5=restricted, 1=free).
                        If None, all residues are free.
            max_iterations: override default max iterations
            **energy_kwargs: passed to energy_fn.compute()

        Returns:
            MinimizationResult with before/after energies
        """
        n = chain.chain_length
        if n == 0:
            return MinimizationResult(0.0, 0.0, 0, True, "empty chain")

        max_iter = max_iterations or self.max_iterations

        # Identify which variables are free
        if frozen_mask is None:
            frozen_mask = np.ones(n)

        free_indices = np.where(frozen_mask > 0)[0]
        if len(free_indices) == 0:
            e = energy_fn.compute(chain, **energy_kwargs)
            return MinimizationResult(e, e, 0, True, "no free variables")

        # Build variable vector (only free phi/psi)
        # Map: var_idx -> (residue_idx, angle_type)
        # angle_type: 0=phi, 1=psi
        var_map = []
        for ri in free_indices:
            var_map.append((ri, 0))  # phi
            var_map.append((ri, 1))  # psi
        n_vars = len(var_map)

        def _get_vars() -> np.ndarray:
            x = np.zeros(n_vars)
            for vi, (ri, at) in enumerate(var_map):
                x[vi] = chain.backbone.phi[ri] if at == 0 else chain.backbone.psi[ri]
            return x

        def _set_vars(x: np.ndarray) -> None:
            for vi, (ri, at) in enumerate(var_map):
                if at == 0:
                    chain.backbone.phi[ri] = x[vi]
                else:
                    chain.backbone.psi[ri] = x[vi]

        def _objective(x: np.ndarray) -> float:
            _set_vars(x)
            return energy_fn.compute(chain, **energy_kwargs)

        def _gradient(x: np.ndarray) -> np.ndarray:
            grad = np.zeros(n_vars)
            h = self.gradient_step
            e0 = _objective(x)
            for i in range(n_vars):
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                grad[i] = (_objective(x_plus) - _objective(x_minus)) / (2 * h)
            _set_vars(x)  # Restore
            return grad

        # Set up bounds
        bounds = []
        for ri, at in var_map:
            bounds.append(PHI_BOUNDS if at == 0 else PSI_BOUNDS)

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
