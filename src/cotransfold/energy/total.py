"""Energy function compositor and abstract base class for energy terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from cotransfold.core.chain import NascentChain
from cotransfold.core.conformation import BackboneState
from cotransfold.structure.coordinates import torsion_to_cartesian


class EnergyTerm(ABC):
    """Abstract base class for an individual energy contribution."""

    @abstractmethod
    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        """Compute this energy contribution in kcal/mol.

        Args:
            coords: Cartesian coordinates, shape (N, 3, 3) for [N, CA, C]
            backbone: current torsion angle state
            sequence: list of AminoAcid
            **kwargs: additional context (e.g., tunnel geometry)

        Returns:
            Energy in kcal/mol
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this energy term."""
        ...


class TotalEnergy:
    """Compositor that sums weighted energy terms.

    E_total = sum_i (weight_i * E_i)
    """

    def __init__(self) -> None:
        self._terms: list[tuple[EnergyTerm, float]] = []

    def add_term(self, term: EnergyTerm, weight: float = 1.0) -> None:
        """Add an energy term with a weight."""
        self._terms.append((term, weight))

    def compute(self, chain: NascentChain, **kwargs) -> float:
        """Compute total weighted energy."""
        if chain.chain_length == 0:
            return 0.0
        coords = torsion_to_cartesian(chain.backbone)
        total = 0.0
        for term, weight in self._terms:
            total += weight * term.compute(
                coords, chain.backbone, chain.sequence, **kwargs)
        return total

    def compute_decomposed(self, chain: NascentChain,
                           **kwargs) -> dict[str, float]:
        """Return individual weighted contributions."""
        if chain.chain_length == 0:
            return {}
        coords = torsion_to_cartesian(chain.backbone)
        result = {}
        for term, weight in self._terms:
            e = weight * term.compute(
                coords, chain.backbone, chain.sequence, **kwargs)
            result[term.name] = e
        return result

    def compute_with_gradient(self, chain: NascentChain,
                              step_size: float = 1e-4,
                              frozen_mask: np.ndarray | None = None,
                              **kwargs) -> tuple[float, np.ndarray]:
        """Compute energy and numerical gradient w.r.t. phi/psi angles.

        Uses central finite differences for robustness.

        Args:
            chain: the nascent chain
            step_size: finite difference step (radians)
            frozen_mask: array of shape (N,) with 0=frozen, 0.5=restricted, 1=free.
                         Gradient is zeroed for frozen residues.

        Returns:
            (energy, gradient) where gradient has shape (2*N,)
            matching BackboneState.get_torsion_vector() layout
        """
        n = chain.chain_length
        vec = chain.backbone.get_torsion_vector()  # shape (2*N,)
        grad = np.zeros_like(vec)

        # Build mask: which variables to differentiate
        if frozen_mask is not None:
            # Expand per-residue mask to per-variable (phi, psi per residue)
            var_mask = np.repeat(frozen_mask, 2)  # shape (2*N,)
        else:
            var_mask = np.ones(len(vec))

        energy = self.compute(chain, **kwargs)

        for i in range(len(vec)):
            if var_mask[i] == 0.0:
                continue

            # Central finite difference
            vec_plus = vec.copy()
            vec_plus[i] += step_size
            chain.backbone.set_from_torsion_vector(vec_plus)
            e_plus = self.compute(chain, **kwargs)

            vec_minus = vec.copy()
            vec_minus[i] -= step_size
            chain.backbone.set_from_torsion_vector(vec_minus)
            e_minus = self.compute(chain, **kwargs)

            grad[i] = (e_plus - e_minus) / (2 * step_size)

        # Restore original angles
        chain.backbone.set_from_torsion_vector(vec)
        return energy, grad
