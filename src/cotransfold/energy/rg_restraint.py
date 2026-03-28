"""Radius of gyration restraint energy.

Drives the chain toward the expected compactness for a globular protein
of its length. Without this, chains tend to remain too extended.

E_rg = (Rg_actual - Rg_expected)^2

where Rg_expected = 0.395 * N^0.6 + 7.257 (Å), empirical fit from
Entelis et al. (2008) for globular proteins.
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.energy.total import EnergyTerm
from cotransfold.structure.coordinates import get_ca_coords


def expected_rg(n_residues: int) -> float:
    """Expected radius of gyration for a globular protein."""
    return 0.395 * n_residues ** 0.6 + 7.257


def compute_rg(ca: np.ndarray) -> float:
    """Compute radius of gyration from CA coordinates."""
    centroid = ca.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((ca - centroid) ** 2, axis=1))))


class RgRestraintEnergy(EnergyTerm):
    """Radius of gyration restraint — penalizes deviation from expected Rg."""

    @property
    def name(self) -> str:
        return "rg_restraint"

    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        n = len(coords)
        if n < 5:
            return 0.0

        ca = get_ca_coords(coords)
        rg = compute_rg(ca)
        rg_exp = expected_rg(n)

        return (rg - rg_exp) ** 2
