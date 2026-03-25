"""Van der Waals / steric repulsion energy (vectorized).

E_vdw = sum_{i<j, |i-j|>2} epsilon * (sigma/r_ij)^12

Only the repulsive r^-12 term is used (no attractive r^-6).
Excluded: residue pairs within 2 positions in sequence.
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.energy.total import EnergyTerm
from cotransfold.structure.coordinates import get_ca_coords

# Parameters for CA-CA repulsive potential
CA_SIGMA = 3.8      # Å, equilibrium CA-CA distance
CA_EPSILON = 0.05   # kcal/mol, repulsion strength
MIN_SEQ_SEP = 3     # Skip bonded neighbors (i,i+1) and (i,i+2)

# For all-atom backbone: N, CA, C atoms
BACKBONE_SIGMA = 2.8    # Å, minimum approach distance for backbone heavy atoms
BACKBONE_EPSILON = 0.10  # kcal/mol


class VanDerWaalsEnergy(EnergyTerm):
    """Steric repulsion between backbone atoms (vectorized)."""

    def __init__(self, use_all_atoms: bool = False) -> None:
        self._use_all_atoms = use_all_atoms

    @property
    def name(self) -> str:
        return "vanderwaals"

    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        n = len(coords)
        if n < MIN_SEQ_SEP + 1:
            return 0.0

        if self._use_all_atoms:
            return self._compute_all_atoms(coords, n)
        else:
            return self._compute_ca_only(coords, n)

    def _compute_ca_only(self, coords: np.ndarray, n: int) -> float:
        """CA-CA repulsive potential (vectorized)."""
        ca = get_ca_coords(coords)  # (N, 3)

        # Pairwise distance matrix
        diff = ca[:, None, :] - ca[None, :, :]  # (N, N, 3)
        dist = np.linalg.norm(diff, axis=2)      # (N, N)

        # Upper triangle mask with sequence separation
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        mask = (jj > ii) & ((jj - ii) >= MIN_SEQ_SEP) & (dist < CA_SIGMA)

        if not np.any(mask):
            return 0.0

        r = np.maximum(dist[mask], 0.1)
        ratios = CA_SIGMA / r
        return float(np.sum(CA_EPSILON * ratios ** 12))

    def _compute_all_atoms(self, coords: np.ndarray, n: int) -> float:
        """All backbone atom repulsive potential (vectorized)."""
        flat = coords.reshape(-1, 3)  # (3*N, 3)
        n_atoms = len(flat)

        diff = flat[:, None, :] - flat[None, :, :]  # (3N, 3N, 3)
        dist = np.linalg.norm(diff, axis=2)          # (3N, 3N)

        # Residue indices for each atom
        res_idx = np.arange(n_atoms) // 3
        res_i = res_idx[:, None]
        res_j = res_idx[None, :]

        mask = (res_j > res_i) & (np.abs(res_j - res_i) >= MIN_SEQ_SEP) & (dist < BACKBONE_SIGMA)

        if not np.any(mask):
            return 0.0

        r = np.maximum(dist[mask], 0.1)
        ratios = BACKBONE_SIGMA / r
        return float(np.sum(BACKBONE_EPSILON * ratios ** 12))
