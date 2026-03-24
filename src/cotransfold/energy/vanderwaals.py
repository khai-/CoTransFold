"""Van der Waals / steric repulsion energy.

Uses a soft repulsive potential between backbone atoms to prevent
steric clashes. For the backbone-only MVP, we use CA-CA distances
with a modified Lennard-Jones repulsive term.

E_vdw = sum_{i<j, |i-j|>2} epsilon * (sigma/r_ij)^12

Only the repulsive r^-12 term is used (no attractive r^-6),
since backbone-only representation lacks the detail for
accurate van der Waals attractions.

Excluded: residue pairs within 2 positions in sequence (bonded neighbors).
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
    """Steric repulsion between backbone atoms.

    Uses CA-based coarse model for speed, with optional all-backbone-atom mode.
    """

    def __init__(self, use_all_atoms: bool = False) -> None:
        """
        Args:
            use_all_atoms: if True, compute repulsion between all N/CA/C atoms.
                          if False (default), use only CA-CA pairs.
        """
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
        """CA-CA repulsive potential."""
        ca = get_ca_coords(coords)  # shape (N, 3)
        energy = 0.0

        for i in range(n):
            for j in range(i + MIN_SEQ_SEP, n):
                r = np.linalg.norm(ca[j] - ca[i])
                if r < CA_SIGMA:
                    # Soft repulsion: epsilon * (sigma/r)^12
                    ratio = CA_SIGMA / max(r, 0.1)
                    energy += CA_EPSILON * ratio ** 12
        return energy

    def _compute_all_atoms(self, coords: np.ndarray, n: int) -> float:
        """All backbone atom (N, CA, C) repulsive potential."""
        flat = coords.reshape(-1, 3)  # shape (3*N, 3)
        n_atoms = len(flat)
        energy = 0.0

        for i in range(n_atoms):
            res_i = i // 3
            for j in range(i + 1, n_atoms):
                res_j = j // 3
                if abs(res_j - res_i) < MIN_SEQ_SEP:
                    continue

                r = np.linalg.norm(flat[j] - flat[i])
                if r < BACKBONE_SIGMA:
                    ratio = BACKBONE_SIGMA / max(r, 0.1)
                    energy += BACKBONE_EPSILON * ratio ** 12
        return energy
