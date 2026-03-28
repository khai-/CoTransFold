"""Beta-sheet strand pairing energy.

Rewards formation of parallel and antiparallel beta-sheet hydrogen bond
patterns. This is separate from the general H-bond term because it
specifically looks for the alternating i→j, i+2→j-2 (antiparallel) or
i→j, i+2→j+2 (parallel) register patterns that define beta-sheets.

E_sheet = sum over strand pairs: e_register * f(geometry)
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.energy.total import EnergyTerm
from cotransfold.structure.coordinates import get_ca_coords

# Beta-strand phi/psi angles
STRAND_PHI_MIN = np.radians(-170.0)
STRAND_PHI_MAX = np.radians(-70.0)
STRAND_PSI_MIN = np.radians(70.0)
STRAND_PSI_MAX = np.radians(180.0)

# Sheet geometry parameters
STRAND_CA_DIST = 4.7    # Å, typical inter-strand CA-CA distance
STRAND_DIST_TOL = 1.5   # Å, tolerance
MIN_SEQ_SEP = 4         # Minimum sequence separation for sheet partners


def _is_strand_like(phi: float, psi: float) -> bool:
    """Check if phi/psi angles are in the beta-strand region."""
    return (STRAND_PHI_MIN <= phi <= STRAND_PHI_MAX and
            STRAND_PSI_MIN <= psi <= STRAND_PSI_MAX)


class SheetPairingEnergy(EnergyTerm):
    """Energy term that rewards beta-sheet strand pairing register."""

    @property
    def name(self) -> str:
        return "sheet_pairing"

    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        n = len(coords)
        if n < 6:
            return 0.0

        ca = get_ca_coords(coords)
        phi = backbone.phi
        psi = backbone.psi

        # Identify strand-like residues
        is_strand = np.zeros(n, dtype=bool)
        for i in range(n):
            is_strand[i] = _is_strand_like(phi[i], psi[i])

        # Pairwise CA distances
        diff = ca[:, None, :] - ca[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        energy = 0.0

        # Look for antiparallel strand pairs: i paired with j, i+2 with j-2
        for i in range(n - 2):
            if not is_strand[i] or not is_strand[i + 2]:
                continue
            for j in range(i + MIN_SEQ_SEP + 2, n):
                if j < 2:
                    continue
                if not is_strand[j] or not is_strand[j - 2]:
                    continue

                d_ij = dist[i, j]
                d_i2_j2 = dist[i + 2, j - 2]

                # Both pairs should be at strand contact distance
                if (abs(d_ij - STRAND_CA_DIST) < STRAND_DIST_TOL and
                        abs(d_i2_j2 - STRAND_CA_DIST) < STRAND_DIST_TOL):
                    # Gaussian reward
                    g1 = np.exp(-0.5 * ((d_ij - STRAND_CA_DIST) / STRAND_DIST_TOL) ** 2)
                    g2 = np.exp(-0.5 * ((d_i2_j2 - STRAND_CA_DIST) / STRAND_DIST_TOL) ** 2)
                    energy -= 0.5 * g1 * g2  # Negative = favorable

        # Look for parallel strand pairs: i with j, i+2 with j+2
        for i in range(n - 2):
            if not is_strand[i] or not is_strand[i + 2]:
                continue
            for j in range(i + MIN_SEQ_SEP, n - 2):
                if not is_strand[j] or not is_strand[j + 2]:
                    continue

                d_ij = dist[i, j]
                d_i2_j2 = dist[i + 2, j + 2]

                if (abs(d_ij - STRAND_CA_DIST) < STRAND_DIST_TOL and
                        abs(d_i2_j2 - STRAND_CA_DIST) < STRAND_DIST_TOL):
                    g1 = np.exp(-0.5 * ((d_ij - STRAND_CA_DIST) / STRAND_DIST_TOL) ** 2)
                    g2 = np.exp(-0.5 * ((d_i2_j2 - STRAND_CA_DIST) / STRAND_DIST_TOL) ** 2)
                    energy -= 0.3 * g1 * g2  # Slightly less favorable than antiparallel

        return energy
