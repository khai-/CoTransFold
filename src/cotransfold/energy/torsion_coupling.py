"""Torsion angle coupling energy — correlates adjacent backbone angles.

Penalizes adjacent (phi_i, psi_i) → (phi_i+1, psi_i+1) combinations that
are unlikely in real proteins. This helps stabilize secondary structure
by enforcing that consecutive residues adopt compatible conformations.

Uses a simplified model: penalizes large differences in psi_i vs phi_{i+1}
(the omega-adjacent torsion pair) since these are geometrically coupled.
Also rewards consistent secondary structure runs (helix-helix, strand-strand).
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.energy.total import EnergyTerm


# Ideal angle pairs for secondary structure
HELIX_PHI = np.radians(-57.0)
HELIX_PSI = np.radians(-47.0)
STRAND_PHI = np.radians(-120.0)
STRAND_PSI = np.radians(120.0)


def _angle_diff(a: float, b: float) -> float:
    """Smallest signed difference between two angles (radians)."""
    d = a - b
    return d - 2 * np.pi * np.round(d / (2 * np.pi))


class TorsionCouplingEnergy(EnergyTerm):
    """Reward consecutive residues that adopt consistent secondary structure."""

    @property
    def name(self) -> str:
        return "torsion_coupling"

    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        n = backbone.num_residues
        if n < 3:
            return 0.0

        phi = backbone.phi
        psi = backbone.psi

        energy = 0.0

        for i in range(n - 1):
            # Check if residue i and i+1 are both near helix angles
            d_phi_h_i = _angle_diff(phi[i], HELIX_PHI)
            d_psi_h_i = _angle_diff(psi[i], HELIX_PSI)
            d_phi_h_j = _angle_diff(phi[i + 1], HELIX_PHI)
            d_psi_h_j = _angle_diff(psi[i + 1], HELIX_PSI)

            helix_i = (d_phi_h_i ** 2 + d_psi_h_i ** 2) / (np.radians(30) ** 2)
            helix_j = (d_phi_h_j ** 2 + d_psi_h_j ** 2) / (np.radians(30) ** 2)

            # Check if both near strand angles
            d_phi_s_i = _angle_diff(phi[i], STRAND_PHI)
            d_psi_s_i = _angle_diff(psi[i], STRAND_PSI)
            d_phi_s_j = _angle_diff(phi[i + 1], STRAND_PHI)
            d_psi_s_j = _angle_diff(psi[i + 1], STRAND_PSI)

            strand_i = (d_phi_s_i ** 2 + d_psi_s_i ** 2) / (np.radians(30) ** 2)
            strand_j = (d_phi_s_j ** 2 + d_psi_s_j ** 2) / (np.radians(30) ** 2)

            # Reward: both in helix OR both in strand (Gaussian-like)
            helix_pair = np.exp(-0.5 * (helix_i + helix_j))
            strand_pair = np.exp(-0.5 * (strand_i + strand_j))

            # Negative energy = favorable (consistent SS)
            energy -= 0.3 * max(helix_pair, strand_pair)

        return energy
