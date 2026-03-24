"""Ramachandran statistical potential.

E_rama(phi, psi) = -kT * ln(P(phi, psi | residue_class))

Uses analytically defined distributions based on allowed regions from
high-resolution crystal structures. Four residue classes:
- General (most residues)
- Glycine (no side chain, wider allowed region)
- Proline (fixed phi ~-63°)
- Pre-proline (residue before proline, restricted)
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.core.residue import AminoAcid
from cotransfold.energy.total import EnergyTerm
from cotransfold.config import BOLTZMANN_KCAL, DEFAULT_TEMPERATURE


def _gaussian_2d(phi: float, psi: float,
                 mu_phi: float, mu_psi: float,
                 sigma_phi: float, sigma_psi: float) -> float:
    """Unnormalized 2D Gaussian on the torus (periodic in phi, psi)."""
    # Periodic distance on [-pi, pi]
    dphi = phi - mu_phi
    dphi = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi))
    dpsi = psi - mu_psi
    dpsi = dpsi - 2 * np.pi * np.round(dpsi / (2 * np.pi))
    return np.exp(-0.5 * ((dphi / sigma_phi) ** 2 + (dpsi / sigma_psi) ** 2))


# Allowed regions as Gaussian mixture components: (weight, mu_phi, mu_psi, sigma_phi, sigma_psi)
# Angles in radians. Derived from Top500/Top8000 databases.

_GENERAL_REGIONS = [
    # Alpha-helix region
    (0.40, np.radians(-63), np.radians(-43), np.radians(15), np.radians(15)),
    # Beta-sheet region
    (0.25, np.radians(-120), np.radians(130), np.radians(20), np.radians(20)),
    # PPII region (left-handed helix / polyproline II)
    (0.15, np.radians(-65), np.radians(145), np.radians(15), np.radians(15)),
    # Delta region (turns)
    (0.10, np.radians(-80), np.radians(0), np.radians(25), np.radians(30)),
    # Alpha-L (left-handed helix, rare)
    (0.05, np.radians(57), np.radians(47), np.radians(20), np.radians(20)),
    # Background (small uniform)
    (0.05, 0.0, 0.0, np.radians(180), np.radians(180)),
]

_GLYCINE_REGIONS = [
    # Glycine has much wider allowed regions due to no side chain
    # Alpha-R
    (0.25, np.radians(-63), np.radians(-43), np.radians(20), np.radians(20)),
    # Alpha-L (symmetric for glycine)
    (0.20, np.radians(63), np.radians(43), np.radians(20), np.radians(20)),
    # Beta
    (0.15, np.radians(-120), np.radians(130), np.radians(25), np.radians(25)),
    # Beta mirror
    (0.15, np.radians(120), np.radians(-130), np.radians(25), np.radians(25)),
    # Broad center
    (0.15, np.radians(-80), np.radians(0), np.radians(40), np.radians(40)),
    # Background
    (0.10, 0.0, 0.0, np.radians(180), np.radians(180)),
]

_PROLINE_REGIONS = [
    # Proline: phi is constrained to ~-63° by the cyclic side chain
    # Down-pucker (trans proline, most common)
    (0.55, np.radians(-63), np.radians(-35), np.radians(8), np.radians(15)),
    # Up-pucker (cis proline)
    (0.30, np.radians(-63), np.radians(145), np.radians(8), np.radians(15)),
    # Background (very narrow in phi)
    (0.15, np.radians(-63), np.radians(0), np.radians(10), np.radians(180)),
]

_PREPROLINE_REGIONS = [
    # Residue before proline: restricted psi
    # Beta-like
    (0.40, np.radians(-120), np.radians(130), np.radians(20), np.radians(15)),
    # PPII-like
    (0.30, np.radians(-65), np.radians(145), np.radians(15), np.radians(12)),
    # Alpha restricted
    (0.20, np.radians(-63), np.radians(-30), np.radians(15), np.radians(12)),
    # Background
    (0.10, 0.0, 0.0, np.radians(180), np.radians(180)),
]


def _get_regions(aa: AminoAcid, next_aa: AminoAcid | None = None) -> list:
    """Get the appropriate Ramachandran regions for a residue."""
    if aa == AminoAcid.GLY:
        return _GLYCINE_REGIONS
    elif aa == AminoAcid.PRO:
        return _PROLINE_REGIONS
    elif next_aa == AminoAcid.PRO:
        return _PREPROLINE_REGIONS
    else:
        return _GENERAL_REGIONS


def rama_probability(phi: float, psi: float, aa: AminoAcid,
                     next_aa: AminoAcid | None = None) -> float:
    """Compute Ramachandran probability density at (phi, psi).

    Returns a value proportional to P(phi, psi | residue_class).
    """
    regions = _get_regions(aa, next_aa)
    p = 0.0
    for weight, mu_phi, mu_psi, sigma_phi, sigma_psi in regions:
        p += weight * _gaussian_2d(phi, psi, mu_phi, mu_psi, sigma_phi, sigma_psi)
    return max(p, 1e-10)  # Prevent log(0)


class RamachandranEnergy(EnergyTerm):
    """Ramachandran statistical potential.

    E = sum_i -kT * ln(P(phi_i, psi_i | class_i))

    Lower energy for residues in allowed Ramachandran regions.
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE) -> None:
        self._kT = BOLTZMANN_KCAL * temperature

    @property
    def name(self) -> str:
        return "ramachandran"

    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        n = backbone.num_residues
        if n == 0:
            return 0.0

        energy = 0.0
        for i in range(n):
            next_aa = sequence[i + 1] if i + 1 < n else None
            p = rama_probability(backbone.phi[i], backbone.psi[i],
                                 sequence[i], next_aa)
            energy += -self._kT * np.log(p)
        return energy
