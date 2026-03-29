"""Backbone peptide dipole-dipole interaction energy.

Each peptide bond (C=O···N-H) has a permanent dipole moment of ~3.5 Debye.
In alpha-helices, these dipoles align nearly parallel (favorable), making
helical conformations ~4.5x more electrostatically favorable than beta-sheets.

This term is essential for distinguishing helix from sheet — without it,
the energy landscape treats them as nearly equivalent.

The dipole is placed at the midpoint between C(i) and N(i+1), oriented
along the C→N direction. Energy is computed as dipole-dipole interaction
with implicit dielectric screening.

Reference: Alemani et al. (2010) "Role of Backbone Dipole Interactions in
Secondary Structure Formation" (PMC4053078)
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.energy.total import EnergyTerm
from cotransfold.structure.coordinates import get_ca_coords

# Dipole parameters
DIPOLE_MAGNITUDE = 3.5    # Debye
CHARGE_SEP = 1.4          # Å, separation between partial charges
PARTIAL_CHARGE = 0.34     # elementary charges
COULOMB_KCAL = 332.0      # kcal·Å/(mol·e²)
DIELECTRIC = 4.0          # effective dielectric constant (protein interior)
DIPOLE_CUTOFF = 12.0      # Å, distance cutoff
MIN_SEQ_SEP = 2           # minimum sequence separation


class BackboneDipoleEnergy(EnergyTerm):
    """Backbone peptide dipole-dipole interaction energy."""

    @property
    def name(self) -> str:
        return "backbone_dipole"

    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        n = len(coords)
        if n < 3:
            return 0.0

        c_atoms = coords[:, 2]   # (N, 3) C atoms
        n_atoms = coords[:, 0]   # (N, 3) N atoms

        # Peptide bond dipoles: placed between C(i) and N(i+1)
        # Number of peptide bonds = n - 1
        n_dipoles = n - 1

        # Dipole positions: midpoint of C(i) and N(i+1)
        dipole_pos = 0.5 * (c_atoms[:n_dipoles] + n_atoms[1:n_dipoles + 1])  # (n-1, 3)

        # Dipole directions: C(i) → N(i+1), normalized
        dipole_vec = n_atoms[1:n_dipoles + 1] - c_atoms[:n_dipoles]  # (n-1, 3)
        dipole_len = np.linalg.norm(dipole_vec, axis=1, keepdims=True) + 1e-10
        dipole_dir = dipole_vec / dipole_len  # unit vectors

        # Pairwise dipole-dipole interaction
        # E_dd = (1/4πε) * [μ₁·μ₂/r³ - 3(μ₁·r̂)(μ₂·r̂)/r³]
        # In kcal/mol units with partial charges:
        # E = (q²·332/ε) * [cos(θ₁₂)/r³ - 3·cos(α₁)·cos(α₂)/r³] * d²
        # where d = charge separation, θ₁₂ = angle between dipoles,
        # α₁,α₂ = angles of dipoles with inter-dipole vector

        diff = dipole_pos[:, None, :] - dipole_pos[None, :, :]  # (n-1, n-1, 3)
        dist = np.linalg.norm(diff, axis=2)  # (n-1, n-1)

        # Mask: upper triangle, sequence separation >= MIN_SEQ_SEP, distance < cutoff
        idx = np.arange(n_dipoles)
        sep = np.abs(idx[:, None] - idx[None, :])
        mask = (idx[:, None] < idx[None, :]) & (sep >= MIN_SEQ_SEP) & (dist < DIPOLE_CUTOFF)

        if not np.any(mask):
            return 0.0

        r = np.maximum(dist, 0.1)
        r_hat = diff / (r[:, :, None] + 1e-10)  # unit separation vectors

        # Dot products for dipole formula
        # μ₁·μ₂ (dot product of dipole directions)
        mu_dot = np.sum(dipole_dir[:, None, :] * dipole_dir[None, :, :], axis=2)  # (n-1, n-1)

        # μ₁·r̂ and μ₂·r̂
        mu1_r = np.sum(dipole_dir[:, None, :] * r_hat, axis=2)  # (n-1, n-1)
        mu2_r = np.sum(dipole_dir[None, :, :] * r_hat, axis=2)  # (n-1, n-1)

        # Dipole-dipole energy: E = (p²/4πε₀ε) * (μ̂₁·μ̂₂ - 3(μ̂₁·r̂)(μ̂₂·r̂)) / r³
        # Using partial charge formulation:
        prefactor = PARTIAL_CHARGE ** 2 * COULOMB_KCAL * CHARGE_SEP ** 2 / DIELECTRIC

        e_dd = prefactor * (mu_dot - 3.0 * mu1_r * mu2_r) / (r ** 3)

        return float(np.sum(e_dd[mask]))
