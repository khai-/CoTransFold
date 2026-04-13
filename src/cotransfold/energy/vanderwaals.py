"""Van der Waals energy with full Lennard-Jones potential (vectorized).

E_LJ = sum_{i<j, |i-j|>2} epsilon * [(sigma/r_ij)^12 - 2*(sigma/r_ij)^6]

Includes both CA-CA backbone interactions and Cβ-Cβ sidechain interactions
with residue-specific van der Waals radii.
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.core.residue import AminoAcid, RESIDUE_PROPERTIES
from cotransfold.energy.total import EnergyTerm
from cotransfold.structure.coordinates import get_ca_coords, get_cb_coords

# Parameters for CA-CA LJ potential
CA_SIGMA = 3.8      # Å, equilibrium CA-CA distance
CA_EPSILON = 0.05   # kcal/mol
MIN_SEQ_SEP = 3     # Skip bonded neighbors (i,i+1) and (i,i+2)
LJ_CUTOFF = 10.0    # Å, beyond which LJ is negligible

# For all-atom backbone: N, CA, C atoms
BACKBONE_SIGMA = 2.8    # Å, minimum approach distance for backbone heavy atoms
BACKBONE_EPSILON = 0.10  # kcal/mol

# Cβ-Cβ interaction parameters
CB_EPSILON = 0.03   # kcal/mol, sidechain interaction strength

# Soft-core parameters — caps repulsion to prevent infinite barriers
REPULSION_CAP = 10.0   # Maximum repulsive energy per pair (kcal/mol)
SOFT_CORE_DELTA = 0.5  # Å, soft-core shift to round off singularity


class VanDerWaalsEnergy(EnergyTerm):
    """Lennard-Jones potential between backbone and sidechain atoms."""

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
            e_bb = self._compute_all_atoms(coords, n)
        else:
            e_bb = self._compute_ca_only(coords, n)

        e_cb = self._compute_cb_interactions(coords, n, sequence)
        return e_bb + e_cb

    def _compute_ca_only(self, coords: np.ndarray, n: int) -> float:
        """Soft-core CA-CA Lennard-Jones potential (vectorized).

        Uses r_eff = (r⁶ + δ⁶)^(1/6) to round off the r⁻¹² singularity,
        and caps repulsion at REPULSION_CAP to prevent infinite barriers.
        """
        ca = get_ca_coords(coords)  # (N, 3)

        # Pairwise distance matrix
        diff = ca[:, None, :] - ca[None, :, :]  # (N, N, 3)
        dist = np.linalg.norm(diff, axis=2)      # (N, N)

        # Upper triangle mask with sequence separation and cutoff
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        mask = (jj > ii) & ((jj - ii) >= MIN_SEQ_SEP) & (dist < LJ_CUTOFF)

        if not np.any(mask):
            return 0.0

        # Soft-core effective distance: rounds off singularity at r→0
        r_raw = dist[mask]
        r_eff = (r_raw ** 6 + SOFT_CORE_DELTA ** 6) ** (1.0 / 6.0)
        r_eff = np.maximum(r_eff, 0.1)

        ratios = CA_SIGMA / r_eff
        # Capped LJ: min(repulsive, cap) - attractive, floored at -1 per pair.
        # The physical LJ well depth is exactly -ε; without clamping the
        # attractive term the soft-core r_eff floor of 0.5Å lets (σ/r_eff)^6
        # explode, producing unphysical -10^5 energies for crumpled structures.
        repulsive = np.minimum(ratios ** 12, REPULSION_CAP / CA_EPSILON)
        attractive = 2.0 * ratios ** 6
        lj = np.maximum(repulsive - attractive, -1.0)
        return float(np.sum(CA_EPSILON * lj))

    def _compute_cb_interactions(self, coords: np.ndarray, n: int,
                                 sequence: list) -> float:
        """Cβ-Cβ LJ potential with residue-specific radii (vectorized)."""
        cb = get_cb_coords(coords, sequence)  # (N, 3)

        # Per-residue VdW radii
        radii = np.array([RESIDUE_PROPERTIES[aa].vdw_radius for aa in sequence])

        # Pairwise sigma: sum of radii
        sigma_ij = radii[:, None] + radii[None, :]  # (N, N)

        # Skip pairs involving GLY (vdw_radius=0) or where sigma is too small
        valid_radii = sigma_ij > 1.0

        # Pairwise distances
        diff = cb[:, None, :] - cb[None, :, :]  # (N, N, 3)
        dist = np.linalg.norm(diff, axis=2)      # (N, N)

        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        mask = (jj > ii) & ((jj - ii) >= MIN_SEQ_SEP) & (dist < LJ_CUTOFF) & valid_radii

        if not np.any(mask):
            return 0.0

        # Soft-core Cβ-Cβ LJ
        r_raw = dist[mask]
        r_eff = (r_raw ** 6 + SOFT_CORE_DELTA ** 6) ** (1.0 / 6.0)
        r_eff = np.maximum(r_eff, 0.1)
        sig = sigma_ij[mask]
        ratios = sig / r_eff
        repulsive = np.minimum(ratios ** 12, REPULSION_CAP / max(CB_EPSILON, 1e-10))
        attractive = 2.0 * ratios ** 6
        lj = np.maximum(repulsive - attractive, -1.0)
        return float(np.sum(CB_EPSILON * lj))

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
