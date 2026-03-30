"""Implicit solvation energy with desolvation barrier.

Three-zone potential for each Cβ-Cβ pair:
1. Contact well (r < r₀): favorable direct contact
2. Desolvation barrier (r₀ < r < r₀+1.5): unfavorable, water must be expelled
3. Solvent-separated minimum (r₀+1.5 < r < r₀+3.0): water-mediated contact

This produces cooperative folding — the desolvation barrier creates two-state
behavior instead of gradual collapse.

Reference: Cheung et al. (2002) "Protein folding mediated by solvation:
Water expulsion and formation of the hydrophobic core" (PMC117366)

Also retains per-residue burial-based solvation as a global driving force.
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.core.residue import AminoAcid, RESIDUE_PROPERTIES
from cotransfold.energy.total import EnergyTerm
from cotransfold.structure.coordinates import get_cb_coords

# Per-residue solvation parameters (kcal/mol)
SOLVATION_PARAMS = {
    AminoAcid.ALA:  0.5,
    AminoAcid.ARG: -1.5,
    AminoAcid.ASN: -1.0,
    AminoAcid.ASP: -1.5,
    AminoAcid.CYS:  0.3,
    AminoAcid.GLU: -1.5,
    AminoAcid.GLN: -1.0,
    AminoAcid.GLY:  0.0,
    AminoAcid.HIS: -0.5,
    AminoAcid.ILE:  1.5,
    AminoAcid.LEU:  1.3,
    AminoAcid.LYS: -1.5,
    AminoAcid.MET:  0.8,
    AminoAcid.PHE:  1.2,
    AminoAcid.PRO:  0.0,
    AminoAcid.SER: -0.3,
    AminoAcid.THR: -0.2,
    AminoAcid.TRP:  1.0,
    AminoAcid.TYR:  0.5,
    AminoAcid.VAL:  1.2,
}

# Burial parameters
BURIAL_CUTOFF = 8.0  # Å
MAX_NEIGHBORS = 10

# Desolvation barrier parameters
CONTACT_R0 = 6.0        # Å, direct contact distance (Cβ-Cβ)
BARRIER_WIDTH = 1.5      # Å, width of desolvation barrier
WATER_SHELL = 3.0        # Å, water molecule diameter
CONTACT_DEPTH = 1.0      # kcal/mol, contact well depth (scaled by hydrophobicity)
BARRIER_HEIGHT = 0.20    # fraction of contact depth (softened for MC sampling)
WATER_DEPTH = 0.33       # fraction of contact depth (ε' = 0.33ε)
MIN_SEQ_SEP = 3


def _desolvation_potential(r: float, epsilon: float) -> float:
    """Three-zone desolvation potential for a single pair.

    Zone 1 (r < r0): contact well, depth -epsilon
    Zone 2 (r0 < r < r0 + barrier_width): desolvation barrier, height +epsilon*0.33
    Zone 3 (r0+barrier_width < r < r0+water_shell): water-mediated, depth -epsilon*0.33
    Beyond: zero
    """
    r1 = CONTACT_R0
    r2 = CONTACT_R0 + BARRIER_WIDTH
    r3 = CONTACT_R0 + WATER_SHELL

    if r < r1:
        # Contact well: Gaussian-like
        return -epsilon * np.exp(-0.5 * ((r - r1 + 1.0) / 1.0) ** 2)
    elif r < r2:
        # Desolvation barrier
        frac = (r - r1) / BARRIER_WIDTH
        return epsilon * BARRIER_HEIGHT * np.sin(np.pi * frac)
    elif r < r3:
        # Water-mediated minimum
        frac = (r - r2) / (WATER_SHELL - BARRIER_WIDTH)
        return -epsilon * WATER_DEPTH * np.sin(np.pi * frac)
    else:
        return 0.0


# Vectorized version of desolvation potential
def _desolvation_potential_vec(r: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    """Vectorized three-zone desolvation potential."""
    r1 = CONTACT_R0
    r2 = CONTACT_R0 + BARRIER_WIDTH
    r3 = CONTACT_R0 + WATER_SHELL

    result = np.zeros_like(r)

    # Zone 1: contact well
    z1 = r < r1
    result[z1] = -epsilon[z1] * np.exp(-0.5 * ((r[z1] - r1 + 1.0) / 1.0) ** 2)

    # Zone 2: desolvation barrier
    z2 = (r >= r1) & (r < r2)
    frac2 = (r[z2] - r1) / BARRIER_WIDTH
    result[z2] = epsilon[z2] * BARRIER_HEIGHT * np.sin(np.pi * frac2)

    # Zone 3: water-mediated minimum
    z3 = (r >= r2) & (r < r3)
    frac3 = (r[z3] - r2) / (WATER_SHELL - BARRIER_WIDTH)
    result[z3] = -epsilon[z3] * WATER_DEPTH * np.sin(np.pi * frac3)

    return result


class SolventEnergy(EnergyTerm):
    """Implicit solvation with desolvation barrier.

    Combines:
    1. Pairwise desolvation potential (cooperative folding)
    2. Per-residue burial-based solvation (global hydrophobic collapse)
    """

    @property
    def name(self) -> str:
        return "solvent"

    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        n = len(coords)
        if n < 2:
            return 0.0

        cb = get_cb_coords(coords, sequence)
        tunnel_positions = kwargs.get('tunnel_positions')
        tunnel_length = kwargs.get('tunnel_length', 90.0)

        if tunnel_positions is not None:
            exposed_mask = tunnel_positions > tunnel_length
        else:
            exposed_mask = np.ones(n, dtype=bool)

        if not np.any(exposed_mask):
            return 0.0

        # --- Part 1: Pairwise desolvation potential ---
        hydrophobicity = np.array([
            RESIDUE_PROPERTIES[aa].hydrophobicity for aa in sequence])

        # Pairwise Cβ distances
        diff = cb[:, None, :] - cb[None, :, :]
        dist = np.linalg.norm(diff, axis=2) + 1e-20

        idx = np.arange(n)
        sep = np.abs(idx[:, None] - idx[None, :])
        # Only exposed pairs with sufficient sequence separation
        pair_mask = ((idx[:, None] < idx[None, :]) &
                     (sep >= MIN_SEQ_SEP) &
                     (dist < CONTACT_R0 + WATER_SHELL) &
                     exposed_mask[:, None] & exposed_mask[None, :])

        e_desolv = 0.0
        if np.any(pair_mask):
            # Pairwise hydrophobic interaction strength
            # Geometric mean of hydrophobicities (both must be hydrophobic for strong contact)
            h_i = np.maximum(hydrophobicity[:, None], 0.0)
            h_j = np.maximum(hydrophobicity[None, :], 0.0)
            epsilon_ij = CONTACT_DEPTH * np.sqrt(h_i * h_j + 0.01)

            e_pairs = _desolvation_potential_vec(
                dist[pair_mask], epsilon_ij[pair_mask])
            e_desolv = float(np.sum(e_pairs))

        # --- Part 2: Per-residue burial solvation (unchanged) ---
        dist_burial = dist.copy()
        np.fill_diagonal(dist_burial, BURIAL_CUTOFF + 1)
        neighbor_counts = np.sum(dist_burial < BURIAL_CUTOFF, axis=1)
        exposure = np.maximum(0.0, 1.0 - neighbor_counts / MAX_NEIGHBORS)
        sigma = np.array([SOLVATION_PARAMS.get(aa, 0.0) for aa in sequence])
        e_burial = float(np.sum(sigma[exposed_mask] * exposure[exposed_mask]))

        return e_desolv + e_burial
