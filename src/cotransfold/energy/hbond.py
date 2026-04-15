"""Backbone hydrogen bond energy using the DSSP electrostatic model.

The DSSP model (Kabsch & Sander, 1983) calculates H-bond energy as:

    E = q1 * q2 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 332

where q1 = 0.42e, q2 = 0.20e (partial charges on N-H...O=C groups).
A good hydrogen bond has E < -0.5 kcal/mol.

Key H-bond patterns:
- Alpha helix: i -> i+4 (C=O of residue i bonds to N-H of residue i+4)
- 3_10 helix: i -> i+3
- Beta sheet: long-range i -> j
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.energy.total import EnergyTerm

# DSSP partial charges
Q1 = 0.42   # e, on C=O group
Q2 = 0.20   # e, on N-H group
COULOMB = 332.0  # kcal·Å/(mol·e²)

# Geometric parameters for virtual H and O atoms
BOND_LENGTH_NH = 1.0    # Å, N-H bond
BOND_LENGTH_CO = 1.24   # Å, C=O bond

# Distance cutoffs
MAX_NO_DISTANCE = 5.2   # Å, maximum N-O distance for H-bond consideration
MIN_SEPARATION = 3       # Minimum sequence separation for H-bond (exclude i, i+1, i+2)

# Good H-bond threshold
HBOND_CUTOFF = -0.5  # kcal/mol


def _place_h_atoms_batch(n_atoms: np.ndarray, ca_atoms: np.ndarray,
                         c_atoms: np.ndarray) -> np.ndarray:
    """Place virtual H atoms using trigonal planar geometry.

    The amide N has three bonds (to H, CA, and C of previous residue).
    The trigonal planar geometry constrains H to be anti to the bisector
    of (N→CA) and (N→C_prev), not just anti to CA. The previous version
    used (N - CA) only, which biases H placement by 30-60° and causes
    systematic under-detection of real H-bonds in native structures.

    Returns shape (N, 3). h_atoms[0] is invalid (N-terminus).
    """
    n = len(n_atoms)
    h_atoms = np.zeros((n, 3))
    if n < 2:
        return h_atoms

    # For residues 1..N-1: H direction is anti to (CA + C_prev) from N
    n_minus_ca = n_atoms[1:] - ca_atoms[1:]       # (N-1, 3)
    n_minus_cprev = n_atoms[1:] - c_atoms[:-1]    # (N-1, 3)
    h_dir = n_minus_ca + n_minus_cprev
    h_norm = np.linalg.norm(h_dir, axis=1, keepdims=True)
    h_norm = np.maximum(h_norm, 1e-10)
    h_atoms[1:] = n_atoms[1:] + BOND_LENGTH_NH * h_dir / h_norm
    return h_atoms


def _place_o_atoms_batch(c_atoms: np.ndarray, ca_atoms: np.ndarray,
                         n_atoms: np.ndarray) -> np.ndarray:
    """Place virtual O atoms for all residues at once.

    O is placed along the bisector of CA-C and N(next)-C directions.
    Returns shape (N, 3). o_atoms[-1] is invalid (C-terminus).
    """
    n = len(c_atoms)
    o_atoms = np.zeros((n, 3))

    if n < 2:
        return o_atoms

    # For residues 0..N-2: use N of next residue
    ca_c = c_atoms[:n-1] - ca_atoms[:n-1]  # (N-1, 3)
    n_c = c_atoms[:n-1] - n_atoms[1:n]     # (N-1, 3)

    ca_c_norm = np.linalg.norm(ca_c, axis=1, keepdims=True)
    ca_c_norm = np.maximum(ca_c_norm, 1e-10)
    n_c_norm = np.linalg.norm(n_c, axis=1, keepdims=True)
    n_c_norm = np.maximum(n_c_norm, 1e-10)

    bisector = ca_c / ca_c_norm + n_c / n_c_norm  # (N-1, 3)
    bis_norm = np.linalg.norm(bisector, axis=1, keepdims=True)
    bis_norm = np.maximum(bis_norm, 1e-10)
    bisector_normalized = bisector / bis_norm

    o_atoms[:n-1] = c_atoms[:n-1] + BOND_LENGTH_CO * bisector_normalized
    return o_atoms


class HydrogenBondEnergy(EnergyTerm):
    """Backbone hydrogen bond energy (vectorized).

    Sums over all valid donor-acceptor pairs (sequence separation >= 3).
    Only attractive (negative) energies are counted.
    """

    @property
    def name(self) -> str:
        return "hbond"

    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        n = len(coords)
        if n < MIN_SEPARATION + 1:
            return 0.0

        n_atoms = coords[:, 0]   # (N, 3)
        ca_atoms = coords[:, 1]  # (N, 3)
        c_atoms = coords[:, 2]   # (N, 3)

        h_atoms = _place_h_atoms_batch(n_atoms, ca_atoms, c_atoms)
        o_atoms = _place_o_atoms_batch(c_atoms, ca_atoms, n_atoms)

        # Build pairwise distance matrices for O(i) vs N(j)
        # Acceptor i: C=O of residue i (valid for i in 0..N-2)
        # Donor j: N-H of residue j (valid for j in 1..N-1)
        # Condition: j >= i + MIN_SEPARATION

        # O-N distances: o_atoms[i] vs n_atoms[j]
        # Shape: (N, N) but we only need upper triangle with offset >= MIN_SEPARATION
        diff_on = o_atoms[:, None, :] - n_atoms[None, :, :]  # (N, N, 3)
        r_on = np.linalg.norm(diff_on, axis=2)  # (N, N)

        # Build validity mask
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        mask = (jj >= ii + MIN_SEPARATION)  # sequence separation
        mask &= (ii < n - 1)               # acceptor valid (has next N for O placement)
        mask &= (jj > 0)                   # donor valid (not first residue)
        mask &= (r_on > 2.0) & (r_on < MAX_NO_DISTANCE)  # physical lower bound + cutoff

        if not np.any(mask):
            return 0.0

        # Compute remaining distances only for valid pairs
        acc_idx, don_idx = np.where(mask)

        r_on_valid = r_on[acc_idx, don_idx]
        r_ch = np.linalg.norm(c_atoms[acc_idx] - h_atoms[don_idx], axis=1)
        r_oh = np.linalg.norm(o_atoms[acc_idx] - h_atoms[don_idx], axis=1)
        r_cn = np.linalg.norm(c_atoms[acc_idx] - n_atoms[don_idx], axis=1)

        # Avoid division by zero
        safe = (r_on_valid > 0.01) & (r_ch > 0.01) & (r_oh > 0.01) & (r_cn > 0.01)

        energies = np.zeros(len(acc_idx))
        energies[safe] = (Q1 * Q2 * COULOMB *
                          (1.0 / r_on_valid[safe] + 1.0 / r_ch[safe]
                           - 1.0 / r_oh[safe] - 1.0 / r_cn[safe]))

        # Clamp per-pair to physical range. A real backbone H-bond is ~-2 to
        # -3 kcal/mol; capping at -3 prevents the simulation from over-rewarding
        # geometrically perfect packing where multiple "H-bonds" pile up at the
        # same residue. The previous -5 cap allowed runaway minima.
        energies = np.clip(energies, -3.0, 0.0)

        # Enforce physical exclusivity: each backbone N-H can donate to only
        # ONE H-bond and each C=O can accept only ONE. Without this, tightly
        # packed structures over-count H-bonds and the energy function develops
        # a spurious global minimum where the chain collapses into an
        # artificially H-bond-saturated bundle.
        # For each donor, keep only its strongest (most negative) H-bond.
        # Then for each acceptor, keep only its strongest among the survivors.
        best_per_donor = {}
        for k in range(len(acc_idx)):
            if energies[k] >= 0:
                continue
            d = int(don_idx[k])
            if d not in best_per_donor or energies[k] < energies[best_per_donor[d]]:
                best_per_donor[d] = k
        keep_donor = set(best_per_donor.values())

        best_per_acceptor = {}
        for k in keep_donor:
            a = int(acc_idx[k])
            if a not in best_per_acceptor or energies[k] < energies[best_per_acceptor[a]]:
                best_per_acceptor[a] = k
        keep = set(best_per_acceptor.values())

        kept_mask = np.zeros(len(acc_idx), dtype=bool)
        for k in keep:
            kept_mask[k] = True
        energies = np.where(kept_mask, energies, 0.0)

        # No cooperativity bonus — it amplifies the runaway minimum where
        # the sim packs every residue into maximum H-bond geometry. With
        # cooperativity, long uniform H-bond stretches got up to 1.5x bonus,
        # which combined with the per-pair cap allowed the simulation to find
        # spurious global minima with ~2x the H-bond energy of native folds.

        return float(np.sum(energies))
