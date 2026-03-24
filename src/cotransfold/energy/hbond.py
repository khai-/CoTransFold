"""Backbone hydrogen bond energy using the DSSP electrostatic model.

The DSSP model (Kabsch & Sander, 1983) calculates H-bond energy as:

    E = q1 * q2 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 332

where q1 = 0.42e, q2 = 0.20e (partial charges on N-H...O=C groups).
A good hydrogen bond has E < -0.5 kcal/mol.

In backbone-only representation:
- The H atom is placed along N->CA direction, 1.0Å from N
- The O atom is placed in the C=O direction, 1.24Å from C
  (approximately opposite the CA->C->N bisector)

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


def _place_h_atom(n_pos: np.ndarray, ca_pos: np.ndarray) -> np.ndarray:
    """Place virtual H atom along N->CA direction, 1.0Å from N.

    In reality H is roughly opposite the CA direction from N,
    but for backbone-only model this is a reasonable approximation.
    The H is placed opposite to CA relative to N.
    """
    ca_n = n_pos - ca_pos
    ca_n_norm = ca_n / np.linalg.norm(ca_n)
    return n_pos + BOND_LENGTH_NH * ca_n_norm


def _place_o_atom(c_pos: np.ndarray, ca_pos: np.ndarray,
                  n_next_pos: np.ndarray) -> np.ndarray:
    """Place virtual O atom in the C=O direction.

    O is roughly opposite the bisector of CA-C-N(next).
    """
    ca_c = c_pos - ca_pos
    n_c = c_pos - n_next_pos
    bisector = ca_c / np.linalg.norm(ca_c) + n_c / np.linalg.norm(n_c)
    if np.linalg.norm(bisector) < 1e-10:
        # Degenerate: place along CA-C direction
        bisector = ca_c
    bisector_norm = bisector / np.linalg.norm(bisector)
    return c_pos + BOND_LENGTH_CO * bisector_norm


def _dssp_energy(r_on: float, r_ch: float,
                 r_oh: float, r_cn: float) -> float:
    """Compute DSSP hydrogen bond energy from four distances."""
    if r_on < 0.01 or r_ch < 0.01 or r_oh < 0.01 or r_cn < 0.01:
        return 0.0  # Avoid division by zero
    e = Q1 * Q2 * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn) * COULOMB
    return min(e, 0.0)  # Only attractive (negative) values count


class HydrogenBondEnergy(EnergyTerm):
    """Backbone hydrogen bond energy.

    Sums over all valid donor-acceptor pairs (sequence separation >= 3).
    Only attractive (negative) energies are counted.
    """

    @property
    def name(self) -> str:
        return "hbond"

    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        """Compute total H-bond energy.

        Args:
            coords: shape (N, 3, 3) for [N, CA, C] per residue
        """
        n = len(coords)
        if n < MIN_SEPARATION + 1:
            return 0.0

        # Extract backbone atom positions
        n_atoms = coords[:, 0]   # N positions
        ca_atoms = coords[:, 1]  # CA positions
        c_atoms = coords[:, 2]   # C positions

        # Place virtual H atoms (on residues 1..N-1, since residue 0 has no preceding C)
        h_atoms = np.zeros((n, 3))
        for i in range(1, n):
            h_atoms[i] = _place_h_atom(n_atoms[i], ca_atoms[i])

        # Place virtual O atoms (on residues 0..N-2, since last residue has no next N)
        o_atoms = np.zeros((n, 3))
        for i in range(n - 1):
            o_atoms[i] = _place_o_atom(c_atoms[i], ca_atoms[i], n_atoms[i + 1])

        total_energy = 0.0

        # Check all donor(j)-acceptor(i) pairs where j > i + MIN_SEPARATION
        # Donor: N-H of residue j
        # Acceptor: C=O of residue i
        for i in range(n - 1):  # acceptor C=O
            for j in range(i + MIN_SEPARATION, n):  # donor N-H
                if j == 0:
                    continue  # No H on first residue

                # Quick distance check (N-O)
                r_on = np.linalg.norm(o_atoms[i] - n_atoms[j])
                if r_on > MAX_NO_DISTANCE:
                    continue

                r_ch = np.linalg.norm(c_atoms[i] - h_atoms[j])
                r_oh = np.linalg.norm(o_atoms[i] - h_atoms[j])
                r_cn = np.linalg.norm(c_atoms[i] - n_atoms[j])

                e = _dssp_energy(r_on, r_ch, r_oh, r_cn)
                total_energy += e

        return total_energy
