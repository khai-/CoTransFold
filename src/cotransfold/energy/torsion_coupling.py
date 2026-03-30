"""Multibody torsion correlation energy (inspired by UNRES U_corr).

Captures the coupling between backbone torsion angles and electrostatic
interactions that stabilizes secondary structure cooperatively. This is
a 3-4 body term that cannot be decomposed into pairwise interactions.

Key physics: consecutive residues in helical conformation create aligned
backbone dipoles whose interaction reinforces the helical state. Similarly,
consecutive strand residues create cooperative sheet stabilization.

Three components:
1. Pairwise torsion consistency (original): rewards consecutive SS
2. Three-body dipole-torsion coupling: helix dipole alignment enhances
   helical torsion preference
3. Four-body turn correlation: rewards proper turn patterns between
   SS elements

Reference: Liwo et al., UNRES cumulant-based multibody terms
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.energy.total import EnergyTerm

# Ideal torsion angles
HELIX_PHI = np.radians(-63.0)
HELIX_PSI = np.radians(-42.0)
STRAND_PHI = np.radians(-120.0)
STRAND_PSI = np.radians(130.0)
TURN_PHI_1 = np.radians(-60.0)
TURN_PSI_1 = np.radians(-30.0)
TURN_PHI_2 = np.radians(-90.0)
TURN_PSI_2 = np.radians(0.0)


def _angle_diff(a, b):
    """Smallest signed angular difference (vectorized)."""
    d = a - b
    return d - 2 * np.pi * np.round(d / (2 * np.pi))


def _ss_membership(phi, psi):
    """Compute soft membership in helix/strand/turn states.

    Returns (helix_score, strand_score, turn_score) each in [0, 1].
    """
    # Helix membership
    d_h = (_angle_diff(phi, HELIX_PHI) ** 2 + _angle_diff(psi, HELIX_PSI) ** 2)
    helix = np.exp(-d_h / (2 * np.radians(30) ** 2))

    # Strand membership — wider basin for MC acceptance
    d_s = (_angle_diff(phi, STRAND_PHI) ** 2 + _angle_diff(psi, STRAND_PSI) ** 2)
    strand = np.exp(-d_s / (2 * np.radians(35) ** 2))

    # Turn membership (type I turn, position i+1)
    d_t = (_angle_diff(phi, TURN_PHI_1) ** 2 + _angle_diff(psi, TURN_PSI_1) ** 2)
    turn = np.exp(-d_t / (2 * np.radians(35) ** 2))

    return helix, strand, turn


class TorsionCouplingEnergy(EnergyTerm):
    """Multibody torsion correlation with dipole-torsion coupling."""

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

        # Compute SS membership for all residues
        h, s, t = _ss_membership(phi, psi)

        energy = 0.0

        # --- Component 1: Pairwise torsion consistency ---
        # Reward consecutive residues in same SS state
        for i in range(n - 1):
            # Helix-helix pair
            energy -= 0.4 * h[i] * h[i + 1]
            # Strand-strand pair
            energy -= 0.3 * s[i] * s[i + 1]

        # --- Component 2: Three-body helix cooperativity ---
        # Three consecutive helical residues are more than 3x as stable
        # as three independent helical residues (cooperative enhancement)
        for i in range(n - 2):
            helix_triple = h[i] * h[i + 1] * h[i + 2]
            energy -= 0.5 * helix_triple  # Extra cooperative bonus

            strand_triple = s[i] * s[i + 1] * s[i + 2]
            energy -= 0.3 * strand_triple

        # --- Component 3: Four-body helix nucleation ---
        # Four consecutive helix residues = nucleation threshold
        # Strong cooperative bonus for nucleation
        for i in range(n - 3):
            helix_quad = h[i] * h[i + 1] * h[i + 2] * h[i + 3]
            energy -= 0.8 * helix_quad  # Strong nucleation bonus

        # --- Component 4: Turn patterns ---
        # Reward helix-turn-helix and strand-turn-strand motifs
        for i in range(n - 4):
            # Helix-turn-helix: H H T H H pattern at i, i+1, i+2, i+3, i+4
            hth = h[i] * h[i + 1] * t[i + 2] * h[i + 3] * h[i + 4]
            energy -= 0.3 * hth

        # --- Component 5: Penalize helix-strand transitions ---
        # Direct helix→strand transitions are rare; usually need a turn/coil
        for i in range(n - 1):
            h_to_s = h[i] * s[i + 1]
            s_to_h = s[i] * h[i + 1]
            energy += 0.2 * (h_to_s + s_to_h)  # Penalty

        return energy
