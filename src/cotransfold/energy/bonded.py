"""Bonded energy terms: dihedral angle potentials.

In the torsion angle representation, bond lengths and bond angles are
fixed by the NeRF algorithm. The only bonded energy that matters is
the intrinsic torsion potential for the omega angle (peptide bond planarity).

E_omega = sum_i k_omega * (1 - cos(omega_i - pi))

This penalizes deviation from the trans peptide bond (omega = pi).
Cis peptide bonds (omega = 0) are rare except for X-Pro bonds.
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.core.residue import AminoAcid
from cotransfold.energy.total import EnergyTerm

# Force constant for omega planarity restraint
K_OMEGA = 10.0  # kcal/mol, strong preference for trans


class BondedEnergy(EnergyTerm):
    """Peptide bond planarity (omega angle) energy.

    Penalizes deviation from trans (omega = pi).
    For X-Pro bonds, both trans and cis are allowed.
    """

    @property
    def name(self) -> str:
        return "bonded"

    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        n = backbone.num_residues
        if n == 0:
            return 0.0

        energy = 0.0
        for i in range(n):
            omega = backbone.omega[i]
            if i + 1 < n and sequence[i + 1] == AminoAcid.PRO:
                # X-Pro: allow both trans (pi) and cis (0)
                # Use a double-well potential
                e_trans = K_OMEGA * (1.0 - np.cos(omega - np.pi))
                e_cis = K_OMEGA * (1.0 - np.cos(omega))
                energy += min(e_trans, e_cis)
            else:
                # Standard: strong preference for trans
                energy += K_OMEGA * (1.0 - np.cos(omega - np.pi))
        return energy
