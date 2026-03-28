"""Residue-pair statistical potential.

Knowledge-based pairwise potential derived from amino acid contact preferences
in the PDB. Similar to Rosetta's 'pair' term and Miyazawa-Jernigan contact
energies. Encodes sequence-specific distance preferences between residue types.

E_pair = sum_{i<j, |i-j|>=3} e(aa_i, aa_j) * f(r_ij)

where e(aa_i, aa_j) is the contact energy and f(r) is a distance-dependent
weighting that peaks at typical contact distance (~6-8 Å for Cβ-Cβ).
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.core.residue import AminoAcid
from cotransfold.energy.total import EnergyTerm
from cotransfold.structure.coordinates import get_cb_coords

# Simplified Miyazawa-Jernigan contact energies (kcal/mol)
# Negative = favorable contact, positive = unfavorable
# Grouped by residue class for efficiency
# Hydrophobic-hydrophobic contacts are favorable
# Charged-charged same sign unfavorable, opposite favorable

_HP = {  # Hydrophobic residues
    AminoAcid.ALA, AminoAcid.VAL, AminoAcid.LEU, AminoAcid.ILE,
    AminoAcid.PHE, AminoAcid.TRP, AminoAcid.MET, AminoAcid.PRO,
}
_POS = {AminoAcid.ARG, AminoAcid.LYS, AminoAcid.HIS}
_NEG = {AminoAcid.ASP, AminoAcid.GLU}
_POLAR = {AminoAcid.SER, AminoAcid.THR, AminoAcid.ASN, AminoAcid.GLN,
          AminoAcid.TYR, AminoAcid.CYS}


def _contact_energy(aa_i: AminoAcid, aa_j: AminoAcid) -> float:
    """Simplified contact energy between two residue types."""
    # Hydrophobic-hydrophobic: strongly favorable
    if aa_i in _HP and aa_j in _HP:
        return -1.0

    # Salt bridges: opposite charges strongly favorable
    if (aa_i in _POS and aa_j in _NEG) or (aa_i in _NEG and aa_j in _POS):
        return -1.5

    # Same-charge repulsion
    if (aa_i in _POS and aa_j in _POS) or (aa_i in _NEG and aa_j in _NEG):
        return 0.5

    # Hydrophobic-polar: mildly unfavorable
    if (aa_i in _HP and aa_j not in _HP) or (aa_j in _HP and aa_i not in _HP):
        return 0.2

    # Cysteine-cysteine: disulfide potential
    if aa_i == AminoAcid.CYS and aa_j == AminoAcid.CYS:
        return -2.0

    # Aromatic stacking (Phe-Phe, Phe-Tyr, Trp-Phe, etc.)
    _AROM = {AminoAcid.PHE, AminoAcid.TYR, AminoAcid.TRP}
    if aa_i in _AROM and aa_j in _AROM:
        return -1.2

    # Default: weakly favorable (general vdW attraction)
    return -0.1


# Precompute contact energy matrix for all 20x20 pairs
_AA_LIST = list(AminoAcid)
_N_AA = len(_AA_LIST)
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_LIST)}
_CONTACT_MATRIX = np.zeros((_N_AA, _N_AA))
for _i, _aa_i in enumerate(_AA_LIST):
    for _j, _aa_j in enumerate(_AA_LIST):
        _CONTACT_MATRIX[_i, _j] = _contact_energy(_aa_i, _aa_j)

# Distance parameters
CONTACT_DIST = 7.0     # Å, optimal Cβ-Cβ contact distance
CONTACT_WIDTH = 2.0    # Å, Gaussian width
CONTACT_CUTOFF = 12.0  # Å, beyond which potential is zero
MIN_SEQ_SEP = 3


class PairPotentialEnergy(EnergyTerm):
    """Knowledge-based residue-pair contact potential."""

    @property
    def name(self) -> str:
        return "pair_potential"

    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        n = len(coords)
        if n < MIN_SEQ_SEP + 1:
            return 0.0

        cb = get_cb_coords(coords, sequence)

        # Pairwise Cβ distances
        diff = cb[:, None, :] - cb[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        # Sequence separation mask
        idx = np.arange(n)
        sep = np.abs(idx[:, None] - idx[None, :])
        mask = (idx[:, None] < idx[None, :]) & (sep >= MIN_SEQ_SEP) & (dist < CONTACT_CUTOFF)

        if not np.any(mask):
            return 0.0

        # Contact energy matrix for this sequence
        aa_idx = np.array([_AA_TO_IDX[aa] for aa in sequence])
        e_ij = _CONTACT_MATRIX[aa_idx[:, None], aa_idx[None, :]]

        # Distance-dependent weighting: Gaussian centered at CONTACT_DIST
        f_r = np.exp(-0.5 * ((dist - CONTACT_DIST) / CONTACT_WIDTH) ** 2)

        # Sum over valid pairs
        return float(np.sum(e_ij[mask] * f_r[mask]))
