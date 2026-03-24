"""Implicit solvation energy.

Hydrophobic residues pay an energy penalty when exposed to solvent.
This drives the hydrophobic collapse that is central to protein folding.

Model: for each exposed residue, the solvation energy is proportional
to its hydrophobicity and its degree of solvent exposure (estimated
from the number of nearby CA atoms — more neighbors = more buried).

E_solvent = sum_i sigma_i * f_exposed(i)

where:
- sigma_i = solvation parameter (positive for hydrophobic = penalty when exposed)
- f_exposed(i) = fraction of surface exposed to solvent (0 = fully buried, 1 = fully exposed)

This is a simplified version of the EEF1 model (Lazaridis & Karplus, 1999).
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.core.residue import AminoAcid, RESIDUE_PROPERTIES
from cotransfold.energy.total import EnergyTerm
from cotransfold.structure.coordinates import get_ca_coords

# Solvation parameters (kcal/mol) — positive means cost of solvent exposure
# Hydrophobic residues have high solvation cost (want to be buried)
# Hydrophilic residues have low or negative cost (happy in solvent)
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

# Distance cutoff for counting neighbors (burial estimation)
BURIAL_CUTOFF = 8.0  # Å
# Number of neighbors for "fully buried"
MAX_NEIGHBORS = 10


class SolventEnergy(EnergyTerm):
    """Implicit solvation energy based on hydrophobic burial.

    Hydrophobic residues that are exposed to solvent (few neighbors)
    incur an energy penalty, driving hydrophobic collapse.
    """

    @property
    def name(self) -> str:
        return "solvent"

    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        n = len(coords)
        if n < 2:
            return 0.0

        ca = get_ca_coords(coords)
        tunnel_positions = kwargs.get('tunnel_positions')

        energy = 0.0
        for i in range(n):
            # Skip residues inside the tunnel (no solvent contact)
            if tunnel_positions is not None:
                tunnel_length = kwargs.get('tunnel_length', 90.0)
                if tunnel_positions[i] <= tunnel_length:
                    continue

            # Count neighbors within cutoff (burial proxy)
            n_neighbors = 0
            for j in range(n):
                if i == j:
                    continue
                dist = np.linalg.norm(ca[i] - ca[j])
                if dist < BURIAL_CUTOFF:
                    n_neighbors += 1

            # Exposure fraction: 1 = fully exposed, 0 = fully buried
            exposure = max(0.0, 1.0 - n_neighbors / MAX_NEIGHBORS)

            # Solvation cost
            sigma = SOLVATION_PARAMS.get(sequence[i], 0.0)
            energy += sigma * exposure

        return energy
