"""Parse PDB files to extract backbone CA coordinates.

Reads experimental structures from PDB files (downloaded from RCSB)
and extracts CA atom coordinates for structural comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PDBStructure:
    """Parsed PDB structure with CA coordinates."""
    pdb_id: str
    chain_id: str
    sequence: str               # One-letter amino acid sequence
    ca_coords: np.ndarray       # Shape (N, 3), CA coordinates in Å
    resolution: float | None    # Resolution in Å (from REMARK 2)
    n_residues: int

    @property
    def length(self) -> int:
        return self.n_residues


# Standard 3-letter to 1-letter mapping
_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # Common modified residues mapped to standard
    'MSE': 'M', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
}


def parse_pdb(filepath: str, chain_id: str = 'A') -> PDBStructure:
    """Parse a PDB file and extract CA coordinates for a specific chain.

    Args:
        filepath: path to PDB file
        chain_id: chain identifier to extract

    Returns:
        PDBStructure with CA coordinates and sequence
    """
    ca_coords = []
    sequence = []
    resolution = None
    pdb_id = ''
    seen_residues = set()

    with open(filepath) as f:
        for line in f:
            # PDB ID from HEADER
            if line.startswith('HEADER'):
                pdb_id = line[62:66].strip()

            # Resolution from REMARK 2
            if line.startswith('REMARK   2 RESOLUTION.'):
                try:
                    res_str = line[23:30].strip()
                    if res_str and res_str != 'NOT':
                        resolution = float(res_str)
                except (ValueError, IndexError):
                    pass

            # ATOM records for CA atoms
            if (line.startswith('ATOM') or line.startswith('HETATM')):
                atom_name = line[12:16].strip()
                chain = line[21]
                res_name = line[17:20].strip()
                res_seq = line[22:27].strip()  # Residue sequence number + insertion code

                if atom_name == 'CA' and chain == chain_id:
                    # Skip duplicate residues (alternate conformations)
                    res_key = f"{chain}_{res_seq}"
                    if res_key in seen_residues:
                        continue
                    seen_residues.add(res_key)

                    # Skip non-standard residues we can't map
                    one_letter = _THREE_TO_ONE.get(res_name)
                    if one_letter is None:
                        continue

                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append([x, y, z])
                    sequence.append(one_letter)

    ca_array = np.array(ca_coords) if ca_coords else np.zeros((0, 3))
    seq_str = ''.join(sequence)

    return PDBStructure(
        pdb_id=pdb_id,
        chain_id=chain_id,
        sequence=seq_str,
        ca_coords=ca_array,
        resolution=resolution,
        n_residues=len(sequence),
    )


def parse_pdb_all_chains(filepath: str) -> dict[str, PDBStructure]:
    """Parse all chains from a PDB file.

    Returns:
        Dictionary mapping chain_id -> PDBStructure
    """
    # First pass: find all chain IDs
    chain_ids = set()
    with open(filepath) as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21]
                chain_ids.add(chain)

    # Parse each chain
    result = {}
    for chain_id in sorted(chain_ids):
        structure = parse_pdb(filepath, chain_id)
        if structure.n_residues > 0:
            result[chain_id] = structure

    return result
