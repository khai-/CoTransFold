"""PDB file input/output for backbone structures."""

from __future__ import annotations

import numpy as np
from cotransfold.core.residue import AminoAcid, THREE_LETTER


BACKBONE_ATOM_NAMES = ['N', 'CA', 'C']


def write_pdb(filepath: str,
              coords: np.ndarray,
              sequence: list[AminoAcid],
              chain_id: str = 'A',
              model_num: int | None = None) -> None:
    """Write backbone coordinates to a PDB file.

    Args:
        filepath: output file path
        coords: shape (N, 3, 3) — [N, CA, C] for each residue
        sequence: amino acid sequence
        chain_id: PDB chain identifier
        model_num: optional MODEL number for multi-model files
    """
    assert len(coords) == len(sequence), (
        f"coords ({len(coords)}) and sequence ({len(sequence)}) length mismatch")

    lines = []
    if model_num is not None:
        lines.append(f"MODEL     {model_num:4d}")

    atom_serial = 1
    for res_idx, (res_coords, aa) in enumerate(zip(coords, sequence)):
        res_name = THREE_LETTER[aa]
        res_num = res_idx + 1

        for atom_idx, atom_name in enumerate(BACKBONE_ATOM_NAMES):
            x, y, z = res_coords[atom_idx]
            # PDB ATOM record format (fixed-width columns)
            line = (
                f"ATOM  {atom_serial:5d} {atom_name:<4s} {res_name:3s} "
                f"{chain_id}{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}          "
                f"{atom_name[0]:>2s}"
            )
            lines.append(line)
            atom_serial += 1

    if model_num is not None:
        lines.append("ENDMDL")
    lines.append("END")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_trajectory_pdb(filepath: str,
                         trajectory_coords: list[np.ndarray],
                         sequence: list[AminoAcid],
                         chain_id: str = 'A') -> None:
    """Write multiple frames as a multi-model PDB file.

    Args:
        filepath: output file path
        trajectory_coords: list of coord arrays, each shape (N_i, 3, 3)
            where N_i grows as residues are added
        sequence: full amino acid sequence
        chain_id: PDB chain identifier
    """
    lines = []
    for model_idx, coords in enumerate(trajectory_coords):
        n_res = len(coords)
        lines.append(f"MODEL     {model_idx + 1:4d}")

        atom_serial = 1
        for res_idx in range(n_res):
            aa = sequence[res_idx]
            res_name = THREE_LETTER[aa]
            res_num = res_idx + 1

            for atom_idx, atom_name in enumerate(BACKBONE_ATOM_NAMES):
                x, y, z = coords[res_idx, atom_idx]
                line = (
                    f"ATOM  {atom_serial:5d} {atom_name:<4s} {res_name:3s} "
                    f"{chain_id}{res_num:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{1.00:6.2f}{0.00:6.2f}          "
                    f"{atom_name[0]:>2s}"
                )
                lines.append(line)
                atom_serial += 1

        lines.append("ENDMDL")

    lines.append("END")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
