"""Torsion angle to Cartesian coordinate conversion using the NeRF algorithm.

The Natural Extension Reference Frame (NeRF) algorithm places each new atom
by extending the chain using the known bond length, bond angle, and torsion
angle relative to the three preceding atoms.

Reference:
    Parsons et al. (2005) "Practical conversion from torsion space to
    Cartesian space for in silico protein structure prediction."
    J Comput Chem 26(10):1063-1068.
"""

import numpy as np
from cotransfold.core.conformation import BackboneState
from cotransfold.config import (
    BOND_LENGTH_N_CA, BOND_LENGTH_CA_C, BOND_LENGTH_C_N,
    BOND_ANGLE_N_CA_C, BOND_ANGLE_CA_C_N, BOND_ANGLE_C_N_CA,
)


def _place_atom(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                bond_length: float, bond_angle: float,
                torsion: float) -> np.ndarray:
    """Place a new atom D given three preceding atoms A, B, C.

    D is placed at distance `bond_length` from C, with angle B-C-D
    equal to `bond_angle`, and torsion A-B-C-D equal to `torsion`.

    Args:
        a, b, c: Cartesian coordinates of three preceding atoms (shape (3,))
        bond_length: distance C-D (Å)
        bond_angle: angle B-C-D (radians)
        torsion: dihedral A-B-C-D (radians)

    Returns:
        Cartesian coordinates of atom D (shape (3,))
    """
    # Vectors along the chain
    bc = c - b
    bc_norm = bc / np.linalg.norm(bc)

    # Build a coordinate frame at atom C
    ab = b - a
    n = np.cross(ab, bc_norm)
    n_norm = n / np.linalg.norm(n)

    # Third basis vector
    m = np.cross(n_norm, bc_norm)

    # Spherical-to-Cartesian in local frame
    d_x = -bond_length * np.cos(bond_angle)
    d_y = bond_length * np.sin(bond_angle) * np.cos(torsion)
    d_z = bond_length * np.sin(bond_angle) * np.sin(torsion)

    # Transform to global frame
    d = c + d_x * bc_norm + d_y * m + d_z * n_norm
    return d


def torsion_to_cartesian(state: BackboneState) -> np.ndarray:
    """Convert backbone torsion angles to Cartesian coordinates.

    The backbone atoms cycle as: N - CA - C - N - CA - C - ...

    For residue i:
    - omega[i]: torsion around C(i-1)-N(i) bond (CA(i-1)-C(i-1)-N(i)-CA(i))
    - phi[i]:   torsion around N(i)-CA(i) bond (C(i-1)-N(i)-CA(i)-C(i))
    - psi[i]:   torsion around CA(i)-C(i) bond (N(i)-CA(i)-C(i)-N(i+1))

    Returns:
        coords: array of shape (N, 3, 3) where coords[i] = [N_i, CA_i, C_i]
                each with shape (3,) for x, y, z
    """
    n_res = state.num_residues
    if n_res == 0:
        return np.zeros((0, 3, 3))

    # Atom placement sequence and corresponding geometry
    # We build a flat list of all backbone atoms, then reshape.
    # Atom order: N0, CA0, C0, N1, CA1, C1, N2, CA2, C2, ...
    #
    # The bond/angle/torsion for placing each atom:
    #   CA0: bond=N-CA, angle=C(-1)-N0-CA0,  torsion=omega[0]  (first residue: use defaults)
    #   C0:  bond=CA-C,  angle=N0-CA0-C0,     torsion=phi[0]    (but phi[0] is undefined)
    #   N1:  bond=C-N,   angle=CA0-C0-N1,     torsion=psi[0]
    #   CA1: bond=N-CA,  angle=C0-N1-CA1,     torsion=omega[1]
    #   C1:  bond=CA-C,  angle=N1-CA1-C1,     torsion=phi[1]
    #   N2:  bond=C-N,   angle=CA1-C1-N2,     torsion=psi[1]
    #   ...

    total_atoms = n_res * 3
    coords = np.zeros((total_atoms, 3))

    # Place first three atoms (N0, CA0, C0) in a canonical frame
    # N0 at origin
    coords[0] = [0.0, 0.0, 0.0]
    # CA0 along x-axis
    coords[1] = [BOND_LENGTH_N_CA, 0.0, 0.0]
    # C0 using bond angle N-CA-C
    angle_n_ca_c = BOND_ANGLE_N_CA_C
    coords[2] = [
        BOND_LENGTH_N_CA + BOND_LENGTH_CA_C * np.cos(np.pi - angle_n_ca_c),
        BOND_LENGTH_CA_C * np.sin(np.pi - angle_n_ca_c),
        0.0,
    ]

    # Now place remaining atoms using NeRF
    # For each subsequent atom, we need the three preceding atoms and the
    # appropriate bond length, bond angle, and torsion angle.
    #
    # Atom index in flat array:
    #   residue i, atom_type j (0=N, 1=CA, 2=C) -> index = 3*i + j
    #
    # Starting from atom index 3 (N of residue 1):
    for atom_idx in range(3, total_atoms):
        res_idx = atom_idx // 3
        atom_type = atom_idx % 3  # 0=N, 1=CA, 2=C

        a = coords[atom_idx - 3]
        b = coords[atom_idx - 2]
        c = coords[atom_idx - 1]

        if atom_type == 0:  # Placing N(i)
            # Previous atoms: N(i-1), CA(i-1), C(i-1)
            # Bond: C(i-1)-N(i) = C_N
            # Angle: CA(i-1)-C(i-1)-N(i) = CA_C_N
            # Torsion: psi[i-1]
            bond_length = BOND_LENGTH_C_N
            bond_angle = BOND_ANGLE_CA_C_N
            torsion = state.psi[res_idx - 1]

        elif atom_type == 1:  # Placing CA(i)
            # Previous atoms: CA(i-1), C(i-1), N(i)
            # Bond: N(i)-CA(i) = N_CA
            # Angle: C(i-1)-N(i)-CA(i) = C_N_CA
            # Torsion: omega[i]
            bond_length = BOND_LENGTH_N_CA
            bond_angle = BOND_ANGLE_C_N_CA
            torsion = state.omega[res_idx]

        else:  # atom_type == 2, Placing C(i)
            # Previous atoms: C(i-1), N(i), CA(i)
            # Bond: CA(i)-C(i) = CA_C
            # Angle: N(i)-CA(i)-C(i) = N_CA_C
            # Torsion: phi[i]
            bond_length = BOND_LENGTH_CA_C
            bond_angle = BOND_ANGLE_N_CA_C
            torsion = state.phi[res_idx]

        coords[atom_idx] = _place_atom(a, b, c, bond_length, bond_angle, torsion)

    return coords.reshape(n_res, 3, 3)


def get_ca_coords(coords: np.ndarray) -> np.ndarray:
    """Extract CA coordinates from full backbone coordinates.

    Args:
        coords: shape (N, 3, 3) from torsion_to_cartesian

    Returns:
        CA coordinates, shape (N, 3)
    """
    return coords[:, 1, :]


def compute_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def compute_end_to_end(coords: np.ndarray) -> float:
    """End-to-end distance of the chain (first N to last C)."""
    if len(coords) == 0:
        return 0.0
    return compute_distance(coords[0, 0], coords[-1, 2])


def compute_rise_per_residue(coords: np.ndarray) -> float:
    """Average rise per residue along the chain axis (CA-CA distance)."""
    ca = get_ca_coords(coords)
    if len(ca) < 2:
        return 0.0
    distances = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    return float(np.mean(distances))
