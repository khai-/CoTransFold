"""Secondary structure assignment from backbone torsion angles.

Assigns secondary structure types based on Ramachandran region:
- H (alpha-helix): phi ~ -57°, psi ~ -47°
- E (beta-strand): phi ~ -120°, psi ~ 130°
- C (coil/other): everything else

Also detects:
- G (3_10 helix): phi ~ -49°, psi ~ -26°
- P (PPII helix): phi ~ -75°, psi ~ 145°
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState


def _angle_distance(a: float, b: float) -> float:
    """Shortest angular distance on the circle (radians)."""
    d = a - b
    return abs(d - 2 * np.pi * round(d / (2 * np.pi)))


def assign_secondary_structure(backbone: BackboneState,
                               tolerance: float = 35.0) -> str:
    """Assign secondary structure from backbone torsion angles.

    Args:
        backbone: BackboneState with phi/psi angles
        tolerance: angular tolerance in degrees for region matching

    Returns:
        String of length N with characters: H (helix), E (strand), C (coil)
    """
    tol = np.radians(tolerance)
    n = backbone.num_residues
    ss = []

    for i in range(n):
        phi = backbone.phi[i]
        psi = backbone.psi[i]

        # Alpha helix: phi=-57°, psi=-47°
        if (_angle_distance(phi, np.radians(-57)) < tol and
                _angle_distance(psi, np.radians(-47)) < tol):
            ss.append('H')
        # Beta strand: phi=-120°, psi=130°
        elif (_angle_distance(phi, np.radians(-120)) < tol and
              _angle_distance(psi, np.radians(130)) < tol):
            ss.append('E')
        # PPII: phi=-75°, psi=145°
        elif (_angle_distance(phi, np.radians(-75)) < tol and
              _angle_distance(psi, np.radians(145)) < tol):
            ss.append('E')  # Classify PPII as extended
        else:
            ss.append('C')

    return ''.join(ss)


def helix_fraction(ss: str) -> float:
    """Fraction of residues in helical conformation."""
    if len(ss) == 0:
        return 0.0
    return ss.count('H') / len(ss)


def strand_fraction(ss: str) -> float:
    """Fraction of residues in strand/extended conformation."""
    if len(ss) == 0:
        return 0.0
    return ss.count('E') / len(ss)


def coil_fraction(ss: str) -> float:
    """Fraction of residues in coil."""
    if len(ss) == 0:
        return 0.0
    return ss.count('C') / len(ss)


def longest_helix(ss: str) -> int:
    """Length of the longest continuous helix segment."""
    max_len = 0
    current = 0
    for c in ss:
        if c == 'H':
            current += 1
            max_len = max(max_len, current)
        else:
            current = 0
    return max_len


def ss_summary(ss: str) -> str:
    """Human-readable secondary structure summary."""
    return (
        f"SS: {ss}\n"
        f"Helix: {helix_fraction(ss)*100:.1f}% | "
        f"Strand: {strand_fraction(ss)*100:.1f}% | "
        f"Coil: {coil_fraction(ss)*100:.1f}%\n"
        f"Longest helix: {longest_helix(ss)} residues"
    )
