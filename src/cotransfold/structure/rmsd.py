"""RMSD and TM-score for structural comparison.

RMSD: Root Mean Square Deviation after optimal superposition (Kabsch algorithm).
TM-score: Template Modeling score — size-independent structural similarity (0-1).

References:
- Kabsch (1976) "A solution for the best rotation to relate two sets of vectors"
- Zhang & Skolnick (2004) "Scoring function for automated assessment of
  protein structure template quality"
"""

from __future__ import annotations

import numpy as np


def rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two sets of coordinates WITHOUT superposition.

    Args:
        coords1, coords2: arrays of shape (N, 3)

    Returns:
        RMSD in Å
    """
    assert coords1.shape == coords2.shape
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def kabsch_superpose(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Optimally superpose mobile onto target using the Kabsch algorithm.

    Finds the rotation and translation that minimizes RMSD.

    Args:
        mobile: coordinates to move, shape (N, 3)
        target: reference coordinates, shape (N, 3)

    Returns:
        (superposed_mobile, rmsd_value)
    """
    assert mobile.shape == target.shape
    n = len(mobile)

    # Center both coordinate sets
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_centered = mobile - mobile_center
    target_centered = target - target_center

    # Compute cross-covariance matrix
    H = mobile_centered.T @ target_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])

    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and translation
    superposed = (mobile_centered @ R.T) + target_center

    # Compute RMSD
    diff = superposed - target
    rmsd_val = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    return superposed, rmsd_val


def superposed_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """RMSD after optimal superposition.

    Args:
        coords1, coords2: arrays of shape (N, 3)

    Returns:
        RMSD in Å after Kabsch superposition
    """
    _, rmsd_val = kabsch_superpose(coords1, coords2)
    return rmsd_val


def tm_score(coords1: np.ndarray, coords2: np.ndarray,
             l_target: int | None = None) -> float:
    """Compute TM-score between two structures.

    TM-score is length-independent and ranges from 0 to 1.
    >0.5 generally indicates same fold; >0.17 is better than random.

    Args:
        coords1: predicted coordinates, shape (N, 3)
        coords2: reference coordinates, shape (N, 3)
        l_target: target length for normalization (default: len(coords2))

    Returns:
        TM-score (0 to 1)
    """
    assert coords1.shape == coords2.shape
    n = len(coords1)

    if l_target is None:
        l_target = n

    if l_target <= 0:
        return 0.0

    # TM-score distance scale
    l_eff = max(l_target - 15, 0)
    d0 = 1.24 * (l_eff ** (1.0 / 3.0)) - 1.8
    d0 = max(d0, 0.5)  # Minimum d0

    # Superpose first
    superposed, _ = kabsch_superpose(coords1, coords2)

    # Compute TM-score
    distances = np.linalg.norm(superposed - coords2, axis=1)
    tm = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / l_target

    return float(tm)
