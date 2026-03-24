"""Tests for RMSD and TM-score."""

import numpy as np
from cotransfold.structure.rmsd import rmsd, superposed_rmsd, tm_score, kabsch_superpose


def test_rmsd_identical():
    coords = np.random.RandomState(42).randn(10, 3)
    assert rmsd(coords, coords) == 0.0


def test_rmsd_known():
    c1 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    c2 = np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=float)
    # Each point shifted by 1Å in y -> RMSD = 1.0
    np.testing.assert_allclose(rmsd(c1, c2), 1.0, atol=1e-10)


def test_kabsch_translation_only():
    """Pure translation should give RMSD=0 after superposition."""
    rng = np.random.RandomState(42)
    coords = rng.randn(20, 3)
    shifted = coords + np.array([5, 10, -3])
    superposed, rmsd_val = kabsch_superpose(shifted, coords)
    np.testing.assert_allclose(rmsd_val, 0.0, atol=1e-8)


def test_kabsch_rotation():
    """Pure rotation should give RMSD=0 after superposition."""
    rng = np.random.RandomState(42)
    coords = rng.randn(15, 3)
    # 90-degree rotation around z-axis
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    rotated = coords @ R.T
    _, rmsd_val = kabsch_superpose(rotated, coords)
    np.testing.assert_allclose(rmsd_val, 0.0, atol=1e-6)


def test_tm_score_identical():
    coords = np.random.RandomState(42).randn(30, 3) * 10
    score = tm_score(coords, coords)
    np.testing.assert_allclose(score, 1.0, atol=1e-6)


def test_tm_score_range():
    """TM-score should be between 0 and 1."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(20, 3) * 10
    c2 = rng.randn(20, 3) * 10
    score = tm_score(c1, c2)
    assert 0.0 <= score <= 1.0


def test_superposed_rmsd_less_than_raw():
    """Superposed RMSD should be <= raw RMSD."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(15, 3) * 5
    c2 = c1 + rng.randn(15, 3) * 2 + np.array([10, 0, 0])
    raw = rmsd(c1, c2)
    sup = superposed_rmsd(c1, c2)
    assert sup <= raw + 1e-10
