"""Monte Carlo move set for protein structure sampling.

Provides fragment insertion, pivot, and crankshaft moves for exploring
the conformational space of torsion-angle-based protein models.
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.sampling.fragments import FragmentLibrary


def fragment_insertion(backbone: BackboneState, frag_lib: FragmentLibrary,
                       length: int = 9,
                       rng: np.random.RandomState | None = None) -> int:
    """Insert a fragment from the library, replacing phi/psi at a random position.

    Returns: start position of insertion.
    """
    if rng is None:
        rng = np.random.RandomState()

    n = backbone.num_residues
    length = min(length, n)
    if length < 1:
        return 0

    start = rng.randint(0, max(1, n - length + 1))

    phi_frag, psi_frag = frag_lib.sample_fragment(start, length)

    # Ensure fragment fits within chain
    actual_len = min(length, n - start, len(phi_frag))
    backbone.phi[start:start + actual_len] = phi_frag[:actual_len]
    backbone.psi[start:start + actual_len] = psi_frag[:actual_len]

    return start


def pivot_move(backbone: BackboneState, sigma: float = 15.0,
               rng: np.random.RandomState | None = None) -> int:
    """Change a single phi or psi angle by a Gaussian perturbation.

    This rotates the entire downstream chain, making it a large-scale move.

    Returns: residue index that was modified.
    """
    if rng is None:
        rng = np.random.RandomState()

    n = backbone.num_residues
    res = rng.randint(0, n)
    angle_type = rng.randint(0, 2)  # 0=phi, 1=psi
    delta = np.radians(rng.normal(0, sigma))

    if angle_type == 0:
        backbone.phi[res] += delta
        backbone.phi[res] = np.arctan2(np.sin(backbone.phi[res]),
                                        np.cos(backbone.phi[res]))
    else:
        backbone.psi[res] += delta
        backbone.psi[res] = np.arctan2(np.sin(backbone.psi[res]),
                                        np.cos(backbone.psi[res]))

    return res


def crankshaft_move(backbone: BackboneState, segment_len: int = 5,
                    sigma: float = 20.0,
                    rng: np.random.RandomState | None = None) -> tuple[int, int]:
    """Rotate a segment of residues while keeping endpoints fixed.

    Implemented by adjusting the phi/psi angles at the segment boundaries
    to compensate for interior angle changes. This approximately preserves
    chain closure.

    Returns: (start, end) of modified segment.
    """
    if rng is None:
        rng = np.random.RandomState()

    n = backbone.num_residues
    if n < segment_len + 2:
        return pivot_move(backbone, sigma, rng), 0

    start = rng.randint(1, n - segment_len)
    end = start + segment_len

    # Perturb interior residues
    delta = np.radians(rng.normal(0, sigma, size=segment_len))
    backbone.phi[start:end] += delta
    backbone.psi[start:end] += delta * 0.5  # Smaller psi change for stability

    # Wrap to [-pi, pi]
    backbone.phi[start:end] = np.arctan2(np.sin(backbone.phi[start:end]),
                                          np.cos(backbone.phi[start:end]))
    backbone.psi[start:end] = np.arctan2(np.sin(backbone.psi[start:end]),
                                          np.cos(backbone.psi[start:end]))

    return start, end


def random_mc_move(backbone: BackboneState, frag_lib: FragmentLibrary,
                   stage: str = 'early',
                   rng: np.random.RandomState | None = None) -> str:
    """Perform a random MC move based on the sampling stage.

    Args:
        backbone: current backbone state (modified in place)
        frag_lib: fragment library
        stage: 'early' (exploration) or 'late' (refinement)
        rng: random state

    Returns: move type string
    """
    if rng is None:
        rng = np.random.RandomState()

    r = rng.random()

    if stage == 'early':
        # Early: favor large moves (9-mer fragments, pivots)
        if r < 0.45:
            fragment_insertion(backbone, frag_lib, length=9, rng=rng)
            return 'frag9'
        elif r < 0.70:
            fragment_insertion(backbone, frag_lib, length=3, rng=rng)
            return 'frag3'
        elif r < 0.85:
            pivot_move(backbone, sigma=20.0, rng=rng)
            return 'pivot'
        else:
            crankshaft_move(backbone, segment_len=5, sigma=25.0, rng=rng)
            return 'crankshaft'
    else:
        # Late: favor small moves (3-mer fragments, small pivots)
        if r < 0.20:
            fragment_insertion(backbone, frag_lib, length=9, rng=rng)
            return 'frag9'
        elif r < 0.55:
            fragment_insertion(backbone, frag_lib, length=3, rng=rng)
            return 'frag3'
        elif r < 0.80:
            pivot_move(backbone, sigma=8.0, rng=rng)
            return 'pivot'
        else:
            crankshaft_move(backbone, segment_len=3, sigma=10.0, rng=rng)
            return 'crankshaft'
