"""Per-residue confidence scoring for CoTransFold predictions.

Produces a physics-grounded confidence score (0-100) for each residue,
analogous to AlphaFold's pLDDT but derived from energy landscape
properties rather than learned from data.

Components:
1. Local energy: how favorable is this residue's energy contribution
2. H-bond satisfaction: are backbone H-bonds fulfilled
3. Steric quality: absence of clashes
4. Ramachandran quality: is phi/psi in an allowed region
5. Convergence: did the minimizer settle this residue

Final score = weighted combination, scaled 0-100.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.core.residue import AminoAcid
from cotransfold.structure.coordinates import torsion_to_cartesian, get_ca_coords
from cotransfold.energy.ramachandran import rama_probability
from cotransfold.energy.hbond import (
    _place_h_atoms_batch, _place_o_atoms_batch,
    Q1, Q2, COULOMB, MAX_NO_DISTANCE, MIN_SEPARATION,
)


@dataclass
class ConfidenceResult:
    """Per-residue and global confidence scores."""
    per_residue: np.ndarray      # Shape (N,), scores 0-100
    rama_scores: np.ndarray      # Ramachandran quality per residue
    hbond_scores: np.ndarray     # H-bond satisfaction per residue
    clash_scores: np.ndarray     # Steric quality per residue
    packing_scores: np.ndarray   # Local packing density per residue
    global_score: float          # Average confidence

    def summary(self) -> str:
        n = len(self.per_residue)
        high = np.sum(self.per_residue > 70)
        med = np.sum((self.per_residue > 50) & (self.per_residue <= 70))
        low = np.sum(self.per_residue <= 50)
        return (
            f"Confidence Summary\n"
            f"{'=' * 40}\n"
            f"Global score:  {self.global_score:.1f}/100\n"
            f"High (>70):    {high}/{n} residues ({high/n*100:.0f}%)\n"
            f"Medium (50-70): {med}/{n} residues ({med/n*100:.0f}%)\n"
            f"Low (<50):     {low}/{n} residues ({low/n*100:.0f}%)\n"
            f"Min/Max:       {self.per_residue.min():.1f} / {self.per_residue.max():.1f}\n"
        )


def compute_confidence(backbone: BackboneState,
                       sequence: list[AminoAcid]) -> ConfidenceResult:
    """Compute per-residue confidence scores.

    Args:
        backbone: final backbone conformation
        sequence: amino acid sequence

    Returns:
        ConfidenceResult with per-residue and global scores
    """
    n = backbone.num_residues
    coords = torsion_to_cartesian(backbone)
    ca = get_ca_coords(coords)

    rama = _score_ramachandran(backbone, sequence, n)
    hbond = _score_hbonds(coords, n)
    clash = _score_clashes(ca, n)
    packing = _score_packing(ca, n)

    # Weighted combination: Rama 30%, H-bond 30%, Clash 20%, Packing 20%
    per_residue = (0.30 * rama + 0.30 * hbond + 0.20 * clash + 0.20 * packing)

    return ConfidenceResult(
        per_residue=per_residue,
        rama_scores=rama,
        hbond_scores=hbond,
        clash_scores=clash,
        packing_scores=packing,
        global_score=float(np.mean(per_residue)),
    )


def _score_ramachandran(backbone: BackboneState, sequence: list,
                        n: int) -> np.ndarray:
    """Score 0-100 based on Ramachandran probability.

    High probability region = high score.
    """
    scores = np.zeros(n)
    for i in range(n):
        next_aa = sequence[i + 1] if i + 1 < n else None
        p = rama_probability(backbone.phi[i], backbone.psi[i],
                             sequence[i], next_aa)
        # Map probability to 0-100: p>0.3 = 100, p<0.01 = 0
        scores[i] = np.clip(p / 0.3, 0.0, 1.0) * 100
    return scores


def _score_hbonds(coords: np.ndarray, n: int) -> np.ndarray:
    """Score 0-100 based on backbone H-bond satisfaction.

    Each residue gets credit for participating in H-bonds
    (either as donor N-H or acceptor C=O).
    """
    scores = np.zeros(n)
    if n < MIN_SEPARATION + 1:
        return scores + 50  # Neutral for very short chains

    n_atoms = coords[:, 0]
    ca_atoms = coords[:, 1]
    c_atoms = coords[:, 2]

    h_atoms = _place_h_atoms_batch(n_atoms, ca_atoms)
    o_atoms = _place_o_atoms_batch(c_atoms, ca_atoms, n_atoms)

    # Pairwise O-N distances
    r_on = np.linalg.norm(
        o_atoms[:, None, :] - n_atoms[None, :, :], axis=2)

    idx = np.arange(n)
    sep = idx[None, :] - idx[:, None]
    mask = (sep >= MIN_SEPARATION) & (idx[:, None] < n - 1) & (idx[None, :] > 0)
    mask = mask & (r_on < MAX_NO_DISTANCE)

    if not np.any(mask):
        return scores + 30  # Low baseline if no H-bonds possible

    # Compute H-bond energies
    r_ch = np.linalg.norm(c_atoms[:, None, :] - h_atoms[None, :, :], axis=2)
    r_oh = np.linalg.norm(o_atoms[:, None, :] - h_atoms[None, :, :], axis=2)
    r_cn = np.linalg.norm(c_atoms[:, None, :] - n_atoms[None, :, :], axis=2)

    r_on_s = np.maximum(r_on, 0.01)
    r_ch_s = np.maximum(r_ch, 0.01)
    r_oh_s = np.maximum(r_oh, 0.01)
    r_cn_s = np.maximum(r_cn, 0.01)

    e = Q1 * Q2 * COULOMB * (
        1.0 / r_on_s + 1.0 / r_ch_s - 1.0 / r_oh_s - 1.0 / r_cn_s)
    e = np.where(mask, np.minimum(e, 0.0), 0.0)

    # Per-residue: best H-bond as acceptor (row) + best as donor (col)
    best_acceptor = np.min(e, axis=1)  # Best H-bond where residue is acceptor
    best_donor = np.min(e, axis=0)     # Best H-bond where residue is donor

    # Map energy to score: -1.0 kcal/mol or better = 100, 0 = 30
    for i in range(n):
        best_e = min(best_acceptor[i], best_donor[i])
        if best_e < -0.5:
            scores[i] = min(100, 30 + 70 * (-best_e / 1.0))
        else:
            scores[i] = 30  # No good H-bond

    return np.clip(scores, 0, 100)


def _score_clashes(ca: np.ndarray, n: int) -> np.ndarray:
    """Score 0-100 based on absence of steric clashes.

    No clashes = 100, severe clash = 0.
    """
    if n < 4:
        return np.full(n, 80.0)

    diff = ca[:, None, :] - ca[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    # Only check non-bonded pairs
    valid = sep >= 3

    scores = np.full(n, 100.0)
    for i in range(n):
        if not np.any(valid[i]):
            continue
        min_dist = np.min(dist[i, valid[i]])
        if min_dist < 2.0:
            scores[i] = 0.0  # Severe clash
        elif min_dist < 3.0:
            scores[i] = (min_dist - 2.0) / 1.0 * 80  # Mild clash
        else:
            scores[i] = 100.0  # No clash

    return scores


def _score_packing(ca: np.ndarray, n: int) -> np.ndarray:
    """Score 0-100 based on local packing density.

    Well-packed interior = high score, isolated = low score.
    A well-folded residue typically has 6-12 CA neighbors within 10Å.
    """
    if n < 5:
        return np.full(n, 50.0)

    diff = ca[:, None, :] - ca[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 100.0)  # Exclude self

    # Count neighbors within 10Å (excluding sequence neighbors ±1)
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    neighbors = np.sum((dist < 10.0) & (sep > 1), axis=1)

    # Map: 0 neighbors = 20, 6+ = 100
    scores = np.clip(20 + 80 * neighbors / 6.0, 20, 100)
    return scores
