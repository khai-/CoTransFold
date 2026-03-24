"""Structure quality metrics for validation.

Provides a unified interface for comparing predicted structures
against experimental references.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cotransfold.structure.rmsd import superposed_rmsd, tm_score, kabsch_superpose
from cotransfold.structure.coordinates import get_ca_coords, torsion_to_cartesian
from cotransfold.structure.secondary import (
    assign_secondary_structure, helix_fraction, strand_fraction, ss_summary,
)
from cotransfold.core.conformation import BackboneState


@dataclass
class ValidationResult:
    """Results from comparing predicted structure to reference."""
    ca_rmsd: float          # CA RMSD after superposition (Å)
    tm_score: float         # TM-score (0-1)
    predicted_ss: str       # Predicted secondary structure string
    reference_ss: str       # Reference secondary structure string (if available)
    helix_fraction_pred: float
    strand_fraction_pred: float
    n_residues: int
    gdt_ts: float           # GDT-TS score (0-1)

    def summary(self) -> str:
        return (
            f"Validation Results\n"
            f"{'=' * 40}\n"
            f"Residues:     {self.n_residues}\n"
            f"CA RMSD:      {self.ca_rmsd:.2f} Å\n"
            f"TM-score:     {self.tm_score:.3f}\n"
            f"GDT-TS:       {self.gdt_ts:.3f}\n"
            f"Pred SS:      {self.predicted_ss}\n"
            f"Helix:        {self.helix_fraction_pred*100:.1f}%\n"
            f"Strand:       {self.strand_fraction_pred*100:.1f}%\n"
        )


def gdt_ts(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Global Distance Test - Total Score.

    GDT-TS = (GDT_1 + GDT_2 + GDT_4 + GDT_8) / 4

    where GDT_d = fraction of CA atoms within d Å after optimal superposition.

    Args:
        coords1: predicted CA coordinates, shape (N, 3)
        coords2: reference CA coordinates, shape (N, 3)

    Returns:
        GDT-TS score (0-1)
    """
    superposed, _ = kabsch_superpose(coords1, coords2)
    distances = np.linalg.norm(superposed - coords2, axis=1)
    n = len(distances)

    gdt_scores = []
    for cutoff in [1.0, 2.0, 4.0, 8.0]:
        frac = np.sum(distances < cutoff) / n
        gdt_scores.append(frac)

    return float(np.mean(gdt_scores))


def validate_structure(predicted_backbone: BackboneState,
                       reference_ca: np.ndarray,
                       reference_ss: str = '') -> ValidationResult:
    """Compare predicted backbone to reference CA coordinates.

    Args:
        predicted_backbone: predicted BackboneState
        reference_ca: reference CA coordinates, shape (N, 3)
        reference_ss: optional reference secondary structure string

    Returns:
        ValidationResult with all metrics
    """
    pred_coords = torsion_to_cartesian(predicted_backbone)
    pred_ca = get_ca_coords(pred_coords)

    n = min(len(pred_ca), len(reference_ca))
    pred_ca = pred_ca[:n]
    ref_ca = reference_ca[:n]

    ca_rmsd_val = superposed_rmsd(pred_ca, ref_ca)
    tm_val = tm_score(pred_ca, ref_ca)
    gdt_val = gdt_ts(pred_ca, ref_ca)

    pred_ss = assign_secondary_structure(predicted_backbone)

    return ValidationResult(
        ca_rmsd=ca_rmsd_val,
        tm_score=tm_val,
        predicted_ss=pred_ss,
        reference_ss=reference_ss,
        helix_fraction_pred=helix_fraction(pred_ss),
        strand_fraction_pred=strand_fraction(pred_ss),
        n_residues=n,
        gdt_ts=gdt_val,
    )
