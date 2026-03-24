"""Hsp70 / DnaK / Ssb — post-exit holdase chaperone.

Hsp70 family chaperones bind exposed hydrophobic patches on the
nascent chain after it exits the tunnel. They prevent aggregation
and premature misfolding by holding hydrophobic segments in a
soluble, unfolded state until the full domain has emerged.

Hsp70 engagement correlates with the Native Fold Delay (NFD) metric
(Nature Comms, 2025): when N-terminal interaction partners are not
yet synthesized, Hsp70 holds the chain to prevent misfolding.
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.chain import NascentChain
from cotransfold.core.residue import RESIDUE_PROPERTIES
from cotransfold.chaperones.base import ChaperoneEffect, ChaperoneAction


class Hsp70(ChaperoneEffect):
    """Hsp70 family holdase chaperone.

    Engages when:
    1. Chain has significant exposed hydrophobic surface
    2. Exposed segment is long enough (a partial domain is out)
    """

    def __init__(self, min_exposed: int = 15,
                 hydrophobic_threshold: float = 0.15) -> None:
        self._min_exposed = min_exposed
        self._threshold = hydrophobic_threshold

    @property
    def name(self) -> str:
        return "Hsp70"

    def should_engage(self, chain: NascentChain, elapsed_time: float) -> bool:
        if chain.num_exposed < self._min_exposed:
            return False
        return self._exposed_hydrophobic_fraction(chain) > self._threshold

    def compute_effect(self, chain: NascentChain) -> ChaperoneAction:
        # Hsp70 holds hydrophobic residues to prevent aggregation
        holdase = np.zeros(chain.chain_length, dtype=bool)

        for i, exposed in enumerate(chain.is_exposed):
            if exposed:
                hydro = RESIDUE_PROPERTIES[chain.sequence[i]].hydrophobicity
                if hydro > 0.3:  # Only hold strongly hydrophobic residues
                    holdase[i] = True

        n_held = int(holdase.sum())
        return ChaperoneAction(
            chaperone_name=self.name,
            energy_modifier=-0.3 * n_held,  # Stabilization per held residue
            compaction_scale=0.9,  # Slightly reduce compaction (keep unfolded)
            holdase_mask=holdase,
            description=f"Hsp70: holding {n_held} hydrophobic residues",
        )

    def _exposed_hydrophobic_fraction(self, chain: NascentChain) -> float:
        """Fraction of exposed residues that are hydrophobic."""
        if chain.num_exposed == 0:
            return 0.0
        n_hydro = 0
        n_total = 0
        for i, exposed in enumerate(chain.is_exposed):
            if exposed:
                n_total += 1
                if RESIDUE_PROPERTIES[chain.sequence[i]].hydrophobicity > 0.2:
                    n_hydro += 1
        return n_hydro / n_total if n_total > 0 else 0.0
