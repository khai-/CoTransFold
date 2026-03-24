"""Trigger Factor (TF) — bacterial ribosome-associated chaperone.

Trigger Factor docks at the tunnel exit (uL23) and interacts with
the nascent chain as it emerges. It accelerates folding by enhancing
polypeptide collapse (Till et al., PNAS 2025).

Two modes:
1. Scanning (~50ms): transient binding, probes chain hydrophobicity
2. Locking (~1s): stable binding when chain has hydrophobic patches

Effect: enhances compaction of hydrophobic segments, prevents
premature aggregation while allowing productive folding.
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.chain import NascentChain
from cotransfold.core.residue import RESIDUE_PROPERTIES
from cotransfold.chaperones.base import ChaperoneEffect, ChaperoneAction


class TriggerFactor(ChaperoneEffect):
    """Trigger Factor chaperone (bacterial).

    Engages when:
    1. Chain has >= min_exposed exposed residues (exited tunnel)
    2. Exposed segment has significant hydrophobicity
    """

    def __init__(self, min_exposed: int = 5,
                 hydrophobic_threshold: float = 0.2) -> None:
        self._min_exposed = min_exposed
        self._hydrophobic_threshold = hydrophobic_threshold

    @property
    def name(self) -> str:
        return "TriggerFactor"

    def should_engage(self, chain: NascentChain, elapsed_time: float) -> bool:
        if chain.num_exposed < self._min_exposed:
            return False
        # Check hydrophobicity of exposed segment
        avg_hydro = self._exposed_hydrophobicity(chain)
        return avg_hydro > self._hydrophobic_threshold

    def compute_effect(self, chain: NascentChain) -> ChaperoneAction:
        avg_hydro = self._exposed_hydrophobicity(chain)

        # Stronger effect for more hydrophobic segments
        compaction = 1.0 + 0.3 * min(avg_hydro, 1.0)
        energy_mod = -1.0 * min(avg_hydro, 1.0)

        # Hold the most recently exposed residues slightly
        # to prevent premature long-range contacts
        holdase = np.zeros(chain.chain_length, dtype=bool)
        n_hold = min(3, chain.num_exposed)
        # Hold the last few exposed residues (closest to tunnel exit)
        exposed_indices = np.where(chain.is_exposed)[0]
        if len(exposed_indices) >= n_hold:
            holdase[exposed_indices[-n_hold:]] = True

        return ChaperoneAction(
            chaperone_name=self.name,
            energy_modifier=energy_mod,
            compaction_scale=compaction,
            holdase_mask=holdase,
            description=f"TF: compaction={compaction:.2f}, hydro={avg_hydro:.2f}",
        )

    def _exposed_hydrophobicity(self, chain: NascentChain) -> float:
        """Average hydrophobicity of exposed residues."""
        if chain.num_exposed == 0:
            return 0.0
        total = 0.0
        count = 0
        for i, exposed in enumerate(chain.is_exposed):
            if exposed:
                props = RESIDUE_PROPERTIES[chain.sequence[i]]
                total += props.hydrophobicity
                count += 1
        return total / count if count > 0 else 0.0
