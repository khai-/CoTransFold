"""Nascent polypeptide-Associated Complex (NAC).

NAC is the first chaperone to interact with the nascent chain.
It senses nascent chains INSIDE the tunnel when they are <30 amino acids
long (Lee et al., Nature 2025).

NAC effect: mild compaction enhancement, preparing the chain for
proper folding as it exits the tunnel. Prevents premature
misfolding of the earliest residues.
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.chain import NascentChain
from cotransfold.chaperones.base import ChaperoneEffect, ChaperoneAction


class NAC(ChaperoneEffect):
    """Nascent polypeptide-Associated Complex.

    Engages when chain length is between 10-30 residues.
    Senses the chain inside the tunnel and prepares it for exit.
    """

    def __init__(self, min_length: int = 10, max_length: int = 30) -> None:
        self._min_length = min_length
        self._max_length = max_length

    @property
    def name(self) -> str:
        return "NAC"

    def should_engage(self, chain: NascentChain, elapsed_time: float) -> bool:
        return self._min_length <= chain.chain_length <= self._max_length

    def compute_effect(self, chain: NascentChain) -> ChaperoneAction:
        return ChaperoneAction(
            chaperone_name=self.name,
            energy_modifier=-0.5,  # Mild stabilization
            compaction_scale=1.1,  # Slight compaction enhancement
            description="NAC tunnel sensing: mild stabilization",
        )
