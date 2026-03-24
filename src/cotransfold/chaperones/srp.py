"""Signal Recognition Particle (SRP).

SRP recognizes N-terminal signal sequences (typically hydrophobic
stretches of 7-15 residues) as they emerge from the tunnel exit.
It targets the ribosome-nascent-chain complex to the membrane
for co-translational membrane protein insertion.

In this simulator, SRP detection flags a protein as a potential
membrane protein and modifies the folding environment accordingly.
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.chain import NascentChain
from cotransfold.core.residue import RESIDUE_PROPERTIES
from cotransfold.chaperones.base import ChaperoneEffect, ChaperoneAction


class SRP(ChaperoneEffect):
    """Signal Recognition Particle.

    Detects signal sequences in the first ~30 exposed residues.
    A signal sequence is a stretch of >= 7 consecutive hydrophobic residues.
    """

    def __init__(self, min_hydrophobic_stretch: int = 7,
                 hydrophobic_threshold: float = 0.3) -> None:
        self._min_stretch = min_hydrophobic_stretch
        self._threshold = hydrophobic_threshold

    @property
    def name(self) -> str:
        return "SRP"

    def should_engage(self, chain: NascentChain, elapsed_time: float) -> bool:
        if chain.num_exposed < self._min_stretch:
            return False
        return self._has_signal_sequence(chain)

    def compute_effect(self, chain: NascentChain) -> ChaperoneAction:
        # SRP binding pauses translation and targets to membrane
        # In the simulator: holds exposed hydrophobic segment
        holdase = np.zeros(chain.chain_length, dtype=bool)
        exposed_indices = np.where(chain.is_exposed)[0]
        for idx in exposed_indices[:30]:  # Only check first 30 exposed
            if RESIDUE_PROPERTIES[chain.sequence[idx]].hydrophobicity > self._threshold:
                holdase[idx] = True

        return ChaperoneAction(
            chaperone_name=self.name,
            energy_modifier=-2.0,  # Stabilization from SRP binding
            compaction_scale=0.8,  # Reduce compaction (keep extended for membrane insertion)
            holdase_mask=holdase,
            description="SRP: signal sequence detected, membrane targeting",
        )

    def _has_signal_sequence(self, chain: NascentChain) -> bool:
        """Check for hydrophobic stretch in first exposed residues."""
        exposed_indices = np.where(chain.is_exposed)[0]
        if len(exposed_indices) == 0:
            return False

        # Check first 30 exposed residues for hydrophobic stretch
        stretch = 0
        for idx in exposed_indices[:30]:
            hydro = RESIDUE_PROPERTIES[chain.sequence[idx]].hydrophobicity
            if hydro > self._threshold:
                stretch += 1
                if stretch >= self._min_stretch:
                    return True
            else:
                stretch = 0
        return False
