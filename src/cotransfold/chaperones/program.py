"""Chaperone program — orchestrates which chaperones are active at each step.

Different organisms have different chaperone repertoires:
- Bacteria: NAC, Trigger Factor, DnaK(Hsp70), SRP, GroEL/ES
- Eukaryotes: NAC, RAC/Ssb(Hsp70), SRP, TRiC/CCT

The program determines which chaperones engage based on:
1. Organism (determines available chaperones)
2. Chain length and exposure state
3. Sequence features (hydrophobicity, signal sequences)

References:
- Proteome-wide chaperone program (Nature Comms, 2025)
- NAC tunnel sensing (Lee et al., Nature 2025)
- Trigger factor compaction (Till et al., PNAS 2025)
"""

from __future__ import annotations

from cotransfold.core.chain import NascentChain
from cotransfold.chaperones.base import ChaperoneEffect, ChaperoneAction
from cotransfold.chaperones.nac import NAC
from cotransfold.chaperones.trigger_factor import TriggerFactor
from cotransfold.chaperones.srp import SRP
from cotransfold.chaperones.hsp70 import Hsp70


# Organism chaperone repertoires
_BACTERIAL_CHAPERONES = [NAC, TriggerFactor, SRP, Hsp70]
_EUKARYOTIC_CHAPERONES = [NAC, SRP, Hsp70]  # TF is bacterial-specific


class ChaperoneProgram:
    """Orchestrates the combinatorial chaperone program.

    Queries all available chaperones at each step and returns
    the list of active effects.
    """

    def __init__(self, organism: str = 'ecoli') -> None:
        self._organism = organism
        self._chaperones: list[ChaperoneEffect] = []
        self._setup(organism)

    def _setup(self, organism: str) -> None:
        """Initialize chaperone repertoire for the organism."""
        if organism == 'ecoli':
            self._chaperones = [cls() for cls in _BACTERIAL_CHAPERONES]
        elif organism in ('yeast', 'human'):
            self._chaperones = [cls() for cls in _EUKARYOTIC_CHAPERONES]
        else:
            raise ValueError(f"Unknown organism: {organism}")

    @property
    def available_chaperones(self) -> list[str]:
        """Names of all available chaperones for this organism."""
        return [c.name for c in self._chaperones]

    def active_chaperones(self, chain: NascentChain,
                          elapsed_time: float) -> list[ChaperoneAction]:
        """Get list of active chaperone actions at current state.

        Args:
            chain: current nascent chain
            elapsed_time: total elapsed translation time (seconds)

        Returns:
            List of ChaperoneAction from all engaged chaperones
        """
        actions = []
        for chaperone in self._chaperones:
            if chaperone.should_engage(chain, elapsed_time):
                action = chaperone.compute_effect(chain)
                actions.append(action)
        return actions

    def get_total_energy_modifier(self, actions: list[ChaperoneAction]) -> float:
        """Sum energy modifiers from all active chaperones."""
        return sum(a.energy_modifier for a in actions)

    def get_combined_compaction_scale(self, actions: list[ChaperoneAction]) -> float:
        """Product of compaction scales from all active chaperones."""
        scale = 1.0
        for a in actions:
            scale *= a.compaction_scale
        return scale
