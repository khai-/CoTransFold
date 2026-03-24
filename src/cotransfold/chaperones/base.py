"""Abstract base class for chaperone effects on nascent chain folding.

Chaperones modify the energy landscape of the nascent chain:
- Energy modifiers: additive bias to total energy
- Restraints: spring-like forces on specific torsion angles
- Holdase mask: prevent specific residues from folding prematurely
- Compaction modifier: scale attractive forces to enhance/prevent collapse
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from cotransfold.core.chain import NascentChain


@dataclass
class ChaperoneAction:
    """The effect of a chaperone on the nascent chain at one simulation step."""
    chaperone_name: str = ""
    energy_modifier: float = 0.0        # Additive energy bias (kcal/mol)
    compaction_scale: float = 1.0       # Scale factor for attractive interactions (>1 = enhance collapse)
    holdase_mask: np.ndarray | None = None   # Boolean: True = residue held unfolded
    description: str = ""


class ChaperoneEffect(ABC):
    """Interface for all chaperone models."""

    @abstractmethod
    def should_engage(self, chain: NascentChain,
                      elapsed_time: float) -> bool:
        """Determine if this chaperone engages at this step.

        Args:
            chain: current nascent chain state
            elapsed_time: total time since translation start (seconds)

        Returns:
            True if chaperone is active
        """
        ...

    @abstractmethod
    def compute_effect(self, chain: NascentChain) -> ChaperoneAction:
        """Compute the effect of this chaperone on the energy landscape.

        Args:
            chain: current nascent chain state

        Returns:
            ChaperoneAction describing modifications to the energy function
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this chaperone."""
        ...
