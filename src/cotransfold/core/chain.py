"""NascentChain - the growing polypeptide during co-translational folding."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from cotransfold.core.residue import AminoAcid
from cotransfold.core.conformation import BackboneState
from cotransfold.config import RESIDUE_ADVANCE_DISTANCE


@dataclass
class NascentChain:
    """The growing polypeptide chain during translation.

    Tracks the amino acid sequence, backbone conformation, and each
    residue's position relative to the ribosome exit tunnel.

    tunnel_position[i] = distance of residue i from the PTC (Å).
    The most recently added residue is at position 0 (at the PTC).
    As translation proceeds, earlier residues advance through the tunnel.
    Residues with tunnel_position > tunnel_length are "exposed" (outside the tunnel).
    """
    sequence: list[AminoAcid] = field(default_factory=list)
    backbone: BackboneState = field(default_factory=BackboneState.empty)
    tunnel_position: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64))
    is_exposed: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool))

    @classmethod
    def empty(cls) -> NascentChain:
        """Create an empty nascent chain."""
        return cls()

    @property
    def chain_length(self) -> int:
        return len(self.sequence)

    @property
    def num_exposed(self) -> int:
        return int(self.is_exposed.sum()) if len(self.is_exposed) > 0 else 0

    def add_residue(self, aa: AminoAcid,
                    phi: float = np.radians(-120.0),
                    psi: float = np.radians(120.0),
                    omega: float = np.pi) -> None:
        """Add a new residue at the C-terminus (PTC end of tunnel).

        1. Advance all existing residues by one step through the tunnel
        2. Append new residue at tunnel position 0 (PTC)
        3. Extend backbone with initial torsion angles
        """
        # Advance existing residues
        if len(self.tunnel_position) > 0:
            self.tunnel_position += RESIDUE_ADVANCE_DISTANCE

        # Add new residue at PTC
        self.sequence.append(aa)
        self.backbone = self.backbone.extend(phi, psi, omega)
        self.tunnel_position = np.append(self.tunnel_position, 0.0)
        self.is_exposed = np.append(self.is_exposed, False)

    def update_exposure(self, tunnel_length: float) -> None:
        """Update which residues are exposed (outside the tunnel).

        Args:
            tunnel_length: total tunnel length from PTC to exit (Å)
        """
        self.is_exposed = self.tunnel_position > tunnel_length

    def get_frozen_mask(self, upper_end: float = 30.0,
                        constriction_end: float = 50.0) -> np.ndarray:
        """Classify residues by mobility.

        Returns array of shape (N,) with values:
        0.0 = frozen (in upper tunnel, no room to move)
        0.5 = restricted (in constriction zone, limited phi/psi)
        1.0 = free (in vestibule or exposed)
        """
        mask = np.ones(self.chain_length, dtype=np.float64)
        for i in range(self.chain_length):
            d = self.tunnel_position[i]
            if d < upper_end:
                mask[i] = 0.0
            elif d < constriction_end:
                mask[i] = 0.5
        return mask

    def copy(self) -> NascentChain:
        """Return a deep copy."""
        return NascentChain(
            sequence=list(self.sequence),
            backbone=self.backbone.copy(),
            tunnel_position=self.tunnel_position.copy(),
            is_exposed=self.is_exposed.copy(),
        )
