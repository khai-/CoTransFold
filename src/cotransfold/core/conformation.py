"""Backbone conformation represented as torsion angles."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class BackboneState:
    """Backbone conformation as torsion angles.

    phi[i], psi[i], omega[i] are the torsion angles for residue i.
    Arrays of length N (number of residues currently in the chain).

    Conventions:
    - phi[0] is undefined for the N-terminal residue (set to 0)
    - psi[N-1] is undefined for the C-terminal residue (set to 0)
    - omega is typically ~pi (trans); ~0 for cis-proline
    - All angles in radians
    """
    phi: np.ndarray    # shape (N,)
    psi: np.ndarray    # shape (N,)
    omega: np.ndarray  # shape (N,)

    @property
    def num_residues(self) -> int:
        return len(self.phi)

    @classmethod
    def empty(cls) -> BackboneState:
        """Create an empty backbone state (no residues)."""
        return cls(
            phi=np.array([], dtype=np.float64),
            psi=np.array([], dtype=np.float64),
            omega=np.array([], dtype=np.float64),
        )

    @classmethod
    def extended(cls, n: int) -> BackboneState:
        """Create an extended conformation (beta-strand-like) for n residues.

        phi = -120°, psi = 120°, omega = 180° (trans)
        """
        return cls(
            phi=np.full(n, np.radians(-120.0)),
            psi=np.full(n, np.radians(120.0)),
            omega=np.full(n, np.pi),
        )

    @classmethod
    def alpha_helix(cls, n: int) -> BackboneState:
        """Create an ideal alpha-helix for n residues.

        phi = -57°, psi = -47°, omega = 180° (trans)
        """
        return cls(
            phi=np.full(n, np.radians(-57.0)),
            psi=np.full(n, np.radians(-47.0)),
            omega=np.full(n, np.pi),
        )

    def extend(self, phi: float = np.radians(-120.0),
               psi: float = np.radians(120.0),
               omega: float = np.pi) -> BackboneState:
        """Return new BackboneState with one additional residue appended."""
        return BackboneState(
            phi=np.append(self.phi, phi),
            psi=np.append(self.psi, psi),
            omega=np.append(self.omega, omega),
        )

    def copy(self) -> BackboneState:
        """Return a deep copy."""
        return BackboneState(
            phi=self.phi.copy(),
            psi=self.psi.copy(),
            omega=self.omega.copy(),
        )

    def set_angles(self, index: int, phi: float | None = None,
                   psi: float | None = None, omega: float | None = None) -> None:
        """Set torsion angles for a specific residue (in-place)."""
        if phi is not None:
            self.phi[index] = phi
        if psi is not None:
            self.psi[index] = psi
        if omega is not None:
            self.omega[index] = omega

    def get_torsion_vector(self) -> np.ndarray:
        """Flatten phi/psi angles into a single vector for optimization.

        Returns array of shape (2*N,): [phi_0, psi_0, phi_1, psi_1, ...]
        Omega is excluded (held fixed at ~pi during minimization).
        """
        return np.column_stack([self.phi, self.psi]).ravel()

    def set_from_torsion_vector(self, vec: np.ndarray) -> None:
        """Set phi/psi from a flattened vector (inverse of get_torsion_vector)."""
        n = self.num_residues
        pairs = vec.reshape(n, 2)
        self.phi[:] = pairs[:, 0]
        self.psi[:] = pairs[:, 1]
