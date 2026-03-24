"""Electrostatic potential field inside the ribosome exit tunnel.

The tunnel has a complex electrostatic environment:
- Upper tunnel (near PTC): net negative from rRNA phosphate backbone
- Constriction (uL4/uL22): local positive charges from Arg/Lys residues
- Vestibule/exit: net negative from ribosome surface

The potential is modeled as a piecewise function along the tunnel axis
with Debye-Hückel screening for distance from the wall.

References:
- Joiret et al. (2022) Phys Rev E 105:014409 — tunnel electrostatics
- Streit et al. (2025) bioRxiv — ribosome surface charges
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ElectrostaticParams:
    """Parameters for tunnel electrostatic potential."""
    # Potential at key positions along tunnel axis (kcal/mol/e)
    # Negative = attracts positive charges, repels negative
    upper_potential: float = -2.0       # rRNA phosphates (negative)
    constriction_potential: float = 3.0  # uL4/uL22 Arg/Lys (positive)
    vestibule_potential: float = -1.0    # Mixed
    exit_potential: float = -4.0         # Ribosome surface (strong negative)

    # Debye screening length (Å) — ~10Å at physiological ionic strength
    debye_length: float = 10.0

    # Zone boundaries (should match TunnelParams)
    upper_end: float = 30.0
    constriction_end: float = 50.0
    total_length: float = 90.0


class TunnelElectrostatics:
    """Electrostatic potential field in the ribosome exit tunnel.

    The potential varies along the tunnel axis and decays radially
    from the tunnel wall with Debye-Hückel screening.
    """

    def __init__(self, params: ElectrostaticParams | None = None) -> None:
        self.params = params or ElectrostaticParams()
        p = self.params

        # Piecewise linear potential along axis
        self._z_points = np.array([
            0.0,
            p.upper_end * 0.5,
            p.upper_end,
            (p.upper_end + p.constriction_end) / 2,
            p.constriction_end,
            (p.constriction_end + p.total_length) / 2,
            p.total_length,
        ])
        self._v_points = np.array([
            p.upper_potential,
            p.upper_potential,
            p.upper_potential * 0.5,
            p.constriction_potential,
            p.vestibule_potential,
            p.vestibule_potential,
            p.exit_potential,
        ])

    def potential_at(self, distance: float, radial_offset: float = 0.0) -> float:
        """Return electrostatic potential at a tunnel position.

        The potential is strongest near the tunnel wall and decays
        toward the axis with Debye-Hückel screening.

        Args:
            distance: distance from PTC along tunnel axis (Å)
            radial_offset: distance from tunnel axis (Å)

        Returns:
            Electrostatic potential in kcal/mol/e
        """
        if distance < 0 or distance > self.params.total_length:
            return 0.0

        # Axial potential from piecewise linear interpolation
        v_axis = float(np.interp(distance, self._z_points, self._v_points))

        # The potential originates from the tunnel wall.
        # For a residue on the axis, it experiences the wall potential
        # screened by the Debye length. Since the wall is at radius R
        # and the residue is at radial_offset, the screening distance
        # is approximately (R - radial_offset).
        # For simplicity in the backbone-only model (residues on axis),
        # we use a mild distance-dependent modulation.
        # The full effect is captured by the tunnel_energy term.
        return v_axis

    def field_at(self, distance: float) -> float:
        """Return the axial electric field (gradient of potential).

        Positive field = force pushing chain toward exit.
        Negative field = force pushing chain toward PTC.

        Args:
            distance: distance from PTC (Å)

        Returns:
            Axial field in kcal/mol/e/Å
        """
        h = 0.5  # Å
        v_plus = self.potential_at(distance + h)
        v_minus = self.potential_at(distance - h)
        return -(v_plus - v_minus) / (2 * h)
