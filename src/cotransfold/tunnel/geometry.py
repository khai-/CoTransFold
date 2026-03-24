"""Ribosome exit tunnel geometry model.

The tunnel is modeled as a three-zone cylinder along the z-axis:

Zone 1 (Upper tunnel): PTC (z=0) to constriction start
  - Narrow, prevents all but extended/nascent conformations
  - Lined by 23S rRNA

Zone 2 (Constriction): formed by uL4/uL22 protein loops
  - Narrowest point of the tunnel (~4Å radius in bacteria)
  - Dynamic gate (McGrath & Kolar, bioRxiv 2026)

Zone 3 (Vestibule): widens toward the exit
  - Alpha helices can form here (~20Å wide)
  - Exit site where chaperones dock (uL23)

The tunnel axis is modeled as a straight line (z-axis) for the MVP.
Future: curved spline from cryo-EM data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TunnelParams:
    """Parameters defining a three-zone tunnel."""
    # Zone boundaries (distance from PTC in Å)
    upper_end: float = 30.0
    constriction_end: float = 50.0
    total_length: float = 90.0

    # Radii at key points (Å)
    upper_radius: float = 5.0
    constriction_min_radius: float = 4.0
    constriction_position: float = 40.0   # Distance of narrowest point from PTC
    vestibule_entry_radius: float = 7.5
    vestibule_exit_radius: float = 10.0


class TunnelGeometry:
    """Three-zone cylindrical tunnel model.

    Coordinates: the tunnel axis runs along z from z=0 (PTC) to z=total_length (exit).
    Residues are positioned by their distance from PTC (= z coordinate).
    The tunnel has circular cross-sections with radius varying along z.
    """

    def __init__(self, params: TunnelParams | None = None) -> None:
        self.params = params or TunnelParams()
        p = self.params

        # Build piecewise linear radius profile as (z, r) control points
        self._z_points = np.array([
            0.0,
            p.upper_end,
            p.constriction_position,
            p.constriction_end,
            p.total_length,
        ])
        self._r_points = np.array([
            p.upper_radius,
            p.upper_radius,
            p.constriction_min_radius,
            p.vestibule_entry_radius,
            p.vestibule_exit_radius,
        ])

    @property
    def length(self) -> float:
        """Total tunnel length from PTC to exit (Å)."""
        return self.params.total_length

    def radius_at(self, distance: float) -> float:
        """Return tunnel radius at given distance from PTC (Å).

        Args:
            distance: distance from PTC along tunnel axis

        Returns:
            Tunnel radius in Å. Returns 0 for positions beyond the tunnel.
        """
        if distance < 0:
            return self.params.upper_radius
        if distance > self.params.total_length:
            return float('inf')  # Outside tunnel = no constraint
        return float(np.interp(distance, self._z_points, self._r_points))

    def is_inside(self, distance: float, radial_offset: float = 0.0) -> bool:
        """Test if a point is inside the tunnel.

        Args:
            distance: distance from PTC along tunnel axis
            radial_offset: distance from tunnel axis (Å)

        Returns:
            True if the point is within the tunnel walls
        """
        if distance < 0 or distance > self.params.total_length:
            return False
        return radial_offset < self.radius_at(distance)

    def wall_distance(self, distance: float, radial_offset: float) -> float:
        """Signed distance to nearest tunnel wall.

        Positive = inside tunnel (distance to wall).
        Negative = outside tunnel (penetrating wall).

        Args:
            distance: distance from PTC along tunnel axis
            radial_offset: distance from tunnel axis (Å)

        Returns:
            Signed distance to wall in Å
        """
        if distance < 0 or distance > self.params.total_length:
            return float('inf')  # Outside tunnel entirely
        r = self.radius_at(distance)
        return r - radial_offset

    def get_zone(self, distance: float) -> str:
        """Identify which tunnel zone a position is in.

        Returns: "upper", "constriction", "vestibule", or "outside"
        """
        if distance < 0:
            return "upper"
        if distance <= self.params.upper_end:
            return "upper"
        if distance <= self.params.constriction_end:
            return "constriction"
        if distance <= self.params.total_length:
            return "vestibule"
        return "outside"

    def get_radius_profile(self, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Return (z, radius) arrays for plotting the tunnel profile.

        Args:
            n_points: number of sample points

        Returns:
            (z_array, radius_array) each of shape (n_points,)
        """
        z = np.linspace(0, self.params.total_length, n_points)
        r = np.array([self.radius_at(zi) for zi in z])
        return z, r
