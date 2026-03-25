"""Tunnel constraint energy term.

Applies two forces to residues inside the ribosome exit tunnel:

1. Steric wall repulsion: soft harmonic wall that penalizes atoms
   approaching or penetrating the tunnel walls.
   E_wall = k_wall * max(0, r_atom - r_tunnel + buffer)²

2. Electrostatic interaction: charge of residue × electrostatic
   potential at that tunnel position.
   E_elec = charge(residue) * potential(tunnel_position)

Only residues INSIDE the tunnel contribute. Exposed residues
(outside the tunnel) contribute zero tunnel energy.

The tunnel constrains backbone geometry:
- Upper tunnel: only extended conformations fit (narrow)
- Constriction: very restricted, acts as a gate
- Vestibule: wider, alpha-helices can form here
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.core.residue import AminoAcid, RESIDUE_PROPERTIES
from cotransfold.energy.total import EnergyTerm
from cotransfold.tunnel.geometry import TunnelGeometry
from cotransfold.tunnel.electrostatics import TunnelElectrostatics
from cotransfold.structure.coordinates import get_ca_coords


class TunnelEnergy(EnergyTerm):
    """Energy from tunnel steric and electrostatic constraints.

    For each residue inside the tunnel:
    1. Compute its radial distance from the tunnel axis
    2. Apply wall repulsion if too close to the wall
    3. Apply electrostatic interaction based on residue charge
    """

    def __init__(self,
                 geometry: TunnelGeometry,
                 electrostatics: TunnelElectrostatics,
                 wall_spring: float = 5.0,
                 wall_buffer: float = 1.5,
                 elec_weight: float = 1.0) -> None:
        """
        Args:
            geometry: tunnel geometry model
            electrostatics: tunnel electrostatic field
            wall_spring: spring constant for wall repulsion (kcal/mol/Å²)
            wall_buffer: buffer distance from wall before repulsion starts (Å)
            elec_weight: scaling factor for electrostatic contribution
        """
        self._geom = geometry
        self._elec = electrostatics
        self._wall_spring = wall_spring
        self._wall_buffer = wall_buffer
        self._elec_weight = elec_weight

    @property
    def name(self) -> str:
        return "tunnel"

    def compute(self, coords: np.ndarray, backbone: BackboneState,
                sequence: list, **kwargs) -> float:
        """Compute tunnel constraint energy (vectorized)."""
        tunnel_positions = kwargs.get('tunnel_positions')
        if tunnel_positions is None:
            return 0.0

        n = len(coords)
        if n == 0:
            return 0.0

        ca = get_ca_coords(coords)  # (N, 3)
        tunnel_length = self._geom.length

        # Mask: residues inside the tunnel
        in_tunnel = (tunnel_positions >= 0) & (tunnel_positions <= tunnel_length)
        if not np.any(in_tunnel):
            return 0.0

        energy = 0.0

        # --- Wall repulsion (vectorized) ---
        # Compute radial offsets for all in-tunnel residues at once
        radial_offsets = self._estimate_radial_offsets_batch(ca, n)

        # Tunnel radii at each position
        tunnel_radii = np.array([self._geom.radius_at(d) for d in tunnel_positions])

        wall_dist = tunnel_radii - radial_offsets - self._wall_buffer
        wall_clash = in_tunnel & (wall_dist < 0)
        if np.any(wall_clash):
            energy += float(np.sum(self._wall_spring * wall_dist[wall_clash] ** 2))

        # --- Electrostatic interaction (vectorized) ---
        charges = np.array([RESIDUE_PROPERTIES[aa].charge for aa in sequence])
        charged_in_tunnel = in_tunnel & (charges != 0.0)
        if np.any(charged_in_tunnel):
            potentials = np.array([self._elec.potential_at(d) for d in tunnel_positions[charged_in_tunnel]])
            energy += float(np.sum(self._elec_weight * charges[charged_in_tunnel] * potentials))

        return energy

    def _estimate_radial_offsets_batch(self, ca: np.ndarray, n: int) -> np.ndarray:
        """Estimate radial offsets for all residues at once.

        Uses local backbone curvature: helical = larger offset, extended = smaller.
        """
        offsets = np.zeros(n)
        if n < 3:
            return offsets

        # Vectors between consecutive CAs
        v1 = ca[1:-1] - ca[:-2]    # (N-2, 3)
        v2 = ca[2:] - ca[1:-1]     # (N-2, 3)

        # Cosine of angle between consecutive segments
        norms1 = np.linalg.norm(v1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(v2, axis=1, keepdims=True)
        norms1 = np.maximum(norms1, 1e-10)
        norms2 = np.maximum(norms2, 1e-10)

        cos_angles = np.sum((v1 / norms1) * (v2 / norms2), axis=1)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)

        # Radial offset: 2.3 * (1 - cos) / 2 + 1.0 (backbone width)
        offsets[1:-1] = np.maximum(2.3 * (1.0 - cos_angles) / 2.0 + 1.0, 0.0)

        return offsets
