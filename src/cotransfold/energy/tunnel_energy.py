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
        """Compute tunnel constraint energy.

        Requires 'tunnel_positions' in kwargs — array of distances from PTC
        for each residue. This is provided by the NascentChain.
        """
        tunnel_positions = kwargs.get('tunnel_positions')
        if tunnel_positions is None:
            return 0.0

        n = len(coords)
        if n == 0:
            return 0.0

        ca = get_ca_coords(coords)  # shape (N, 3)
        tunnel_length = self._geom.length

        energy = 0.0

        for i in range(n):
            dist_from_ptc = tunnel_positions[i]

            # Skip residues outside the tunnel
            if dist_from_ptc < 0 or dist_from_ptc > tunnel_length:
                continue

            # --- Wall repulsion ---
            # In the backbone-only model with a straight tunnel axis (z-axis),
            # the radial offset of a residue is approximated by how far its
            # CA atom deviates from where it would be if perfectly on-axis.
            #
            # For a chain threading through the tunnel, we approximate the
            # radial offset as the distance of the CA from the tunnel axis.
            # The tunnel axis passes through the centroid of in-tunnel CAs.
            #
            # Simplified model: use the CA-CA distance to adjacent residues
            # to estimate how compact the chain is in this region. If the
            # backbone is helical (compact), it has larger radial extent
            # than if extended.
            #
            # For the MVP, we estimate radial offset from the backbone
            # geometry relative to the tunnel axis direction.
            tunnel_radius = self._geom.radius_at(dist_from_ptc)
            effective_radius = self._estimate_radial_offset(
                ca, i, dist_from_ptc, tunnel_length)

            wall_dist = tunnel_radius - effective_radius - self._wall_buffer
            if wall_dist < 0:
                energy += self._wall_spring * wall_dist ** 2

            # --- Electrostatic interaction ---
            if i < len(sequence):
                charge = RESIDUE_PROPERTIES[sequence[i]].charge
                if charge != 0.0:
                    potential = self._elec.potential_at(dist_from_ptc)
                    energy += self._elec_weight * charge * potential

        return energy

    def _estimate_radial_offset(self, ca: np.ndarray, res_idx: int,
                                dist_from_ptc: float,
                                tunnel_length: float) -> float:
        """Estimate how far a CA atom is from the tunnel axis.

        Uses the local chain geometry: if the backbone is helical,
        the CA traces a helix of radius ~2.3Å around the axis.
        If extended, it's nearly on-axis (~0Å radial offset).

        For the MVP, we use a simplified estimate based on the
        local curvature of the CA chain.
        """
        n = len(ca)
        if n < 3 or res_idx == 0 or res_idx == n - 1:
            return 0.0

        # Local chain direction vectors
        v1 = ca[res_idx] - ca[res_idx - 1]
        v2 = ca[res_idx + 1] - ca[res_idx]

        # Angle between consecutive segments
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Higher curvature (smaller angle) = larger radial offset
        # Alpha helix: ~91° between consecutive CA-CA vectors
        # Extended: ~150-180°
        angle = np.arccos(cos_angle)

        # Approximate radial offset from curvature
        # For a helix with CA-CA ~3.8Å and angle ~91°: radius ~2.3Å
        # For extended with angle ~160°: radius ~0.5Å
        ca_ca_dist = np.linalg.norm(v1)
        if angle < 0.01:
            return 0.0

        # radius = (ca_ca_dist / 2) / sin(angle/2)... but simplified:
        # Map angle to approximate radial offset
        # 90° -> ~2.3Å, 120° -> ~1.5Å, 150° -> ~0.7Å, 180° -> 0Å
        radial_offset = 2.3 * (1.0 - cos_angle) / 2.0

        # Add side chain radius contribution (approximate)
        if res_idx < len(ca):
            # Larger side chains take more space
            # For backbone-only, just use a fixed estimate
            radial_offset += 1.0  # ~1Å for backbone + minimal sidechain

        return max(radial_offset, 0.0)
