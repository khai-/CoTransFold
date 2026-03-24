"""Organism-specific ribosome exit tunnel configurations.

Parameters derived from cryo-EM structures and the literature:
- Dao Duc et al. (bioRxiv 2025) — tunnel geometry across tree of life
- Joiret et al. (Phys Rev E 2022) — tunnel electrostatics
- McGrath & Kolar (bioRxiv 2026) — constriction dynamics
"""

from __future__ import annotations

from cotransfold.tunnel.geometry import TunnelGeometry, TunnelParams
from cotransfold.tunnel.electrostatics import TunnelElectrostatics, ElectrostaticParams


def ecoli_tunnel() -> tuple[TunnelGeometry, TunnelElectrostatics]:
    """E. coli ribosome exit tunnel.

    Bacterial: larger volume (~3.85e4 Å³), extended uL23 loop.
    Total length ~90Å, wider constriction than eukaryotes.
    """
    geom = TunnelGeometry(TunnelParams(
        upper_end=30.0,
        constriction_end=50.0,
        total_length=90.0,
        upper_radius=5.0,
        constriction_min_radius=4.0,
        constriction_position=40.0,
        vestibule_entry_radius=7.5,
        vestibule_exit_radius=10.0,
    ))
    elec = TunnelElectrostatics(ElectrostaticParams(
        upper_potential=-2.0,
        constriction_potential=3.0,
        vestibule_potential=-1.0,
        exit_potential=-4.0,
        debye_length=10.0,
        upper_end=30.0,
        constriction_end=50.0,
        total_length=90.0,
    ))
    return geom, elec


def yeast_tunnel() -> tuple[TunnelGeometry, TunnelElectrostatics]:
    """S. cerevisiae ribosome exit tunnel.

    Eukaryotic: smaller volume (~2.78e4 Å³), eL39 present.
    Narrower vestibule than bacteria.
    """
    geom = TunnelGeometry(TunnelParams(
        upper_end=30.0,
        constriction_end=50.0,
        total_length=85.0,
        upper_radius=5.0,
        constriction_min_radius=4.5,
        constriction_position=40.0,
        vestibule_entry_radius=6.5,
        vestibule_exit_radius=8.5,
    ))
    elec = TunnelElectrostatics(ElectrostaticParams(
        upper_potential=-2.5,
        constriction_potential=2.5,
        vestibule_potential=-1.5,
        exit_potential=-5.0,
        debye_length=10.0,
        upper_end=30.0,
        constriction_end=50.0,
        total_length=85.0,
    ))
    return geom, elec


def human_tunnel() -> tuple[TunnelGeometry, TunnelElectrostatics]:
    """Human ribosome exit tunnel.

    Similar to yeast but with specific differences in vestibule
    architecture affecting domain docking timing (Pellowe et al. 2025).
    Slightly stronger surface charge.
    """
    geom = TunnelGeometry(TunnelParams(
        upper_end=30.0,
        constriction_end=50.0,
        total_length=85.0,
        upper_radius=5.0,
        constriction_min_radius=4.5,
        constriction_position=40.0,
        vestibule_entry_radius=6.5,
        vestibule_exit_radius=9.0,
    ))
    elec = TunnelElectrostatics(ElectrostaticParams(
        upper_potential=-2.5,
        constriction_potential=2.5,
        vestibule_potential=-1.5,
        exit_potential=-6.0,  # Stronger surface charge
        debye_length=10.0,
        upper_end=30.0,
        constriction_end=50.0,
        total_length=85.0,
    ))
    return geom, elec


ORGANISM_TUNNELS = {
    'ecoli': ecoli_tunnel,
    'yeast': yeast_tunnel,
    'human': human_tunnel,
}


def get_tunnel(organism: str = 'ecoli') -> tuple[TunnelGeometry, TunnelElectrostatics]:
    """Get tunnel configuration for an organism.

    Args:
        organism: one of 'ecoli', 'yeast', 'human'

    Returns:
        (TunnelGeometry, TunnelElectrostatics) tuple
    """
    if organism not in ORGANISM_TUNNELS:
        raise ValueError(
            f"Unknown organism '{organism}'. Choose from: {list(ORGANISM_TUNNELS.keys())}")
    return ORGANISM_TUNNELS[organism]()
