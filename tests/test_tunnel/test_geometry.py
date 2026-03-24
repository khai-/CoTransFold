"""Tests for tunnel geometry."""

import numpy as np
from cotransfold.tunnel.geometry import TunnelGeometry, TunnelParams


def test_default_tunnel_length():
    tunnel = TunnelGeometry()
    assert tunnel.length == 90.0


def test_radius_at_ptc():
    tunnel = TunnelGeometry()
    assert tunnel.radius_at(0.0) == 5.0


def test_radius_at_constriction():
    """Constriction should be the narrowest point."""
    tunnel = TunnelGeometry()
    r_constriction = tunnel.radius_at(40.0)  # Default constriction position
    r_upper = tunnel.radius_at(15.0)
    r_vestibule = tunnel.radius_at(70.0)

    assert r_constriction < r_upper
    assert r_constriction < r_vestibule
    assert r_constriction == 4.0  # Default min radius


def test_radius_widens_toward_exit():
    tunnel = TunnelGeometry()
    r_vest_entry = tunnel.radius_at(50.0)
    r_vest_exit = tunnel.radius_at(90.0)
    assert r_vest_exit > r_vest_entry


def test_outside_tunnel_infinite_radius():
    tunnel = TunnelGeometry()
    r = tunnel.radius_at(100.0)
    assert r == float('inf')


def test_is_inside():
    tunnel = TunnelGeometry()
    assert tunnel.is_inside(45.0, radial_offset=2.0)   # Inside
    assert not tunnel.is_inside(40.0, radial_offset=5.0)  # Outside (wall at constriction r=4)
    assert not tunnel.is_inside(-5.0)   # Before tunnel
    assert not tunnel.is_inside(95.0)   # After tunnel


def test_wall_distance():
    tunnel = TunnelGeometry()
    # On axis at constriction: wall_dist = 4.0 - 0.0 = 4.0
    assert tunnel.wall_distance(40.0, 0.0) == 4.0
    # At wall: wall_dist = 0
    assert tunnel.wall_distance(40.0, 4.0) == 0.0
    # Past wall: negative
    assert tunnel.wall_distance(40.0, 5.0) < 0


def test_get_zone():
    tunnel = TunnelGeometry()
    assert tunnel.get_zone(10.0) == "upper"
    assert tunnel.get_zone(40.0) == "constriction"
    assert tunnel.get_zone(70.0) == "vestibule"
    assert tunnel.get_zone(100.0) == "outside"


def test_radius_profile():
    tunnel = TunnelGeometry()
    z, r = tunnel.get_radius_profile(50)
    assert len(z) == 50
    assert len(r) == 50
    assert z[0] == 0.0
    assert z[-1] == 90.0
    assert all(ri > 0 for ri in r)


def test_custom_params():
    params = TunnelParams(
        total_length=85.0,
        constriction_min_radius=4.5,
        vestibule_exit_radius=8.5,
    )
    tunnel = TunnelGeometry(params)
    assert tunnel.length == 85.0
    assert tunnel.radius_at(40.0) == 4.5
