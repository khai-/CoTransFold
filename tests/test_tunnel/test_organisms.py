"""Tests for organism-specific tunnel configurations."""

from cotransfold.tunnel.organisms import get_tunnel, ecoli_tunnel, yeast_tunnel, human_tunnel


def test_ecoli_tunnel():
    geom, elec = ecoli_tunnel()
    assert geom.length == 90.0
    assert geom.radius_at(40.0) == 4.0


def test_yeast_tunnel():
    geom, elec = yeast_tunnel()
    assert geom.length == 85.0
    assert geom.radius_at(40.0) == 4.5


def test_human_tunnel():
    geom, elec = human_tunnel()
    assert geom.length == 85.0


def test_ecoli_larger_than_eukaryotic():
    """E. coli tunnel should be larger in volume than eukaryotic."""
    ecoli_geom, _ = ecoli_tunnel()
    yeast_geom, _ = yeast_tunnel()

    # Compare vestibule radius (biggest contributor to volume difference)
    assert ecoli_geom.radius_at(70.0) > yeast_geom.radius_at(70.0)


def test_get_tunnel():
    geom, elec = get_tunnel('ecoli')
    assert geom.length == 90.0


def test_get_tunnel_invalid():
    try:
        get_tunnel('unknown')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
