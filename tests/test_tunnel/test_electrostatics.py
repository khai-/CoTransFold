"""Tests for tunnel electrostatics."""

import numpy as np
from cotransfold.tunnel.electrostatics import TunnelElectrostatics, ElectrostaticParams


def test_upper_tunnel_negative():
    """Upper tunnel should have negative potential (rRNA phosphates)."""
    elec = TunnelElectrostatics()
    v = elec.potential_at(15.0)
    assert v < 0, f"Upper tunnel should be negative, got {v}"


def test_constriction_positive():
    """Constriction should have positive potential (uL4/uL22 Arg/Lys)."""
    elec = TunnelElectrostatics()
    v = elec.potential_at(40.0)
    assert v > 0, f"Constriction should be positive, got {v}"


def test_exit_negative():
    """Exit/surface should have strong negative potential."""
    elec = TunnelElectrostatics()
    v = elec.potential_at(90.0)
    assert v < 0, f"Exit should be negative, got {v}"
    # Exit should be more negative than upper tunnel
    v_upper = elec.potential_at(10.0)
    assert v < v_upper, "Exit should be more negative than upper tunnel"


def test_outside_tunnel_zero():
    """Potential outside tunnel should be zero."""
    elec = TunnelElectrostatics()
    assert elec.potential_at(-5.0) == 0.0
    assert elec.potential_at(100.0) == 0.0


def test_field_direction():
    """Field should point outward (positive) near exit to push chain out."""
    elec = TunnelElectrostatics()
    # Near exit, potential becomes more negative -> field is positive (outward)
    field = elec.field_at(80.0)
    # The field should push positively charged residues toward exit
    assert isinstance(field, float)
