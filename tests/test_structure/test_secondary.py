"""Tests for secondary structure assignment."""

import numpy as np
from cotransfold.core.conformation import BackboneState
from cotransfold.structure.secondary import (
    assign_secondary_structure, helix_fraction, strand_fraction,
    longest_helix, ss_summary,
)


def test_helix_assignment():
    backbone = BackboneState.alpha_helix(10)
    ss = assign_secondary_structure(backbone)
    assert all(c == 'H' for c in ss), f"Expected all H, got {ss}"


def test_strand_assignment():
    backbone = BackboneState.extended(10)
    ss = assign_secondary_structure(backbone)
    assert all(c == 'E' for c in ss), f"Expected all E, got {ss}"


def test_mixed_structure():
    n = 10
    phi = np.zeros(n)
    psi = np.zeros(n)
    # First 5 helix, last 5 strand
    for i in range(5):
        phi[i] = np.radians(-57)
        psi[i] = np.radians(-47)
    for i in range(5, 10):
        phi[i] = np.radians(-120)
        psi[i] = np.radians(130)
    backbone = BackboneState(phi=phi, psi=psi, omega=np.full(n, np.pi))
    ss = assign_secondary_structure(backbone)
    assert ss[:5] == 'HHHHH'
    assert ss[5:] == 'EEEEE'


def test_helix_fraction():
    assert helix_fraction("HHHHH") == 1.0
    assert helix_fraction("HHCCC") == 0.4
    assert helix_fraction("EEEEE") == 0.0


def test_strand_fraction():
    assert strand_fraction("EEEEE") == 1.0
    assert strand_fraction("HHHHH") == 0.0


def test_longest_helix():
    assert longest_helix("HHHCCHHHHH") == 5
    assert longest_helix("CCCCCC") == 0
    assert longest_helix("H") == 1


def test_ss_summary():
    s = ss_summary("HHHHCCCEEE")
    assert "Helix" in s
    assert "Strand" in s
