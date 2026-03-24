"""Tests for TotalEnergy compositor."""

import numpy as np
from cotransfold.core.residue import AminoAcid, sequence_from_string
from cotransfold.core.chain import NascentChain
from cotransfold.energy.total import TotalEnergy
from cotransfold.energy.ramachandran import RamachandranEnergy
from cotransfold.energy.hbond import HydrogenBondEnergy
from cotransfold.energy.vanderwaals import VanDerWaalsEnergy
from cotransfold.energy.bonded import BondedEnergy


def _make_chain(seq_str: str, helix: bool = True) -> NascentChain:
    """Helper to create a chain with given sequence."""
    chain = NascentChain.empty()
    phi = np.radians(-57.0) if helix else np.radians(-120.0)
    psi = np.radians(-47.0) if helix else np.radians(120.0)
    for aa in sequence_from_string(seq_str):
        chain.add_residue(aa, phi=phi, psi=psi)
    return chain


def test_empty_chain():
    energy = TotalEnergy()
    energy.add_term(RamachandranEnergy())
    chain = NascentChain.empty()
    assert energy.compute(chain) == 0.0


def test_single_term():
    energy = TotalEnergy()
    energy.add_term(RamachandranEnergy(), weight=1.0)
    chain = _make_chain("AAAAAAAAAA")
    e = energy.compute(chain)
    assert isinstance(e, float)
    assert np.isfinite(e)


def test_weighted_terms():
    """Doubling weight should double contribution."""
    chain = _make_chain("AAAAAAAAAA")

    energy1 = TotalEnergy()
    energy1.add_term(RamachandranEnergy(), weight=1.0)
    e1 = energy1.compute(chain)

    energy2 = TotalEnergy()
    energy2.add_term(RamachandranEnergy(), weight=2.0)
    e2 = energy2.compute(chain)

    np.testing.assert_allclose(e2, 2 * e1, rtol=1e-10)


def test_decomposed():
    energy = TotalEnergy()
    energy.add_term(RamachandranEnergy(), weight=1.0)
    energy.add_term(HydrogenBondEnergy(), weight=1.0)
    energy.add_term(BondedEnergy(), weight=1.0)

    chain = _make_chain("AAAAAAAAAAAAAAA")
    decomposed = energy.compute_decomposed(chain)

    assert "ramachandran" in decomposed
    assert "hbond" in decomposed
    assert "bonded" in decomposed

    # Total should equal sum of parts
    total = energy.compute(chain)
    np.testing.assert_allclose(total, sum(decomposed.values()), rtol=1e-10)


def test_all_terms_together():
    """All energy terms should work together without errors."""
    energy = TotalEnergy()
    energy.add_term(RamachandranEnergy(), weight=1.0)
    energy.add_term(HydrogenBondEnergy(), weight=1.0)
    energy.add_term(VanDerWaalsEnergy(), weight=1.0)
    energy.add_term(BondedEnergy(), weight=0.5)

    chain = _make_chain("AAAAAGAAAAPAAAA")
    e = energy.compute(chain)
    assert isinstance(e, float)
    assert np.isfinite(e)


def test_helix_lower_energy_than_random():
    """Helix should have lower total energy than random conformation."""
    energy = TotalEnergy()
    energy.add_term(RamachandranEnergy(), weight=1.0)
    energy.add_term(HydrogenBondEnergy(), weight=1.0)
    energy.add_term(VanDerWaalsEnergy(), weight=1.0)

    helix_chain = _make_chain("AAAAAAAAAAAAAAA", helix=True)
    e_helix = energy.compute(helix_chain)

    # Random chain
    rng = np.random.RandomState(42)
    random_chain = NascentChain.empty()
    for aa in sequence_from_string("AAAAAAAAAAAAAAA"):
        random_chain.add_residue(
            aa,
            phi=rng.uniform(-np.pi, np.pi),
            psi=rng.uniform(-np.pi, np.pi),
        )
    e_random = energy.compute(random_chain)

    assert e_helix < e_random, (
        f"Helix ({e_helix:.2f}) should have lower energy than random ({e_random:.2f})")
