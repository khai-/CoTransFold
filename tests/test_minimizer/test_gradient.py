"""Tests for L-BFGS-B gradient minimizer."""

import numpy as np
from cotransfold.core.residue import AminoAcid, sequence_from_string
from cotransfold.core.chain import NascentChain
from cotransfold.energy.total import TotalEnergy
from cotransfold.energy.ramachandran import RamachandranEnergy
from cotransfold.energy.hbond import HydrogenBondEnergy
from cotransfold.minimizer.gradient import GradientMinimizer


def _make_random_chain(seq_str: str, seed: int = 42) -> NascentChain:
    rng = np.random.RandomState(seed)
    chain = NascentChain.empty()
    for aa in sequence_from_string(seq_str):
        chain.add_residue(aa, phi=rng.uniform(-np.pi, np.pi), psi=rng.uniform(-np.pi, np.pi))
    return chain


def _make_energy() -> TotalEnergy:
    energy = TotalEnergy()
    energy.add_term(RamachandranEnergy(), weight=1.0)
    energy.add_term(HydrogenBondEnergy(), weight=1.0)
    return energy


def test_minimizer_reduces_energy():
    chain = _make_random_chain("AAAAAAAAAA")
    result = GradientMinimizer(max_iterations=30).minimize(chain, _make_energy())
    assert result.energy_after <= result.energy_before


def test_frozen_mask():
    chain = _make_random_chain("AAAAAAAAAA")
    frozen_mask = np.array([0.0] * 5 + [1.0] * 5)
    original_phi = chain.backbone.phi[:5].copy()
    GradientMinimizer(max_iterations=20).minimize(chain, _make_energy(), frozen_mask=frozen_mask)
    np.testing.assert_allclose(chain.backbone.phi[:5], original_phi)


def test_empty_chain():
    chain = NascentChain.empty()
    result = GradientMinimizer().minimize(chain, _make_energy())
    assert result.converged and result.energy_after == 0.0
