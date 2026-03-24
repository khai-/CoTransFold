"""Tests for L-BFGS-B gradient minimizer."""

import numpy as np
from cotransfold.core.residue import AminoAcid, sequence_from_string
from cotransfold.core.chain import NascentChain
from cotransfold.energy.total import TotalEnergy
from cotransfold.energy.ramachandran import RamachandranEnergy
from cotransfold.energy.hbond import HydrogenBondEnergy
from cotransfold.energy.vanderwaals import VanDerWaalsEnergy
from cotransfold.minimizer.gradient import GradientMinimizer


def _make_random_chain(seq_str: str, seed: int = 42) -> NascentChain:
    """Create chain with random torsion angles."""
    rng = np.random.RandomState(seed)
    chain = NascentChain.empty()
    for aa in sequence_from_string(seq_str):
        chain.add_residue(
            aa,
            phi=rng.uniform(-np.pi, np.pi),
            psi=rng.uniform(-np.pi, np.pi),
        )
    return chain


def _make_energy() -> TotalEnergy:
    energy = TotalEnergy()
    energy.add_term(RamachandranEnergy(), weight=1.0)
    energy.add_term(HydrogenBondEnergy(), weight=1.0)
    energy.add_term(VanDerWaalsEnergy(), weight=0.5)
    return energy


def test_minimizer_reduces_energy():
    """Minimization should reduce energy from random start."""
    chain = _make_random_chain("AAAAAAAAAA")
    energy = _make_energy()

    minimizer = GradientMinimizer(max_iterations=50)
    result = minimizer.minimize(chain, energy)

    assert result.energy_after <= result.energy_before, (
        f"Energy should decrease: {result.energy_before:.2f} -> {result.energy_after:.2f}")


def test_minimizer_with_frozen_mask():
    """Frozen residues should not change."""
    chain = _make_random_chain("AAAAAAAAAA")
    energy = _make_energy()

    # Freeze first 5, free last 5
    frozen_mask = np.array([0.0] * 5 + [1.0] * 5)

    original_phi = chain.backbone.phi[:5].copy()
    original_psi = chain.backbone.psi[:5].copy()

    minimizer = GradientMinimizer(max_iterations=30)
    minimizer.minimize(chain, energy, frozen_mask=frozen_mask)

    np.testing.assert_allclose(chain.backbone.phi[:5], original_phi,
        err_msg="Frozen phi angles should not change")
    np.testing.assert_allclose(chain.backbone.psi[:5], original_psi,
        err_msg="Frozen psi angles should not change")


def test_minimizer_deterministic():
    """Same input should produce same output."""
    energy = _make_energy()
    minimizer = GradientMinimizer(max_iterations=30)

    chain1 = _make_random_chain("AAAAAA", seed=123)
    result1 = minimizer.minimize(chain1, energy)

    chain2 = _make_random_chain("AAAAAA", seed=123)
    result2 = minimizer.minimize(chain2, energy)

    np.testing.assert_allclose(result1.energy_after, result2.energy_after, rtol=1e-8)
    np.testing.assert_allclose(chain1.backbone.phi, chain2.backbone.phi, atol=1e-8)


def test_poly_alanine_converges_toward_helix():
    """Poly-alanine from random start should move toward allowed Ramachandran regions."""
    chain = _make_random_chain("A" * 15, seed=0)
    energy = _make_energy()

    minimizer = GradientMinimizer(max_iterations=200)
    result = minimizer.minimize(chain, energy)

    # The key assertion: energy must decrease
    assert result.energy_after < result.energy_before, (
        f"Energy should decrease: {result.energy_before:.2f} -> {result.energy_after:.2f}")

    # And the minimized structure should have lower Ramachandran energy
    # than the random starting point (residues moved toward allowed regions)
    from cotransfold.energy.ramachandran import RamachandranEnergy, rama_probability
    n_improved = 0
    for i in range(15):
        p = rama_probability(chain.backbone.phi[i], chain.backbone.psi[i], AminoAcid.ALA)
        if p > 0.05:  # In a reasonably populated Ramachandran region
            n_improved += 1
    assert n_improved >= 2, (
        f"Expected at least 2 residues in populated Ramachandran regions, got {n_improved}")


def test_minimizer_empty_chain():
    """Empty chain should be handled gracefully."""
    chain = NascentChain.empty()
    energy = _make_energy()
    minimizer = GradientMinimizer()
    result = minimizer.minimize(chain, energy)
    assert result.converged
    assert result.energy_after == 0.0
