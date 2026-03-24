"""Tests for translation schedule."""

import numpy as np
from cotransfold.core.residue import AminoAcid
from cotransfold.kinetics.translation import TranslationSchedule
from cotransfold.kinetics.codon_table import CodonUsageTable


def test_uniform_schedule():
    sched = TranslationSchedule.uniform(10)
    assert len(sched.base_times) == 10
    assert len(sched.cumulative_times) == 10
    np.testing.assert_allclose(sched.pause_times, 0.0)
    assert sched.total_translation_time > 0


def test_from_amino_acids():
    seq = [AminoAcid.ALA] * 10
    sched = TranslationSchedule.from_amino_acids(seq, 'ecoli')
    assert len(sched.base_times) == 10
    assert sched.total_translation_time > 0


def test_rare_codons_create_pauses():
    """Rare codons should have non-zero pause times."""
    table = CodonUsageTable.ecoli()
    # Mix of common and rare codons
    codons = ['CUG', 'CUG', 'AGG', 'CUG', 'AGA']  # AGG, AGA are rare
    sched = TranslationSchedule.from_codons(codons, table)

    # Positions 2 and 4 should have pauses
    assert sched.is_rare[2]
    assert sched.is_rare[4]
    assert not sched.is_rare[0]
    assert sched.pause_times[2] > 0
    assert sched.pause_times[0] == 0.0


def test_rare_codons_more_folding_time():
    """Positions with rare codons should have more total folding time."""
    table = CodonUsageTable.ecoli()
    codons = ['CUG', 'AGG']  # Common, then rare
    sched = TranslationSchedule.from_codons(codons, table)

    assert sched.total_times[1] > sched.total_times[0], (
        f"Rare codon should have more time: {sched.total_times[1]:.4f} vs {sched.total_times[0]:.4f}")


def test_cumulative_time_monotonic():
    seq = [AminoAcid.ALA, AminoAcid.GLY, AminoAcid.VAL, AminoAcid.LEU, AminoAcid.PRO]
    sched = TranslationSchedule.from_amino_acids(seq, 'ecoli')

    for i in range(1, len(sched.cumulative_times)):
        assert sched.cumulative_times[i] > sched.cumulative_times[i - 1]
