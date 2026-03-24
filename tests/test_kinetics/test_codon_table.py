"""Tests for codon usage table."""

from cotransfold.kinetics.codon_table import CodonUsageTable


def test_ecoli_table():
    table = CodonUsageTable.ecoli()
    assert table.average_rate == 15.0


def test_get_info():
    table = CodonUsageTable.ecoli()
    info = table.get_info('AUG')
    assert info.amino_acid == 'M'
    assert info.frequency == 1.0


def test_translation_time():
    table = CodonUsageTable.ecoli()
    t = table.translation_time('AUG')
    assert t > 0
    assert isinstance(t, float)


def test_rare_codon_detection():
    """AGG (Arg) should be rare in E. coli (frequency 0.02)."""
    table = CodonUsageTable.ecoli()
    assert table.is_rare('AGG')
    # CUG (Leu) should NOT be rare (frequency 0.50)
    assert not table.is_rare('CUG')


def test_rare_codons_slower():
    """Rare codons should take longer to translate."""
    table = CodonUsageTable.ecoli()
    t_rare = table.translation_time('AGG')    # Rare Arg codon
    t_common = table.translation_time('CGC')  # Common Arg codon
    assert t_rare > t_common


def test_preferred_codon():
    table = CodonUsageTable.ecoli()
    assert table.preferred_codon('M') == 'AUG'
    assert table.preferred_codon('L') == 'CUG'


def test_all_codons_for():
    table = CodonUsageTable.ecoli()
    leu_codons = table.all_codons_for('L')
    assert len(leu_codons) == 6  # Leucine has 6 codons


def test_dna_codon_converted():
    """DNA codons (T instead of U) should be auto-converted."""
    table = CodonUsageTable.ecoli()
    info = table.get_info('ATG')  # DNA form
    assert info.amino_acid == 'M'
