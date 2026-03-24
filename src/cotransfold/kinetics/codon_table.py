"""Organism-specific codon usage tables and translation rates.

Codon usage frequency determines translation speed at each position.
Rare codons (low frequency) are translated slowly, creating pauses
that give the nascent chain more time to fold.

Data sources:
- Kazusa Codon Usage Database
- Sharp & Li (1987) codon adaptation index
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodonInfo:
    """Information about a single codon."""
    codon: str
    amino_acid: str       # One-letter code
    frequency: float      # Relative synonymous codon usage (0-1)
    rate: float           # Relative translation rate (codons/sec, normalized)


# E. coli K-12 codon usage (major codons have higher frequency)
# Frequencies are relative within each amino acid group (sum to ~1.0)
# Rates approximate relative tRNA abundance
_ECOLI_CODONS = {
    'UUU': CodonInfo('UUU', 'F', 0.58, 12.0),
    'UUC': CodonInfo('UUC', 'F', 0.42, 18.0),
    'UUA': CodonInfo('UUA', 'L', 0.13, 6.0),
    'UUG': CodonInfo('UUG', 'L', 0.13, 6.0),
    'CUU': CodonInfo('CUU', 'L', 0.10, 5.0),
    'CUC': CodonInfo('CUC', 'L', 0.10, 5.0),
    'CUA': CodonInfo('CUA', 'L', 0.04, 2.0),
    'CUG': CodonInfo('CUG', 'L', 0.50, 20.0),
    'AUU': CodonInfo('AUU', 'I', 0.51, 15.0),
    'AUC': CodonInfo('AUC', 'I', 0.42, 18.0),
    'AUA': CodonInfo('AUA', 'I', 0.07, 3.0),
    'AUG': CodonInfo('AUG', 'M', 1.00, 15.0),
    'GUU': CodonInfo('GUU', 'V', 0.26, 12.0),
    'GUC': CodonInfo('GUC', 'V', 0.22, 10.0),
    'GUA': CodonInfo('GUA', 'V', 0.15, 7.0),
    'GUG': CodonInfo('GUG', 'V', 0.37, 15.0),
    'UCU': CodonInfo('UCU', 'S', 0.15, 8.0),
    'UCC': CodonInfo('UCC', 'S', 0.15, 8.0),
    'UCA': CodonInfo('UCA', 'S', 0.12, 5.0),
    'UCG': CodonInfo('UCG', 'S', 0.15, 8.0),
    'AGU': CodonInfo('AGU', 'S', 0.15, 7.0),
    'AGC': CodonInfo('AGC', 'S', 0.28, 12.0),
    'CCU': CodonInfo('CCU', 'P', 0.16, 6.0),
    'CCC': CodonInfo('CCC', 'P', 0.12, 5.0),
    'CCA': CodonInfo('CCA', 'P', 0.19, 7.0),
    'CCG': CodonInfo('CCG', 'P', 0.53, 18.0),
    'ACU': CodonInfo('ACU', 'T', 0.17, 8.0),
    'ACC': CodonInfo('ACC', 'T', 0.44, 18.0),
    'ACA': CodonInfo('ACA', 'T', 0.13, 5.0),
    'ACG': CodonInfo('ACG', 'T', 0.27, 12.0),
    'GCU': CodonInfo('GCU', 'A', 0.18, 10.0),
    'GCC': CodonInfo('GCC', 'A', 0.26, 14.0),
    'GCA': CodonInfo('GCA', 'A', 0.20, 10.0),
    'GCG': CodonInfo('GCG', 'A', 0.36, 18.0),
    'UAU': CodonInfo('UAU', 'Y', 0.57, 10.0),
    'UAC': CodonInfo('UAC', 'Y', 0.43, 15.0),
    'CAU': CodonInfo('CAU', 'H', 0.57, 10.0),
    'CAC': CodonInfo('CAC', 'H', 0.43, 15.0),
    'CAA': CodonInfo('CAA', 'Q', 0.34, 10.0),
    'CAG': CodonInfo('CAG', 'Q', 0.66, 18.0),
    'AAU': CodonInfo('AAU', 'N', 0.45, 10.0),
    'AAC': CodonInfo('AAC', 'N', 0.55, 15.0),
    'AAA': CodonInfo('AAA', 'K', 0.76, 18.0),
    'AAG': CodonInfo('AAG', 'K', 0.24, 8.0),
    'GAU': CodonInfo('GAU', 'D', 0.63, 14.0),
    'GAC': CodonInfo('GAC', 'D', 0.37, 12.0),
    'GAA': CodonInfo('GAA', 'E', 0.69, 18.0),
    'GAG': CodonInfo('GAG', 'E', 0.31, 10.0),
    'UGU': CodonInfo('UGU', 'C', 0.45, 8.0),
    'UGC': CodonInfo('UGC', 'C', 0.55, 12.0),
    'UGG': CodonInfo('UGG', 'W', 1.00, 12.0),
    'CGU': CodonInfo('CGU', 'R', 0.38, 15.0),
    'CGC': CodonInfo('CGC', 'R', 0.40, 16.0),
    'CGA': CodonInfo('CGA', 'R', 0.06, 2.0),
    'CGG': CodonInfo('CGG', 'R', 0.10, 4.0),
    'AGA': CodonInfo('AGA', 'R', 0.04, 1.5),
    'AGG': CodonInfo('AGG', 'R', 0.02, 1.0),
    'GGU': CodonInfo('GGU', 'G', 0.34, 15.0),
    'GGC': CodonInfo('GGC', 'G', 0.40, 18.0),
    'GGA': CodonInfo('GGA', 'G', 0.11, 5.0),
    'GGG': CodonInfo('GGG', 'G', 0.15, 6.0),
}

# Most common codon for each amino acid (E. coli)
_ECOLI_PREFERRED = {
    'A': 'GCG', 'R': 'CGC', 'N': 'AAC', 'D': 'GAU', 'C': 'UGC',
    'E': 'GAA', 'Q': 'CAG', 'G': 'GGC', 'H': 'CAU', 'I': 'AUU',
    'L': 'CUG', 'K': 'AAA', 'M': 'AUG', 'F': 'UUU', 'P': 'CCG',
    'S': 'AGC', 'T': 'ACC', 'W': 'UGG', 'Y': 'UAU', 'V': 'GUG',
}


class CodonUsageTable:
    """Organism-specific codon usage frequencies and translation rates."""

    def __init__(self, codons: dict[str, CodonInfo],
                 preferred: dict[str, str],
                 average_rate: float = 15.0,
                 pause_threshold: float = 0.10) -> None:
        """
        Args:
            codons: codon -> CodonInfo mapping
            preferred: amino_acid_1letter -> preferred codon
            average_rate: average translation rate (aa/sec)
            pause_threshold: codons with frequency < this are "rare"
        """
        self._codons = codons
        self._preferred = preferred
        self.average_rate = average_rate
        self.pause_threshold = pause_threshold

    @classmethod
    def ecoli(cls) -> CodonUsageTable:
        """E. coli K-12 codon usage table."""
        return cls(_ECOLI_CODONS, _ECOLI_PREFERRED,
                   average_rate=15.0, pause_threshold=0.10)

    def get_info(self, codon: str) -> CodonInfo:
        """Get info for a codon. Raises KeyError if unknown."""
        codon = codon.upper().replace('T', 'U')
        return self._codons[codon]

    def translation_time(self, codon: str) -> float:
        """Time to translate this codon (seconds).

        Inverse of translation rate: slower codons take longer.
        """
        info = self.get_info(codon)
        return 1.0 / max(info.rate, 0.1)

    def is_rare(self, codon: str) -> bool:
        """Is this a rare codon that creates a translational pause?"""
        info = self.get_info(codon)
        return info.frequency < self.pause_threshold

    def preferred_codon(self, aa_one_letter: str) -> str:
        """Get the most commonly used codon for an amino acid."""
        return self._preferred[aa_one_letter.upper()]

    def all_codons_for(self, aa_one_letter: str) -> list[CodonInfo]:
        """Get all codons encoding a given amino acid."""
        return [info for info in self._codons.values()
                if info.amino_acid == aa_one_letter.upper()]
