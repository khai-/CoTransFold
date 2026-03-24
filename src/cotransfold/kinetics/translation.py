"""Translation schedule — folding time available at each residue position.

The translation schedule determines how much time the nascent chain has
to fold at each step of elongation. Rare codons create pauses that
allow more folding time; common codons are translated quickly.

This schedule maps directly to the number of minimization steps
in the simulation engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cotransfold.core.residue import AminoAcid
from cotransfold.kinetics.codon_table import CodonUsageTable


# One-letter code lookup for AminoAcid enum
_AA_TO_LETTER = {
    AminoAcid.ALA: 'A', AminoAcid.ARG: 'R', AminoAcid.ASN: 'N',
    AminoAcid.ASP: 'D', AminoAcid.CYS: 'C', AminoAcid.GLU: 'E',
    AminoAcid.GLN: 'Q', AminoAcid.GLY: 'G', AminoAcid.HIS: 'H',
    AminoAcid.ILE: 'I', AminoAcid.LEU: 'L', AminoAcid.LYS: 'K',
    AminoAcid.MET: 'M', AminoAcid.PHE: 'F', AminoAcid.PRO: 'P',
    AminoAcid.SER: 'S', AminoAcid.THR: 'T', AminoAcid.TRP: 'W',
    AminoAcid.TYR: 'Y', AminoAcid.VAL: 'V',
}


@dataclass
class TranslationSchedule:
    """Time available for folding at each step of translation.

    Attributes:
        base_times: time for one elongation cycle at each position (seconds)
        pause_times: additional pause from rare codons (seconds)
        cumulative_times: total elapsed time since translation start
        is_rare: boolean array indicating rare codon positions
    """
    base_times: np.ndarray
    pause_times: np.ndarray
    cumulative_times: np.ndarray
    is_rare: np.ndarray

    @property
    def total_times(self) -> np.ndarray:
        """Total folding time at each position (base + pause)."""
        return self.base_times + self.pause_times

    @property
    def total_translation_time(self) -> float:
        """Total time to translate the entire sequence (seconds)."""
        return float(self.cumulative_times[-1]) if len(self.cumulative_times) > 0 else 0.0

    @classmethod
    def from_codons(cls, codons: list[str],
                    table: CodonUsageTable) -> TranslationSchedule:
        """Compute schedule from mRNA codon sequence.

        Args:
            codons: list of codon strings (e.g., ['AUG', 'GCG', ...])
            table: organism-specific codon usage table
        """
        n = len(codons)
        base_times = np.zeros(n)
        pause_times = np.zeros(n)
        is_rare = np.zeros(n, dtype=bool)

        for i, codon in enumerate(codons):
            base_times[i] = table.translation_time(codon)
            if table.is_rare(codon):
                is_rare[i] = True
                # Rare codons pause for ~5-10x longer than normal
                pause_times[i] = base_times[i] * 5.0

        total = base_times + pause_times
        cumulative = np.cumsum(total)

        return cls(base_times, pause_times, cumulative, is_rare)

    @classmethod
    def from_amino_acids(cls, sequence: list[AminoAcid],
                         organism: str = 'ecoli') -> TranslationSchedule:
        """Estimate schedule from amino acid sequence (no codon info).

        Uses the preferred (most common) codon for each amino acid.
        This gives a baseline schedule without rare-codon pauses.
        """
        if organism == 'ecoli':
            table = CodonUsageTable.ecoli()
        else:
            raise ValueError(f"Unknown organism: {organism}")

        codons = []
        for aa in sequence:
            letter = _AA_TO_LETTER[aa]
            codons.append(table.preferred_codon(letter))

        return cls.from_codons(codons, table)

    @classmethod
    def uniform(cls, n: int, time_per_residue: float = 0.067) -> TranslationSchedule:
        """Create a uniform schedule (same time for all residues).

        Default: 0.067s per residue = 15 aa/sec (E. coli average).
        """
        base_times = np.full(n, time_per_residue)
        pause_times = np.zeros(n)
        cumulative = np.cumsum(base_times)
        is_rare = np.zeros(n, dtype=bool)
        return cls(base_times, pause_times, cumulative, is_rare)
