"""Amino acid types and their physical properties."""

from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass


class AminoAcid(IntEnum):
    """The 20 standard amino acids."""
    ALA = 0
    ARG = 1
    ASN = 2
    ASP = 3
    CYS = 4
    GLU = 5
    GLN = 6
    GLY = 7
    HIS = 8
    ILE = 9
    LEU = 10
    LYS = 11
    MET = 12
    PHE = 13
    PRO = 14
    SER = 15
    THR = 16
    TRP = 17
    TYR = 18
    VAL = 19


# One-letter code mapping
ONE_LETTER = {
    'A': AminoAcid.ALA, 'R': AminoAcid.ARG, 'N': AminoAcid.ASN,
    'D': AminoAcid.ASP, 'C': AminoAcid.CYS, 'E': AminoAcid.GLU,
    'Q': AminoAcid.GLN, 'G': AminoAcid.GLY, 'H': AminoAcid.HIS,
    'I': AminoAcid.ILE, 'L': AminoAcid.LEU, 'K': AminoAcid.LYS,
    'M': AminoAcid.MET, 'F': AminoAcid.PHE, 'P': AminoAcid.PRO,
    'S': AminoAcid.SER, 'T': AminoAcid.THR, 'W': AminoAcid.TRP,
    'Y': AminoAcid.TYR, 'V': AminoAcid.VAL,
}

THREE_LETTER = {
    AminoAcid.ALA: 'ALA', AminoAcid.ARG: 'ARG', AminoAcid.ASN: 'ASN',
    AminoAcid.ASP: 'ASP', AminoAcid.CYS: 'CYS', AminoAcid.GLU: 'GLU',
    AminoAcid.GLN: 'GLN', AminoAcid.GLY: 'GLY', AminoAcid.HIS: 'HIS',
    AminoAcid.ILE: 'ILE', AminoAcid.LEU: 'LEU', AminoAcid.LYS: 'LYS',
    AminoAcid.MET: 'MET', AminoAcid.PHE: 'PHE', AminoAcid.PRO: 'PRO',
    AminoAcid.SER: 'SER', AminoAcid.THR: 'THR', AminoAcid.TRP: 'TRP',
    AminoAcid.TYR: 'TYR', AminoAcid.VAL: 'VAL',
}


@dataclass(frozen=True)
class ResidueProperties:
    """Physical properties of an amino acid type."""
    charge: float           # Net charge at pH 7.4
    vdw_radius: float       # Side chain centroid VdW radius (Å)
    hydrophobicity: float   # Kyte-Doolittle scale, normalized to [-1, 1]
    mass: float             # Daltons (residue mass, not full AA)
    helix_propensity: float # Chou-Fasman helix propensity (>1 = helix-forming)
    sheet_propensity: float # Chou-Fasman sheet propensity (>1 = sheet-forming)


# Properties from AMBER ff14SB, Kyte-Doolittle, Chou-Fasman
RESIDUE_PROPERTIES: dict[AminoAcid, ResidueProperties] = {
    AminoAcid.ALA: ResidueProperties(
        charge=0.0, vdw_radius=1.7, hydrophobicity=0.40,
        mass=71.08, helix_propensity=1.42, sheet_propensity=0.83),
    AminoAcid.ARG: ResidueProperties(
        charge=1.0, vdw_radius=3.3, hydrophobicity=-1.00,
        mass=156.19, helix_propensity=0.98, sheet_propensity=0.93),
    AminoAcid.ASN: ResidueProperties(
        charge=0.0, vdw_radius=2.4, hydrophobicity=-0.78,
        mass=114.10, helix_propensity=0.67, sheet_propensity=0.89),
    AminoAcid.ASP: ResidueProperties(
        charge=-1.0, vdw_radius=2.4, hydrophobicity=-0.78,
        mass=115.09, helix_propensity=1.01, sheet_propensity=0.54),
    AminoAcid.CYS: ResidueProperties(
        charge=0.0, vdw_radius=2.1, hydrophobicity=0.56,
        mass=103.14, helix_propensity=0.70, sheet_propensity=1.19),
    AminoAcid.GLU: ResidueProperties(
        charge=-1.0, vdw_radius=2.7, hydrophobicity=-0.78,
        mass=129.12, helix_propensity=1.51, sheet_propensity=0.37),
    AminoAcid.GLN: ResidueProperties(
        charge=0.0, vdw_radius=2.7, hydrophobicity=-0.69,
        mass=128.13, helix_propensity=1.11, sheet_propensity=1.10),
    AminoAcid.GLY: ResidueProperties(
        charge=0.0, vdw_radius=0.0, hydrophobicity=-0.09,
        mass=57.05, helix_propensity=0.57, sheet_propensity=0.75),
    AminoAcid.HIS: ResidueProperties(
        charge=0.0, vdw_radius=2.6, hydrophobicity=-0.71,
        mass=137.14, helix_propensity=1.00, sheet_propensity=0.87),
    AminoAcid.ILE: ResidueProperties(
        charge=0.0, vdw_radius=2.2, hydrophobicity=1.00,
        mass=113.16, helix_propensity=1.08, sheet_propensity=1.60),
    AminoAcid.LEU: ResidueProperties(
        charge=0.0, vdw_radius=2.2, hydrophobicity=0.84,
        mass=113.16, helix_propensity=1.21, sheet_propensity=1.30),
    AminoAcid.LYS: ResidueProperties(
        charge=1.0, vdw_radius=3.0, hydrophobicity=-0.69,
        mass=128.17, helix_propensity=1.16, sheet_propensity=0.74),
    AminoAcid.MET: ResidueProperties(
        charge=0.0, vdw_radius=2.4, hydrophobicity=0.42,
        mass=131.20, helix_propensity=1.45, sheet_propensity=1.05),
    AminoAcid.PHE: ResidueProperties(
        charge=0.0, vdw_radius=2.7, hydrophobicity=0.62,
        mass=147.18, helix_propensity=1.13, sheet_propensity=1.38),
    AminoAcid.PRO: ResidueProperties(
        charge=0.0, vdw_radius=1.9, hydrophobicity=-0.31,
        mass=97.12, helix_propensity=0.57, sheet_propensity=0.55),
    AminoAcid.SER: ResidueProperties(
        charge=0.0, vdw_radius=1.8, hydrophobicity=-0.18,
        mass=87.08, helix_propensity=0.77, sheet_propensity=0.75),
    AminoAcid.THR: ResidueProperties(
        charge=0.0, vdw_radius=2.0, hydrophobicity=-0.16,
        mass=101.10, helix_propensity=0.83, sheet_propensity=1.19),
    AminoAcid.TRP: ResidueProperties(
        charge=0.0, vdw_radius=2.9, hydrophobicity=-0.20,
        mass=186.21, helix_propensity=1.08, sheet_propensity=1.37),
    AminoAcid.TYR: ResidueProperties(
        charge=0.0, vdw_radius=2.8, hydrophobicity=-0.27,
        mass=163.18, helix_propensity=0.69, sheet_propensity=1.47),
    AminoAcid.VAL: ResidueProperties(
        charge=0.0, vdw_radius=2.0, hydrophobicity=0.96,
        mass=99.13, helix_propensity=1.06, sheet_propensity=1.70),
}


def sequence_from_string(seq: str) -> list[AminoAcid]:
    """Convert one-letter amino acid string to list of AminoAcid."""
    return [ONE_LETTER[c.upper()] for c in seq]


def sequence_to_string(seq: list[AminoAcid]) -> str:
    """Convert list of AminoAcid to one-letter string."""
    to_one = {v: k for k, v in ONE_LETTER.items()}
    return ''.join(to_one[aa] for aa in seq)
