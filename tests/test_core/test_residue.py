"""Tests for amino acid types and properties."""

from cotransfold.core.residue import (
    AminoAcid, RESIDUE_PROPERTIES, ONE_LETTER, THREE_LETTER,
    sequence_from_string, sequence_to_string,
)


def test_amino_acid_enum_has_20_members():
    assert len(AminoAcid) == 20


def test_one_letter_covers_all():
    assert len(ONE_LETTER) == 20
    for aa in AminoAcid:
        assert aa in ONE_LETTER.values()


def test_three_letter_covers_all():
    assert len(THREE_LETTER) == 20
    for aa in AminoAcid:
        assert aa in THREE_LETTER


def test_residue_properties_all_defined():
    for aa in AminoAcid:
        assert aa in RESIDUE_PROPERTIES
        props = RESIDUE_PROPERTIES[aa]
        assert props.mass > 0
        assert props.vdw_radius >= 0


def test_charged_residues():
    assert RESIDUE_PROPERTIES[AminoAcid.ASP].charge == -1.0
    assert RESIDUE_PROPERTIES[AminoAcid.GLU].charge == -1.0
    assert RESIDUE_PROPERTIES[AminoAcid.ARG].charge == 1.0
    assert RESIDUE_PROPERTIES[AminoAcid.LYS].charge == 1.0
    assert RESIDUE_PROPERTIES[AminoAcid.ALA].charge == 0.0


def test_glycine_smallest_sidechain():
    assert RESIDUE_PROPERTIES[AminoAcid.GLY].vdw_radius == 0.0


def test_sequence_from_string():
    seq = sequence_from_string("ACDE")
    assert seq == [AminoAcid.ALA, AminoAcid.CYS, AminoAcid.ASP, AminoAcid.GLU]


def test_sequence_roundtrip():
    original = "ARNDCEQGHILKMFPSTWYV"
    seq = sequence_from_string(original)
    result = sequence_to_string(seq)
    assert result == original
