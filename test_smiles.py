import pytest
from ringstrain.smiles import parse_smiles, SmilesParseError
from ringstrain.features import perceive_rings


def test_simple_ring_atom_and_bond_counts():
    mol = parse_smiles("C1CC1")  # cyclopropane
    assert mol.num_atoms() == 3
    assert len(mol.bonds()) == 3


def test_chain_has_no_ring():
    mol = parse_smiles("CCCC")
    assert perceive_rings(mol) == []


def test_ring_closure_digit_and_percent_form_agree():
    a = parse_smiles("C1CCCCCCCCC1")  # single-digit closures
    b = parse_smiles("C%10CCCCCCCCC%10")  # two-digit closure form
    assert a.num_atoms() == b.num_atoms()
    assert len(a.bonds()) == len(b.bonds())


def test_bridged_bicyclic_ring_count():
    mol = parse_smiles("C1CC2CCC1C2")  # norbornane
    rings = perceive_rings(mol)
    assert mol.num_atoms() == 7
    assert len(mol.bonds()) == 8  # cyclomatic number 2 -> bicyclic
    assert len(rings) == 2


def test_aromatic_lowercase_atoms():
    mol = parse_smiles("c1ccccc1")  # benzene
    assert mol.num_atoms() == 6
    assert all(a.aromatic for a in mol.atoms)


def test_bracket_atom_with_explicit_hydrogens():
    mol = parse_smiles("[SiH2]1CCC1")  # silacyclobutane
    si = mol.atoms[0]
    assert si.symbol == "Si"
    assert si.explicit_h == 2


def test_triple_bond_in_ring():
    mol = parse_smiles("C1CCCCCC#C1")  # cyclooctyne
    orders = [o for _a, _b, o in mol.bonds()]
    assert 3 in orders


def test_unbalanced_branch_raises():
    with pytest.raises(SmilesParseError):
        parse_smiles("C1CC(C1")


def test_unclosed_ring_raises():
    with pytest.raises(SmilesParseError):
        parse_smiles("C1CCC")
