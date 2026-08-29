import pytest
from ringstrain.features import featurize_smiles, FEATURE_NAMES


def test_featurize_returns_all_expected_keys():
    primary, _all, _mol, _rings = featurize_smiles("C1CC1")
    for name in FEATURE_NAMES:
        assert name in primary


def test_cyclopropane_ring_size_three():
    primary, _all, _mol, _rings = featurize_smiles("C1CC1")
    assert primary["ring_size"] == 3
    assert primary["n_hetero_total"] == 0


def test_oxirane_detects_one_oxygen():
    primary, _all, _mol, _rings = featurize_smiles("C1CO1")
    assert primary["n_hetero_O"] == 1
    assert primary["n_hetero_total"] == 1


def test_triple_bond_sets_sp_ideal_angle_flag():
    primary, _all, _mol, _rings = featurize_smiles("C1CCCCCC#C1")  # cyclooctyne
    assert primary["has_triple_bond"] == 1
    # sp-center forced into an 8-ring should register a much larger Baeyer
    # proxy than the saturated 8-membered ring (cyclooctane).
    sat_primary, _, _, _ = featurize_smiles("C1CCCCCCC1")
    assert primary["baeyer_strain_proxy"] > sat_primary["baeyer_strain_proxy"]


def test_polycyclic_aggregates_multiple_rings():
    primary, all_feats, _mol, rings = featurize_smiles("C1C2CC1C2")  # BCP
    assert primary["n_rings_in_molecule"] == 2
    assert len(all_feats) == 2
    assert primary["sum_baeyer_proxy_all_rings"] >= primary["baeyer_strain_proxy"]


def test_no_ring_raises_valueerror():
    with pytest.raises(ValueError):
        featurize_smiles("CCCC")


def test_adjacent_heteroatom_flag():
    # 1,3-dioxolane: two O atoms are NOT adjacent (separated by C on both sides)
    primary, _all, _mol, _rings = featurize_smiles("C1OCOC1")
    assert primary["hetero_adjacent"] == 0
