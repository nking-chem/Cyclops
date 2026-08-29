import pytest
from ringstrain.predict import predict_strain
from ringstrain.reactions import compare_schemes, is_simple_unsubstituted_cycloalkane


def test_predict_returns_positive_strain_for_cyclopropane():
    result = predict_strain("C1CC1")
    assert result["predicted_strain_kcal_mol"] > 15  # cyclopropane is highly strained
    assert result["ring_class"] == "carbocycle"
    assert result["primary_ring_size"] == 3


def test_predict_flags_heteroatom():
    result = predict_strain("C1CO1")  # oxirane
    assert "O" in result["heteroatoms_in_ring"]
    assert result["ring_class"] == "heterocycle"


def test_predict_flags_novel_element():
    result = predict_strain("B1CCC1")  # boracycle, not in training data
    assert "B" in result["novel_elements_not_in_training_data"]


def test_predict_flags_polycyclic():
    result = predict_strain("C1C2CC1C2")  # bicyclo[1.1.1]pentane
    assert result["polycyclic"] is True
    assert result["n_rings_in_molecule"] == 2


def test_predict_rejects_acyclic_input():
    with pytest.raises(ValueError):
        predict_strain("CCCC")


def test_reaction_scheme_supported_for_simple_cycloalkane():
    assert is_simple_unsubstituted_cycloalkane("C1CCCCC1") == 6
    assert is_simple_unsubstituted_cycloalkane("CC1CCCCC1") is None  # substituted
    assert is_simple_unsubstituted_cycloalkane("C1CO1") is None  # heteroatom


def test_reaction_scheme_comparison_cyclopropane():
    result = compare_schemes("C1CC1", max_level=2)
    assert result["supported"] is True
    assert result["has_thermochemical_data"] is True
    levels = {c["level"]: c["strain_kcal_mol"] for c in result["computed_strain_by_level"]}
    # Known qualitative behavior: isodesmic (methane-referenced) underestimates
    # vs. homodesmotic/hyperhomodesmotic for cyclopropane.
    assert levels[0] < levels[1]


def test_reaction_scheme_unsupported_for_heterocycle():
    result = compare_schemes("C1CO1")
    assert result["supported"] is False


def test_reaction_scheme_generates_equations_even_when_no_data():
    # A cycloalkane ring size beyond our tabulated dHf range still gets
    # correctly-generated equations, just no numeric answer.
    result = compare_schemes("C1CCCCCCCCCCCCCCC1", max_level=1)  # cyclopentadecane
    assert result["supported"] is True
    assert result["has_thermochemical_data"] is False
    assert len(result["reactions"]) == 2
