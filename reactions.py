"""
reactions.py
------------
Compares ring strain energy across the isodesmic -> homodesmotic ->
hyperhomodesmotic (-> higher order) hierarchy of balanced reference
reactions, for simple unsubstituted cycloalkanes (CnH2n).

WHY THIS SCOPE. Constructing a *correct* reaction at each level of the
hierarchy requires reference fragments whose bonding environment matches
the ring atoms' environment more and more closely as the level increases
(Hehre/Pople 1970; George, Trachtman, Bock & Brett 1976; Wheeler, Houk,
Schleyer & Schwarz, JACS 2009, 131, 2547 for the modern RC1-RC5 naming).
For unsubstituted saturated carbocycles this has a clean, well-established
"telescoping" form:

    level L:   cyclo-(CH2)n  +  n * [n-alkane with (L+2) carbons]
                             -> n * [n-alkane with (L+3) carbons]

    L=0  isodesmic         CH4  -> C2H6     (methane/ethane)
    L=1  homodesmotic      C2H6 -> C3H8     (ethane/propane)   <- IUPAC Gold
                                                                   Book's own
                                                                   example for
                                                                   cyclopropane
    L=2  hyperhomodesmotic C3H8 -> C4H10    (propane/n-butane)
    L=3+ higher order      C4H10 -> C5H12, C5H12 -> C6H14, ...

Each step out conserves one more shell of the local bonding environment
around a ring -CH2- unit, which is exactly why strain-energy ESTIMATES
change (and converge) as you climb the hierarchy — the whole point of the
exercise. We are not fabricating this convergence: it's computed from real
experimental gas-phase heats of formation (see the two reference tables
below), so the numbers a user sees here are genuine thermochemical
arithmetic, not model output.

SCOPE LIMIT: this module only handles simple, unsubstituted, saturated,
monocyclic carbocycles (cyclopropane .. cyclodecane). Heteroatom rings,
substituted rings, unsaturated rings, and polycyclics need reference
fragments that are no longer a simple telescoping alkane series — IUPAC
notes the homodesmotic idea "may be extended to molecules with heteroatoms"
but doing that correctly per-ring is beyond what this offline tool
fabricates numbers for. For those cases we only auto-generate the
isodesmic/homodesmotic reaction equation as text (still useful — building
these by hand is tedious and error-prone) without a computed energy.
"""
from __future__ import annotations
from .smiles import parse_smiles
from .features import perceive_rings

# ---------------------------------------------------------------------
# Reference data: standard gas-phase enthalpies of formation, kcal/mol,
# 298 K. These are the widely reproduced textbook/Pedley-compilation
# values used in essentially every ring-strain derivation since Wiberg's
# 1986 Angewandte Chemie review ("The Concept of Strain in Organic
# Chemistry") and Cox & Pilcher's 1970 thermochemistry compendium.
# ---------------------------------------------------------------------
LINEAR_ALKANE_DHF = {  # n_carbons -> dHf(gas), kcal/mol
    1: -17.8,   # methane
    2: -20.2,   # ethane
    3: -25.0,   # propane
    4: -30.1,   # n-butane
    5: -35.1,   # n-pentane
    6: -39.9,   # n-hexane
    7: -44.9,   # n-heptane
    8: -49.9,   # n-octane
}

CYCLOALKANE_DHF = {  # ring_size -> dHf(gas), kcal/mol
    3: 12.7,    # cyclopropane
    4: 6.4,     # cyclobutane
    5: -18.5,   # cyclopentane
    6: -29.4,   # cyclohexane
    7: -28.2,   # cycloheptane
    8: -29.7,   # cyclooctane
    9: -31.7,   # cyclononane
    10: -36.9,  # cyclodecane
}

SCHEME_NAMES = {
    0: "isodesmic",
    1: "homodesmotic",
    2: "hyperhomodesmotic",
}


def _scheme_name(level: int) -> str:
    return SCHEME_NAMES.get(level, f"higher-order (level {level})")


def _alkane_smiles(n_carbons: int) -> str:
    return "C" * n_carbons  # SMILES for the linear alkane CnH(2n+2)


def is_simple_unsubstituted_cycloalkane(smiles: str):
    """Check whether `smiles` is a monocyclic, saturated, all-carbon ring
    with no exocyclic substituents (i.e. exactly CnH2n, one ring, no
    branches) — the class this module can build reaction schemes for.
    Returns the ring size if so, else None.
    """
    try:
        mol = parse_smiles(smiles)
    except Exception:
        return None
    rings = perceive_rings(mol)
    if len(rings) != 1:
        return None
    ring = rings[0]
    if len(ring) != mol.num_atoms():
        return None  # extra atoms outside the ring -> substituted
    for i in ring:
        a = mol.atoms[i]
        if a.symbol.upper() != "C" or a.aromatic:
            return None
        if len(a.neighbors) != 2:
            return None  # exocyclic branch or ring fusion
        if any(order != 1 for _n, order in a.neighbors):
            return None  # double/triple bond in ring
    return len(ring)


def generate_reaction(ring_size: int, level: int) -> dict:
    """Build the balanced reaction equation (text + SMILES) for a
    cyclo-(CH2)ring_size at the given hierarchy level. Always constructible,
    independent of whether we have a real dHf value to evaluate it with.
    """
    reactant_c = level + 1
    product_c = level + 2
    reactant_name = _alkane_name(reactant_c)
    product_name = _alkane_name(product_c)
    ring_smiles = "C1" + "C" * (ring_size - 2) + "1" if ring_size > 2 else "C1C1"
    equation = (
        f"cyclo-(CH2){ring_size}  +  {ring_size} {reactant_name}  "
        f"->  {ring_size} {product_name}"
    )
    return {
        "level": level,
        "scheme_name": _scheme_name(level),
        "equation_text": equation,
        "ring_smiles": ring_smiles,
        "reactant_alkane_smiles": _alkane_smiles(reactant_c),
        "reactant_alkane_name": reactant_name,
        "product_alkane_smiles": _alkane_smiles(product_c),
        "product_alkane_name": product_name,
        "conserves": _conservation_note(level),
    }


def _alkane_name(n_carbons: int) -> str:
    names = {1: "methane", 2: "ethane", 3: "propane", 4: "n-butane",
             5: "n-pentane", 6: "n-hexane", 7: "n-heptane", 8: "n-octane"}
    return names.get(n_carbons, f"C{n_carbons}H{2*n_carbons+2}")


def _conservation_note(level: int) -> str:
    if level == 0:
        return "conserves C-C and C-H bond counts only (isodesmic)"
    if level == 1:
        return ("additionally conserves carbon hybridization and the "
                "H-count on each carbon type (homodesmotic)")
    if level == 2:
        return ("additionally conserves the identity of each carbon's "
                "immediate neighbors, one shell further out "
                "(hyperhomodesmotic)")
    return f"conserves local bonding environment out to {level + 1} bonds from each ring carbon"


def compute_strain_via_scheme(ring_size: int, level: int):
    """Compute the strain energy implied by the reaction at `level`, using
    real experimental gas-phase heats of formation. Returns None if we
    don't have a tabulated dHf for this ring size (rather than guessing).

        SE(L) = dHf(ring) - n * [dHf(A_(L+2)) - dHf(A_(L+1))]

    (derived from ring + n*A_(L+1) -> n*A_(L+2); strain energy is minus the
    reaction enthalpy, i.e. how much less stable the ring is than the
    reaction's product side predicts.)
    """
    if ring_size not in CYCLOALKANE_DHF:
        return None
    reactant_c, product_c = level + 1, level + 2
    if reactant_c not in LINEAR_ALKANE_DHF or product_c not in LINEAR_ALKANE_DHF:
        return None
    dhf_ring = CYCLOALKANE_DHF[ring_size]
    ch2_increment = LINEAR_ALKANE_DHF[product_c] - LINEAR_ALKANE_DHF[reactant_c]
    se = dhf_ring - ring_size * ch2_increment
    return {
        "level": level,
        "scheme_name": _scheme_name(level),
        "strain_kcal_mol": round(se, 2),
        "dhf_ring_kcal_mol": dhf_ring,
        "ch2_increment_kcal_mol": round(ch2_increment, 2),
        "reactant_alkane": _alkane_name(reactant_c),
        "product_alkane": _alkane_name(product_c),
    }


def compare_schemes(smiles: str, max_level: int = 4) -> dict:
    """Top-level entry point: for a given SMILES, return the reaction
    equations (always) and computed strain energies (where we have real
    thermochemical data) across levels 0..max_level.
    """
    ring_size = is_simple_unsubstituted_cycloalkane(smiles)
    if ring_size is None:
        return {
            "supported": False,
            "reason": (
                "This module only builds reaction-scheme comparisons for "
                "simple, unsubstituted, saturated, monocyclic carbocycles "
                "(cyclopropane .. cyclodecane, i.e. CnH2n with no "
                "substituents, heteroatoms, unsaturation, or ring fusion). "
                "Heteroatom/substituted/polycyclic rings need custom "
                "reference fragments per ring that this offline tool "
                "doesn't fabricate — see README."
            ),
        }
    levels = list(range(max_level + 1))
    reactions = [generate_reaction(ring_size, L) for L in levels]
    computed = [compute_strain_via_scheme(ring_size, L) for L in levels]
    computed = [c for c in computed if c is not None]
    has_data = ring_size in CYCLOALKANE_DHF
    return {
        "supported": True,
        "ring_size": ring_size,
        "has_thermochemical_data": has_data,
        "dhf_ring_kcal_mol": CYCLOALKANE_DHF.get(ring_size),
        "reactions": reactions,
        "computed_strain_by_level": computed,
        "note": (
            None if has_data else
            f"No tabulated experimental dHf for the {ring_size}-membered "
            "cycloalkane in this tool's reference table — showing the "
            "balanced reaction equations only; supply your own dHf(ring) "
            "to get numeric strain energies from the formula in this "
            "module's docstring."
        ),
    }


if __name__ == "__main__":
    import sys
    import json
    smi = sys.argv[1] if len(sys.argv) > 1 else "C1CC1"
    print(json.dumps(compare_schemes(smi), indent=2))
