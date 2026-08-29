"""
features.py
-----------
Ring perception + featurization.

Ring perception strategy (works well for the mono/bi/tricyclic small-molecule
regime this project targets): build the molecular graph, then for every
"extra" edge beyond a spanning tree (there are E - N + 1 of them, the
cyclomatic number) compute the shortest path between its two endpoints in the
graph *without* that edge — that path plus the edge is a ring. This isn't a
full minimum-cycle-basis (SSSR) implementation, but it correctly recovers
ring sizes for the fused/bridged bicyclics used here (norbornane, etc.).

Electronegativity and covalent-radius style constants below are standard
textbook values, used only to build smooth numeric features (not for any
quantum-chemical claim).
"""
from __future__ import annotations
from collections import deque
import math

from .smiles import Molecule, parse_smiles

# Pauling electronegativity, used as a smooth per-element numeric feature.
# Extended beyond the CHNOPS/halogen "organic subset" to cover the
# heavier main-group elements (groups 13-16) that show up in recent
# ring-strain literature on element-substituted small rings (silacycles,
# phosphacycles, boracycles, etc.) — see README "Recent/forefront systems".
ELECTRONEGATIVITY = {
    "C": 2.55, "N": 3.04, "O": 3.44, "S": 2.58, "P": 2.19,
    "F": 3.98, "Cl": 3.16, "Br": 2.96, "I": 2.66, "B": 2.04, "Si": 1.90,
    "Se": 2.55, "As": 2.18, "Ge": 2.01, "Sn": 1.96, "Pb": 2.33,
    "Al": 1.61, "Ga": 1.81, "In": 1.78, "Tl": 1.62,
    "Sb": 2.05, "Bi": 2.02, "Te": 2.10, "Po": 2.00,
}
COVALENT_RADIUS = {  # Angstrom, single-bond
    "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07,
    "F": 0.57, "Cl": 1.02, "Br": 1.20, "I": 1.39, "B": 0.84, "Si": 1.11,
    "Se": 1.20, "As": 1.19, "Ge": 1.20, "Sn": 1.39, "Pb": 1.46,
    "Al": 1.21, "Ga": 1.22, "In": 1.42, "Tl": 1.45,
    "Sb": 1.39, "Bi": 1.48, "Te": 1.38, "Po": 1.40,
}
# Elements present in the current curated training set (data/strain_energies.csv)
# — used by predict.py to flag genuine extrapolation onto unseen chemistry
# (e.g. a novel silacycle or organoboron ring) rather than just novel ring size.
TRAINED_ELEMENTS = {"C", "N", "O", "S", "Si", "P"}

IDEAL_ANGLE_SP3 = 109.5
IDEAL_ANGLE_SP2 = 120.0
IDEAL_ANGLE_SP = 180.0


def _shortest_path(mol: Molecule, start: int, goal: int, forbidden_edge):
    """BFS shortest path from start to goal avoiding one specific edge."""
    fu, fv = forbidden_edge
    q = deque([start])
    prev = {start: None}
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v, _order in mol.atoms[u].neighbors:
            if {u, v} == {fu, fv}:
                continue
            if v not in prev:
                prev[v] = u
                q.append(v)
    if goal not in prev:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def perceive_rings(mol: Molecule):
    """Return a list of rings; each ring is a list of atom indices in order."""
    bonds = mol.bonds()
    n_atoms = mol.num_atoms()
    n_bonds = len(bonds)
    n_rings = n_bonds - n_atoms + 1  # cyclomatic number (assumes 1 component)
    if n_rings <= 0:
        return []

    # Build a spanning tree greedily; extra edges are ring-closing edges.
    parent = list(range(n_atoms))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree_edges, extra_edges = [], []
    for u, v, order in bonds:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            tree_edges.append((u, v))
        else:
            extra_edges.append((u, v))

    rings = []
    for (u, v) in extra_edges:
        path = _shortest_path(mol, u, v, (u, v))
        if path:
            rings.append(path)
    return rings


def _ring_bond_orders(mol: Molecule, ring):
    orders = []
    ring_set = set(ring)
    m = len(ring)
    for i in range(m):
        a, b = ring[i], ring[(i + 1) % m]
        order = None
        for nbr, o in mol.atoms[a].neighbors:
            if nbr == b:
                order = o
                break
        orders.append(order if order is not None else 1)
    return orders


def _ideal_internal_angle(ring_size: int) -> float:
    """Interior angle of the regular planar polygon with `ring_size` vertices."""
    return (ring_size - 2) * 180.0 / ring_size


def featurize_ring(mol: Molecule, ring: list) -> dict:
    """Compute a feature dict describing one ring of a parsed molecule."""
    ring_size = len(ring)
    orders = _ring_bond_orders(mol, ring)
    symbols = [mol.atoms[i].symbol.capitalize() if mol.atoms[i].symbol.lower()
               not in ("c",) else ("C" if not mol.atoms[i].aromatic else "c")
               for i in ring]
    # normalize element symbol (strip aromatic lower-casing for chemistry, but
    # keep an aromatic flag separately)
    elements = [mol.atoms[i].symbol[0].upper() + mol.atoms[i].symbol[1:].lower()
                if len(mol.atoms[i].symbol) > 1 else mol.atoms[i].symbol.upper()
                for i in ring]
    aromatic_flags = [mol.atoms[i].aromatic for i in ring]

    n_hetero_O = elements.count("O")
    n_hetero_N = elements.count("N")
    n_hetero_S = elements.count("S")
    n_hetero_other = sum(
        1 for e in elements if e not in ("C", "O", "N", "S")
    )
    n_hetero_total = n_hetero_O + n_hetero_N + n_hetero_S + n_hetero_other

    n_double_in_ring = sum(1 for o in orders if o == 2)
    n_triple_in_ring = sum(1 for o in orders if o == 3)
    n_aromatic_bonds = sum(1 for o in orders if o == 1.5)
    is_aromatic = all(aromatic_flags)

    # ring fusion / bridging: count ring atoms that also belong to >1 ring
    # (approximate via degree > 2 within the ring-atom set, i.e. atoms whose
    # neighbors inside the whole molecule include >2 other ring atoms of ANY
    # ring — computed by caller via shared-atom counting across all rings).
    degrees = [len(mol.atoms[i].neighbors) for i in ring]
    avg_substitution = sum(d - 2 for d in degrees) / ring_size  # exocyclic subst.

    electroneg_sum = sum(ELECTRONEGATIVITY.get(e, 2.2) for e in elements)
    electroneg_mean = electroneg_sum / ring_size
    radius_mean = sum(COVALENT_RADIUS.get(e, 0.77) for e in elements) / ring_size

    # Ideal hybridization angle: sp (180 deg, triple bond) dominates over
    # sp2 (120 deg, double bond / aromatic) which dominates over sp3
    # (109.5 deg). This is what makes small/medium-ring cycloalkynes
    # (cyclooctyne, cycloheptyne...) register as strained even though an
    # 8-membered ring is otherwise comfortable for sp3/sp2 atoms — forcing
    # a ~180 degree center into an 8-membered ring is the real story behind
    # strain-promoted click chemistry.
    if n_triple_in_ring:
        ideal_angle = IDEAL_ANGLE_SP
    elif n_double_in_ring or is_aromatic:
        ideal_angle = IDEAL_ANGLE_SP2
    else:
        ideal_angle = IDEAL_ANGLE_SP3
    planar_interior_angle = _ideal_internal_angle(ring_size)
    # Baeyer angle-strain proxy: deviation of the planar-polygon interior
    # angle from the "ideal" hybridization angle, squared and scaled by ring
    # size (classic Baeyer 1885 argument).
    baeyer_strain_proxy = ring_size * (planar_interior_angle - ideal_angle) ** 2

    # Adjacent-heteroatom flag: two heteroatoms next to each other in the
    # ring (peroxide-like O-O, hydrazine-like N-N, disulfide-like S-S, ...)
    # have extra lone-pair repulsion beyond plain angle/torsional strain.
    hetero_adjacent = 0
    for i in range(ring_size):
        e1, e2 = elements[i], elements[(i + 1) % ring_size]
        if e1 != "C" and e2 != "C":
            hetero_adjacent = 1
            break

    # Pitzer torsional-strain proxy: planar rings force eclipsing; the number
    # of eclipsing C-H/ring-substituent interactions scales with ring size
    # for small rings and vanishes once puckering fully relieves it (~ring
    # size >= 6). We model it as a decaying function peaking at ring size 4-5.
    pitzer_strain_proxy = ring_size * math.exp(-((ring_size - 4.5) ** 2) / 8.0)

    return {
        "ring_size": ring_size,
        "n_hetero_O": n_hetero_O,
        "n_hetero_N": n_hetero_N,
        "n_hetero_S": n_hetero_S,
        "n_hetero_other": n_hetero_other,
        "n_hetero_total": n_hetero_total,
        "n_double_in_ring": n_double_in_ring,
        "n_triple_in_ring": n_triple_in_ring,
        "n_aromatic_bonds": n_aromatic_bonds,
        "is_aromatic": int(is_aromatic),
        "avg_exocyclic_substitution": avg_substitution,
        "electronegativity_mean": electroneg_mean,
        "covalent_radius_mean": radius_mean,
        "baeyer_strain_proxy": baeyer_strain_proxy,
        "pitzer_strain_proxy": pitzer_strain_proxy,
        "hetero_adjacent": hetero_adjacent,
        "has_triple_bond": int(bool(n_triple_in_ring)),
        "inv_ring_size": 1.0 / ring_size,
        "ring_size_sq": ring_size ** 2,
    }


# Per-ring features (computed from just the smallest/primary ring).
RING_FEATURE_NAMES = [
    "ring_size", "n_hetero_O", "n_hetero_N", "n_hetero_S", "n_hetero_other",
    "n_hetero_total", "n_double_in_ring", "n_triple_in_ring",
    "n_aromatic_bonds", "is_aromatic", "avg_exocyclic_substitution",
    "electronegativity_mean", "covalent_radius_mean", "baeyer_strain_proxy",
    "pitzer_strain_proxy", "hetero_adjacent", "has_triple_bond",
    "inv_ring_size", "ring_size_sq",
]

# Molecule-level features (computed across ALL perceived rings), appended
# after the per-ring ones. These matter for fused/bridged/spiro/cage systems
# (norbornane, bicyclo[1.1.1]pentane, cubane, ...) where strain lives in the
# whole polycyclic framework, not just its smallest ring.
MOLECULE_FEATURE_NAMES = [
    "n_rings_in_molecule",
    "sum_baeyer_proxy_all_rings",
    "sum_pitzer_proxy_all_rings",
    "mean_ring_size_all_rings",
    "min_ring_size_all_rings",
]

FEATURE_NAMES = RING_FEATURE_NAMES + MOLECULE_FEATURE_NAMES


def featurize_smiles(smiles: str):
    """Parse a SMILES and return (feature_dict, all_rings_features, mol, rings).

    `feature_dict` combines the smallest/primary ring's features (which
    dominate strain for monocyclic input) with molecule-level aggregates
    computed across every perceived ring (which matter for fused/bridged/
    spiro/cage polycyclics). Its keys match FEATURE_NAMES exactly, so it can
    be fed straight into the model. `all_rings_features` gives the raw
    per-ring dicts for callers that want more detail.
    """
    mol = parse_smiles(smiles)
    rings = perceive_rings(mol)
    if not rings:
        raise ValueError(f"No ring found in SMILES: {smiles!r}")
    all_feats = [featurize_ring(mol, r) for r in rings]
    primary = dict(min(all_feats, key=lambda f: f["ring_size"]))

    primary["n_rings_in_molecule"] = len(rings)
    primary["sum_baeyer_proxy_all_rings"] = sum(f["baeyer_strain_proxy"] for f in all_feats)
    primary["sum_pitzer_proxy_all_rings"] = sum(f["pitzer_strain_proxy"] for f in all_feats)
    primary["mean_ring_size_all_rings"] = sum(f["ring_size"] for f in all_feats) / len(all_feats)
    primary["min_ring_size_all_rings"] = min(f["ring_size"] for f in all_feats)

    return primary, all_feats, mol, rings
