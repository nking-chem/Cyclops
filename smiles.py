"""
smiles.py
---------
A small, dependency-free SMILES parser sufficient for ring-strain feature
extraction. It is NOT a full cheminformatics toolkit (no valence checking,
no stereochemistry, no full aromaticity perception beyond lowercase atoms) —
it exists because the runtime this project was built in has no network
access to install RDKit.

This is a reference/offline implementation, not a permanent design choice.
If you have RDKit available, swapping it in (parse with Chem.MolFromSmiles,
use mol.GetRingInfo() for ring perception) will be more robust for exotic
valences, stereochemistry, and large fused/bridged polycyclics — see the
"Extending this platform" section of README.md for the specific functions
to replace.

Supported syntax:
  - Organic-subset atoms: B C N O P S F Cl Br I, aromatic b c n o p s
  - Bracket atoms: [nH], [O-], [NH2+], [13CH4], etc. (charge/isotope/H-count
    are parsed but only element + aromaticity + explicit H count are used)
  - Bonds: - = # : (default single/aromatic-implicit)
  - Branches: ( )
  - Ring closures: single digit 0-9 and %nn two-digit form
  - Dot-disconnected fragments: '.' (only first/largest fragment is used)
"""
from __future__ import annotations
from dataclasses import dataclass, field

ORGANIC_SUBSET = ["Cl", "Br", "B", "C", "N", "O", "P", "S", "F", "I"]
AROMATIC_SUBSET = ["b", "c", "n", "o", "p", "s"]

BOND_ORDER = {"-": 1, "=": 2, "#": 3, ":": 1.5}


@dataclass
class Atom:
    idx: int
    symbol: str
    aromatic: bool = False
    explicit_h: int | None = None
    charge: int = 0
    ring_ids: set = field(default_factory=set)  # filled in later
    neighbors: list = field(default_factory=list)  # list of (nbr_idx, order)


@dataclass
class Molecule:
    atoms: list
    smiles: str

    def num_atoms(self):
        return len(self.atoms)

    def bonds(self):
        seen = set()
        out = []
        for a in self.atoms:
            for nbr, order in a.neighbors:
                key = tuple(sorted((a.idx, nbr)))
                if key not in seen:
                    seen.add(key)
                    out.append((a.idx, nbr, order))
        return out


class SmilesParseError(ValueError):
    pass


def _read_bracket_atom(s: str, i: int):
    """Parse a bracket atom starting at s[i] == '['. Returns (Atom-partial, new_i)."""
    assert s[i] == "["
    j = i + 1
    start = j
    # isotope
    while j < len(s) and s[j].isdigit():
        j += 1
    j0 = j
    # element (1 or 2 letters)
    aromatic = False
    if j < len(s) and s[j].isalpha():
        if s[j].islower():
            aromatic = True
            elem = s[j]
            j += 1
        else:
            elem = s[j]
            j += 1
            if j < len(s) and s[j].islower() and (elem + s[j]) in (
                "Cl", "Br", "Se", "As", "Si", "Ge", "Sn", "Al", "Ga", "In",
                "Tl", "Sb", "Bi", "Te", "Po", "Pb", "Na", "Mg", "Ca", "Zn"
            ):
                elem += s[j]
                j += 1
    else:
        raise SmilesParseError(f"Bad bracket atom at {i} in {s!r}")
    explicit_h = 0
    charge = 0
    while j < len(s) and s[j] != "]":
        if s[j] == "H":
            j += 1
            k = j
            while j < len(s) and s[j].isdigit():
                j += 1
            explicit_h = int(s[k:j]) if j > k else 1
        elif s[j] in "+-":
            sign = 1 if s[j] == "+" else -1
            j += 1
            k = j
            while j < len(s) and s[j].isdigit():
                j += 1
            if j > k:
                charge = sign * int(s[k:j])
            else:
                # count repeated +/- signs like ++ or --
                cnt = 1
                while j < len(s) and s[j] == s[j - 1]:
                    cnt += 1
                    j += 1
                charge = sign * cnt
        elif s[j] == "@":
            j += 1
            if j < len(s) and s[j] == "@":
                j += 1
        elif s[j] == ":":
            # atom map number
            j += 1
            while j < len(s) and s[j].isdigit():
                j += 1
        else:
            j += 1
    if j >= len(s) or s[j] != "]":
        raise SmilesParseError(f"Unterminated bracket atom in {s!r}")
    j += 1
    return elem, aromatic, explicit_h, charge, j


def parse_smiles(smiles: str) -> Molecule:
    """Parse a (single-fragment) SMILES string into a Molecule graph."""
    s = smiles.strip()
    # only keep the largest fragment if multiple are dot-separated
    if "." in s:
        s = max(s.split("."), key=len)

    atoms: list[Atom] = []
    ring_bonds: dict = {}  # ring number -> (atom_idx, pending_bond_order or None)
    branch_stack: list = []
    prev_idx = None
    pending_bond = None  # bond order for the *next* bond
    i = 0
    n = len(s)

    def new_atom(elem, aromatic=False, explicit_h=None, charge=0):
        a = Atom(idx=len(atoms), symbol=elem, aromatic=aromatic,
                 explicit_h=explicit_h, charge=charge)
        atoms.append(a)
        return a

    def add_bond(i1, i2, order):
        atoms[i1].neighbors.append((i2, order))
        atoms[i2].neighbors.append((i1, order))

    def handle_ring_closure(num):
        nonlocal pending_bond
        if num in ring_bonds:
            other_idx, other_pending = ring_bonds.pop(num)
            order = pending_bond or other_pending or (
                1.5 if (atoms[prev_idx].aromatic and atoms[other_idx].aromatic) else 1
            )
            add_bond(other_idx, prev_idx, order)
            pending_bond = None
        else:
            ring_bonds[num] = (prev_idx, pending_bond)
            pending_bond = None

    while i < n:
        c = s[i]
        if c == "(":
            branch_stack.append(prev_idx)
            i += 1
            continue
        if c == ")":
            if not branch_stack:
                raise SmilesParseError(f"Unbalanced ')' at {i} in {s!r}")
            prev_idx = branch_stack.pop()
            i += 1
            continue
        if c in "-=#:/\\":
            pending_bond = BOND_ORDER.get(c, 1)
            i += 1
            continue
        if c == "[":
            elem, aromatic, eh, charge, j = _read_bracket_atom(s, i)
            a = new_atom(elem, aromatic, eh, charge)
            if prev_idx is not None:
                order = pending_bond if pending_bond is not None else (
                    1.5 if (a.aromatic and atoms[prev_idx].aromatic) else 1
                )
                add_bond(prev_idx, a.idx, order)
            pending_bond = None
            prev_idx = a.idx
            i = j
            continue
        if c == "%":
            # two-digit ring closure, must follow an atom
            num = int(s[i + 1:i + 3])
            i += 3
            handle_ring_closure(num)
            continue
        if c.isdigit():
            num = int(c)
            i += 1
            handle_ring_closure(num)
            continue
        # organic-subset atom (1 or 2 letter) or aromatic lowercase
        matched = None
        for sym in ORGANIC_SUBSET:
            if s.startswith(sym, i):
                matched = (sym, False)
                break
        if matched is None and c in AROMATIC_SUBSET:
            matched = (c, True)
        if matched is None and c == "*":
            matched = ("*", False)
        if matched is None:
            raise SmilesParseError(f"Unexpected character {c!r} at {i} in {s!r}")
        sym, aromatic = matched
        a = new_atom(sym, aromatic=aromatic)
        if prev_idx is not None:
            order = pending_bond if pending_bond is not None else (
                1.5 if (a.aromatic and atoms[prev_idx].aromatic) else 1
            )
            add_bond(prev_idx, a.idx, order)
        pending_bond = None
        prev_idx = a.idx
        i += len(sym)
        continue

    if branch_stack:
        raise SmilesParseError(f"Unbalanced '(' in {s!r}")
    if ring_bonds:
        raise SmilesParseError(f"Unclosed ring bond(s) {list(ring_bonds)} in {s!r}")

    return Molecule(atoms=atoms, smiles=smiles)


