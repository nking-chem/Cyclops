# Cyclops — Ring Strain Energy ML Platform

[![CI](https://github.com/your-org/strain-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/strain-bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

A small, self-contained platform that predicts **ring strain energy**
(kcal/mol) for carbocycles and heterocycles from a SMILES string, compares
strain estimates across the isodesmic → homodesmotic → hyperhomodesmotic
reaction-scheme hierarchy, and recognizes specific scaffolds used in
pharmaceutical chemistry. It includes a Python library, a CLI, and a local
web app with a live "gauge" readout.

> **Read this before trusting a number it gives you:** the ML model is
> trained on ~50 curated data points. That's enough to be a useful,
> honestly-benchmarked starting point — not a replacement for DFT,
> ab initio calculation, or the primary literature. See
> [Honest limitations](#️-honest-limitations--please-read-before-relying-on-this)
> below for exactly where the lines are.

```
strain-bench/
  ringstrain/
    smiles.py      # dependency-free SMILES parser (offline fallback for RDKit)
    features.py    # ring perception + featurization (Baeyer/Pitzer-style proxies)
    model.py        # trains + leave-one-out-evaluates candidate regressors
    predict.py      # prediction API + CLI
    reactions.py    # isodesmic/homodesmotic/hyperhomodesmotic reaction-scheme hierarchy
    data/
      strain_energies.csv    # curated seed dataset (51 rings)
      pharma_scaffolds.csv   # pharma-chemistry scaffold gallery (ML-predicted, no ground truth)
      model.pkl               # trained model (generated on first run, gitignored)
  tests/
    test_smiles.py, test_features.py, test_predict.py
  app.py            # Flask backend for the web platform
  webapp/index.html # frontend (SMILES input, reference chips, gauge, reaction-scheme panel)
  .github/workflows/ci.yml
  pyproject.toml
  requirements.txt
  LICENSE / CONTRIBUTING.md
```

## Quickstart

```bash
git clone https://github.com/your-org/strain-bench.git
cd strain-bench
pip install -e .

# CLI
python -m ringstrain.predict "C1CC1" "C1CO1" "C1CCCCC1"

# Web platform
python app.py
# open http://localhost:5050
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

CI (`.github/workflows/ci.yml`) runs the test suite on Python 3.10–3.12,
validates that every SMILES in both CSV datasets still parses, retrains the
model as a smoke test, and runs a CLI smoke test on every push and PR.

## What's new: extended scope beyond the original published set

The platform now reaches beyond textbook mono-cyclic strain values into
current research chemistry:

- **Strained cyclic alkynes** (cyclopentyne → cyclooctyne, plus the
  bioorthogonal-click reagent difluorocyclooctyne/DIFO) — real G3-level
  ab initio values from Bach, *JACS* 2009, 131, 5233. These are the
  workhorse alkynes behind strain-promoted azide–alkyne "click" chemistry
  (SPAAC), and they exercise a new part of the model: an sp-hybridized
  (180°) ring atom forced into a 5–8-membered ring, distinct from the
  sp2/sp3 strain regimes the original carbocycle/heterocycle set covered.
- **Bicyclo[1.1.1]pentane (BCP)**, the benzene bioisostere widely used in
  modern medicinal chemistry (strain ≈ 66.6 kcal/mol) — a genuinely
  bridged bicyclic cage, testing the new whole-molecule aggregate features.
- **Main-group heterocycles** (silacyclobutane, phosphetane) — rings built
  from elements outside the original CHNOS set, per recent group 13–16
  4-membered-ring strain surveys.
- **A broader heterocycle set**: morpholine, piperazine, 1,4-dioxane,
  1,3-dioxolane, oxazolidine, imidazolidine, thiazolidine, and the
  7-membered azepane/oxepane/thiepane series.
- **Macrocycles out to C17**, including a case (cycloheptadecane) where
  strain is *slightly negative* — a good reminder that "ring strain" isn't
  monotonic in ring size.

### New features that make this possible

- **sp/sp2/sp3-aware Baeyer proxy**: the ideal-angle term now recognizes
  ring triple bonds (180°) as well as double bonds/aromaticity (120°), so
  cyclic alkynes register correctly instead of being scored as ordinary
  saturated rings.
- **Molecule-level aggregate features** (`n_rings_in_molecule`,
  summed Baeyer/Pitzer proxies across every perceived ring, mean/min ring
  size): these feed into the *same* regressor alongside the primary-ring
  features, so bridged/fused/cage systems like BCP get real signal about
  their whole framework, not just their smallest ring.
- **Adjacent-heteroatom flag**, for peroxide-/hydrazine-/disulfide-like
  X–X ring bonds.
- **Extended element table** (Si, Ge, Sn, Al, Ga, In, Tl, Sb, Bi, Te, Po,
  Pb, B, P, As, Se — electronegativity + covalent radius), and the parser's
  bracket-atom reader now recognizes these two-letter symbols.
- **Genuine-extrapolation detection**: predictions now flag not just
  out-of-range ring sizes but ring atoms of an **element never seen in
  training** (e.g. try `B1CCC1`, a borole-type ring) — a more honest
  signal than ring-size range alone.

## Reaction-scheme hierarchy: isodesmic → homodesmotic → hyperhomodesmotic

`ringstrain/reactions.py` compares how the strain-energy *estimate* itself
changes depending on which balanced reference reaction you use to define
"strain-free" — the isodesmic/homodesmotic/hyperhomodesmotic hierarchy
(Hehre, Ditchfield, Radom & Pople 1970; George, Trachtman, Bock & Brett
1976; formalized as RC1–RC5 by Wheeler, Houk, Schleyer & Schwarz, *JACS*
2009, 131, 2547). This is genuinely useful beyond pedagogy: building a
correctly-balanced homodesmotic reaction by hand is tedious and a common
source of error in computational papers.

**What it does, concretely, for simple cycloalkanes** (cyclopropane ..
cyclodecane — unsubstituted, saturated, monocyclic CnH2n): the hierarchy
has a clean telescoping form,

```
level 0 (isodesmic):          cyclo-(CH2)n + n CH4      -> n C2H6
level 1 (homodesmotic):       cyclo-(CH2)n + n C2H6     -> n C3H8
level 2 (hyperhomodesmotic):  cyclo-(CH2)n + n C3H8     -> n C4H10
level 3+ (higher order):      cyclo-(CH2)n + n C4H10    -> n C5H12   ...
```

(Level 1 is exactly the IUPAC Gold Book's own homodesmotic example for
cyclopropane.) Each step conserves one more shell of the local bonding
environment around a ring −CH2− unit. Using real experimental gas-phase
ΔfH° values (Wiberg 1986 / Cox & Pilcher 1970 style compilation — see the
`LINEAR_ALKANE_DHF` / `CYCLOALKANE_DHF` tables in `reactions.py`), the tool
computes a **real, non-fabricated** strain energy at each level via

```
SE(level) = ΔfH°(ring) − n × [ΔfH°(next alkane) − ΔfH°(current alkane)]
```

For example, cyclopropane: isodesmic gives 19.9 kcal/mol, homodesmotic
27.1, hyperhomodesmotic 28.0, converging near the accepted ~27.6 — the
isodesmic (methane-referenced) estimate is known to be poor because
methane's C–H bonds aren't representative of a generic −CH2− environment,
which is exactly why the hierarchy was invented. Cyclohexane is a more
dramatic illustration: isodesmic gives −15.0 kcal/mol (badly wrong sign!),
while homodesmotic/hyperhomodesmotic correctly converge to ≈0.

**Scope limit, explicitly:** this real-arithmetic comparison only runs for
simple unsubstituted cycloalkanes, because that's the class of molecule
where the "add one more CH2" reference-fragment pattern is unambiguous and
where this tool has trustworthy experimental ΔfH° data. For anything else
(heteroatom rings, substituted rings, unsaturated rings, polycyclics), the
tool still **auto-generates the correctly-balanced reaction equation and
explains what it conserves at each level** — useful on its own — but does
not invent a numeric answer, since that would need either real ΔfH° data
for that specific ring or an actual electronic-structure calculation,
neither of which this offline sandbox can produce. Try it via:

```bash
python -m ringstrain.reactions "C1CC1"       # cyclopropane, full comparison
python -m ringstrain.reactions "C1CCCCC1"    # cyclohexane
```

or `POST /api/reaction_schemes {"smiles": ..., "max_level": 4}` / the
"Reaction-scheme hierarchy" panel in the web UI, which appears
automatically under the gauge after every prediction.

## Pharmaceutical-chemistry scaffolds

The platform now recognizes specific ring motifs that show up constantly in
drug discovery — not just generic ring sizes, but the actual named
scaffolds medicinal chemists reach for:

**Added with real data** (in the main training set, `strain_energies.csv`):
- **Bicyclo[2.2.2]octane (BCO)** — 9.8 kcal/mol. Not from a single
  literature-quoted number; computed here via Benson group-additivity
  against BCO's real NIST-tabulated ΔfH°(gas) = −23.67 kcal/mol (Wong &
  Westrum 1971). The same method reproduces norbornane's literature strain
  (17.2 kcal/mol) to within ~1 kcal/mol, which is the cross-check that
  makes the BCO number trustworthy. BCO (and its 2-oxa/1-aza analogs) is a
  benzene bioisostere used in Imatinib- and Vorinostat-type redesigns.
- **Cubane** — 166 kcal/mol, the classic widely-cited figure since Eaton's
  original synthesis papers. Another rigid benzene bioisostere core.

**Pharma scaffold gallery** (`data/pharma_scaffolds.csv`, ML-predicted —
no tabulated experimental value, shown as a distinct chip row in the UI
with drug-name tooltips):
- **2-Oxabicyclo[2.2.2]octane** — 2023 Nature Communications phenyl
  bioisostere (Levterov et al.), used in Imatinib/Vorinostat analogs.
- **Quinuclidine** (1-azabicyclo[2.2.2]octane) — the rigid amine cage in
  quinine, quinidine, solifenacin, cevimeline; topologically BCO with one
  bridgehead swapped for N.
- **Spiro[3.3]heptane** — sp3-rich benzene bioisostere (Prysiazhniuk et
  al., *Angew. Chem.* 2024), incorporated into sonidegib/vorinostat/
  benzocaine analogs.
- **2-Oxaspiro[3.3]heptane** and **2,6-diazaspiro[3.3]heptane (DASE)** —
  oxetane- and piperazine-bioisostere spiro fragments common in
  fragment-based drug discovery and kinase-inhibitor scaffolding
  (DASE replaces piperazine in ciprofloxacin-type analogs).
- **Nortropane** (8-azabicyclo[3.2.1]octane) — the parent skeleton of the
  tropane alkaloids: atropine, cocaine, tropicamide.

**Be aware of a specific model weakness this surfaces.** The two
spiro-heptane predictions come out very high (~65 kcal/mol) — noticeably
higher than a naive "two fused cyclobutanes" estimate would suggest. This
is because the model's only spiro training examples (spiropentane,
bicyclobutane) are *three*-membered-ring spiro/fused systems, and the
molecule-level aggregate features generalize from that 3-ring regime to
the 4-ring spiro[3.3]heptane case less reliably than they do for simple
ring-size interpolation. Treat the spiro[3.3]heptane-family predictions as
the least trustworthy numbers in this gallery, and it's a good illustration
of why the `novel_elements_not_in_training_data` / `polycyclic` flags exist
— this case would benefit from a similar "novel topology" flag, which
would be a natural next extension.

## How it works

1. **Parsing.** `smiles.py` parses the SMILES into an atom/bond graph. No
   RDKit dependency is required — see "Extending this platform" below if
   you have RDKit available and want to swap it in.
2. **Ring perception.** `features.py` finds rings via the graph's cyclomatic
   edges (E − N + 1) and BFS shortest paths. This correctly recovers ring
   sizes for the mono-, bi-, and small bridged/fused-cyclic molecules used
   here; it is not a full SSSR/minimum-cycle-basis implementation.
3. **Featurization**, per ring:
   - ring size, heteroatom counts by element, unsaturation/aromaticity
   - a **Baeyer angle-strain proxy**: `ring_size × (planar_polygon_angle −
     ideal_hybridization_angle)²` — the classic 1885 argument that small and
     very large rings are strained because their planar interior angle
     deviates from ~109.5° (sp3) or ~120° (sp2/aromatic)
   - a **Pitzer torsional-strain proxy**: a bell-shaped function peaking
     around 4–5-membered rings, modeling the eclipsing-interaction penalty
     that puckering can't fully relieve in small rings
   - electronegativity/covalent-radius means, exocyclic substitution, etc.
4. **Model selection.** `model.py` evaluates Ridge regression, k-NN,
   Random Forest, and Gradient Boosting via **leave-one-out cross-validation**
   (the appropriate scheme at n≈50) and automatically keeps the model with
   the lowest LOO mean absolute error. Current result on the seed set:
   Gradient Boosting, **LOO-MAE ≈ 4.3 kcal/mol, LOO-R² ≈ 0.79** (n=49 rings,
   up from n=29 in the first version).
5. **Prediction** returns the estimate plus the physics-informed proxies,
   ring/heteroatom breakdown, and flags for polycyclic input or ring sizes
   outside the training range (extrapolation warning).

## Limitations

- **The training set has ~1600 rows.** That's enough to fit smooth trends
  across ring size, hybridization (sp3/sp2/sp), and heteroatom identity,
  but nowhere near enough to learn subtle substituent or stereoelectronic
  effects. Treat outputs as order-of-magnitude estimates, not
  publication-quality strain energies.
- **Polycyclic scoring is still approximate.** The new molecule-level
  aggregate features (ring count, summed proxies across all perceived
  rings) give the model real signal for bridged/fused/cage systems like
  BCP, but this is not a substitute for a proper multi-ring energy
  decomposition — there's no guarantee it generalizes to cage topologies
  very different from norbornane/BCP/cubane/spiropentane/bicyclobutane
  (the only polycyclics in the training set).
- **The main-group heterocycles (silacyclobutane, phosphetane) use a
  single approximate value (~25 kcal/mol) drawn from a general trend
  reported for second-row-element 4-membered rings**, not
  element-specific computed values — real values vary measurably down a
  group (heavier elements → less strain, per that same literature trend).
  Treat these two rows as placeholders demonstrating that the pipeline
  *can* handle non-CHNOS ring atoms, not as trustworthy numbers.
- **The seed dataset itself is a curated illustrative set**, mostly from
  standard heat-of-combustion literature values for monocyclic rings (high
  confidence: 3–4-membered O/N/S heterocycles, C3–C10 cycloalkanes, and the
  G3-level cyclic-alkyne series) and rougher estimates for larger/polycyclic
  rings (flagged `confidence: low` in the CSV). Verify anything you care
  about against NIST WebBook, primary literature, or your own DFT/ab initio
  calculation (e.g. an isodesmic or homodesmotic reaction scheme) before
  using it for real decisions.
- **No 3D geometry.** Everything is inferred from the 2D graph; there's no
  conformer generation, so genuine substituent-driven puckering effects
  (e.g., 1,3-diaxial strain, anomeric effects) aren't captured.

## Extending this platform

- **Add data**: append rows to `ringstrain/data/strain_energies.csv`
  (`name, smiles, ring_class, strain_kcal_mol, confidence, source_note`),
  delete `ringstrain/data/model.pkl`, and re-run — the model retrains
  automatically on next prediction.
- **Swap in RDKit** (if you have network access): in `features.py`,
  replace `parse_smiles`/`perceive_rings` with `Chem.MolFromSmiles` +
  `mol.GetRingInfo().AtomRings()`; this buys correct handling of exotic
  valences, stereochemistry, and larger polycyclic cages, plus access to
  MMFF/UFF-computed strain as an additional feature (compare the
  force-field strain of the real 3D-embedded ring against an open-chain
  reference — often a stronger single feature than any of the 2D proxies
  here).
- **Better ground truth**: for real work, generate strain energies via an
  isodesmic/homodesmotic reaction scheme at a consistent level of theory
  (e.g., G4, CBS-QB3, or DFT with a large basis) rather than relying on
  literature heat-of-combustion compilations, which vary in precision and
  reference state across sources.

## API

`POST /api/predict` `{"smiles": "C1C2CC1C2"}` (bicyclo[1.1.1]pentane) →

```json
{
  "smiles": "C1C2CC1C2",
  "predicted_strain_kcal_mol": 64.87,
  "predicted_strain_kJ_mol": 271.4,
  "primary_ring_size": 4,
  "ring_class": "carbocycle",
  "heteroatoms_in_ring": [],
  "unsaturated_in_ring": false,
  "n_rings_in_molecule": 2,
  "polycyclic": true,
  "has_triple_bond_in_ring": false,
  "heteroatoms_adjacent_in_ring": false,
  "extrapolating_beyond_training_ring_sizes": false,
  "novel_elements_not_in_training_data": [],
  "model_used": "gradient_boosting",
  "model_loo_mae_kcal_mol": 4.27,
  "n_training_examples": 49,
  "baeyer_angle_strain_proxy": 1521.0,
  "pitzer_torsional_strain_proxy": 3.88,
  "sum_baeyer_proxy_all_rings": 3042.0
}
```

(Reported literature value for BCP: ≈66.6 kcal/mol — within the model's LOO-MAE.)

`GET /api/examples` — the seed dataset (for the reference chips in the UI).
`GET /api/model_info` — active model + full LOO-CV comparison table.
