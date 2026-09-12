# Contributing to Cyclops

Thanks for taking a look. This project is intentionally small and
honest about its limits any contributions that keep it that way are
especially welcome.

## Setup

```bash
git clone https://github.com/your-org/strain-bench.git
cd strain-bench
pip install -e ".[dev]"
pytest tests/ -v
```

## Ways to contribute

- **Add a data point.** Append a row to `ringstrain/data/strain_energies.csv`
  (`name, smiles, ring_class, strain_kcal_mol, confidence, source_note`) or
  `ringstrain/data/pharma_scaffolds.csv`. Always include a real citation in
  `source_note` and an honest `confidence` level (`high` = primary
  literature/NIST value, `medium` = derived via a documented method like
  group additivity, `low` = estimated by analogy). Delete
  `ringstrain/data/model.pkl` and run `python -m ringstrain.model` to
  confirm it retrains and check the new LOO-CV numbers.
- **Improve the SMILES parser** (`ringstrain/smiles.py`). It's a minimal,
  dependency-free parser built because this project's dev environment had
  no network access to install RDKit — if you have RDKit available, PRs
  that add it as an optional, auto-detected backend (falling back to the
  built-in parser when RDKit isn't installed) are very welcome.
- **Add a feature** to `ringstrain/features.py`. New ring-level or
  molecule-level descriptors are easy to add — extend `FEATURE_NAMES`
  and the corresponding dict in `featurize_ring`/`featurize_smiles`, then
  add a test in `tests/test_features.py`.
- **Extend the reaction-scheme hierarchy** (`ringstrain/reactions.py`) to
  heteroatom or substituted rings. This is scoped to simple cycloalkanes
  today specifically because that's where the reference-fragment
  construction is unambiguous — extending it correctly to, say, THF or
  substituted cyclohexanes needs a documented, defensible reference-
  fragment scheme, not just a plausible-looking equation.
- **Report a bad prediction.** If you know a ring's real strain energy and
  the model is off by a lot, please open an issue with the SMILES, the
  real value, and a source — that's exactly the feedback this project
  needs given its small training set.

## Ground rules

- No fabricated numbers. Every value in `strain_energies.csv` needs a
  `source_note` a reader could actually go verify, and every ambiguous
  case should be marked `confidence: low` rather than presented as solid.
- Run `pytest tests/ -v` and make sure the dataset-validation step in CI
  (every SMILES parses) passes before opening a PR.
- Keep the README's "Honest limitations" section up to date — if you add
  a capability that has a known failure mode (like the spiro[3.3]heptane
  extrapolation issue noted there), document it in the same PR.
