"""
model.py
--------
Trains a ring-strain-energy regressor on the curated seed dataset.

Honesty note: the seed dataset has ~30 rows. That is nowhere near enough for
a deep model to learn real chemistry from scratch, so this module:

  1. Evaluates several small, low-variance model families with
     leave-one-out cross-validation (the only sane CV scheme at n=30) and
     reports honest, unbiased error estimates for each.
  2. Picks the best-performing family automatically.
  3. Is designed so that adding more rows to data/strain_energies.csv (e.g.
     computed via DFT/G4/isodesmic-reaction schemes, or pulled from NIST
     WebBook heats of formation) immediately improves it — no code changes
     needed.

This is a *starter platform*, not a validated predictive tool for
publication-quality strain energies.
"""
from __future__ import annotations
import csv
import json
import os
import pickle

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score

from .features import featurize_smiles, FEATURE_NAMES

HERE = os.path.dirname(__file__)
DEFAULT_DATA_PATH = os.path.join(HERE, "data", "strain_energies.csv")
DEFAULT_MODEL_PATH = os.path.join(HERE, "data", "model.pkl")

CANDIDATES = {
    "ridge": Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=3.0))]),
    "knn_k3": Pipeline([("scale", StandardScaler()),
                         ("reg", KNeighborsRegressor(n_neighbors=3, weights="distance"))]),
    "random_forest": RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=0
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0
    ),
}


def load_dataset(path: str = DEFAULT_DATA_PATH):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    X, y, meta = [], [], []
    for r in rows:
        primary, _all, _mol, _rings = featurize_smiles(r["smiles"])
        X.append([primary[name] for name in FEATURE_NAMES])
        y.append(float(r["strain_kcal_mol"]))
        meta.append(r)
    return np.array(X), np.array(y), meta


def evaluate_candidates(X, y):
    """Leave-one-out CV for every candidate model; returns a results dict."""
    loo = LeaveOneOut()
    results = {}
    for name, estimator in CANDIDATES.items():
        preds = np.zeros_like(y)
        for train_idx, test_idx in loo.split(X):
            model = _clone(estimator)
            model.fit(X[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X[test_idx])
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        results[name] = {"loo_mae": mae, "loo_rmse": rmse, "loo_r2": r2}
    return results


def _clone(estimator):
    from sklearn.base import clone
    return clone(estimator)


def train_best_model(data_path: str = DEFAULT_DATA_PATH, verbose: bool = True):
    X, y, meta = load_dataset(data_path)
    results = evaluate_candidates(X, y)
    best_name = min(results, key=lambda k: results[k]["loo_mae"])
    best_estimator = _clone(CANDIDATES[best_name])
    best_estimator.fit(X, y)

    if verbose:
        print(f"{'model':<18}{'LOO MAE':>10}{'LOO RMSE':>10}{'LOO R2':>10}")
        for name, r in sorted(results.items(), key=lambda kv: kv[1]["loo_mae"]):
            marker = "  <- selected" if name == best_name else ""
            print(f"{name:<18}{r['loo_mae']:>10.2f}{r['loo_rmse']:>10.2f}{r['loo_r2']:>10.2f}{marker}")

    bundle = {
        "model_name": best_name,
        "model": best_estimator,
        "feature_names": FEATURE_NAMES,
        "cv_results": results,
        "n_train": len(y),
    }
    return bundle


def save_model(bundle: dict, path: str = DEFAULT_MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_model(path: str = DEFAULT_MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    """Console-script entry point (`strain-bench-train`)."""
    bundle = train_best_model()
    save_model(bundle)
    print(f"\nSaved model ({bundle['model_name']}, trained on {bundle['n_train']} rings) -> {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
