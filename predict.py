"""
predict.py
----------
Public prediction API + CLI.

    python -m ringstrain.predict "C1CC1"
    python -m ringstrain.predict "C1CO1" "C1CCCCC1" --json
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np

from .features import featurize_smiles, FEATURE_NAMES, TRAINED_ELEMENTS
from .model import load_model, DEFAULT_MODEL_PATH, train_best_model, save_model

# Ring sizes present in the training data. Predictions for ring sizes far
# outside this range are extrapolation and are flagged as low-confidence.
TRAIN_RING_SIZE_RANGE = (3, 17)


def _ensure_model():
    if not os.path.exists(DEFAULT_MODEL_PATH):
        bundle = train_best_model(verbose=False)
        save_model(bundle)
    return load_model()


def predict_strain(smiles: str, bundle=None) -> dict:
    """Predict ring strain energy (kcal/mol) for the smallest/primary ring
    in `smiles`. Returns a dict with the prediction plus supporting detail.
    """
    if bundle is None:
        bundle = _ensure_model()

    primary, all_rings_feats, mol, rings = featurize_smiles(smiles)
    x = np.array([[primary[name] for name in FEATURE_NAMES]])
    pred = float(bundle["model"].predict(x)[0])

    ring_size = primary["ring_size"]
    lo, hi = TRAIN_RING_SIZE_RANGE
    extrapolating = not (lo <= ring_size <= hi)
    n_rings_in_molecule = len(rings)

    ring_atom_elements = set()
    for r in rings:
        for i in r:
            ring_atom_elements.add(mol.atoms[i].symbol[0].upper() + mol.atoms[i].symbol[1:].lower())
    novel_elements = sorted(ring_atom_elements - TRAINED_ELEMENTS)

    heteroatoms = []
    if primary["n_hetero_O"]:
        heteroatoms += ["O"] * primary["n_hetero_O"]
    if primary["n_hetero_N"]:
        heteroatoms += ["N"] * primary["n_hetero_N"]
    if primary["n_hetero_S"]:
        heteroatoms += ["S"] * primary["n_hetero_S"]
    if primary["n_hetero_other"]:
        heteroatoms += ["X"] * primary["n_hetero_other"]

    return {
        "smiles": smiles,
        "predicted_strain_kcal_mol": round(pred, 2),
        "predicted_strain_kJ_mol": round(pred * 4.184, 2),
        "primary_ring_size": ring_size,
        "ring_class": "heterocycle" if heteroatoms else "carbocycle",
        "heteroatoms_in_ring": heteroatoms,
        "unsaturated_in_ring": bool(
            primary["n_double_in_ring"] or primary["n_triple_in_ring"] or primary["is_aromatic"]
        ),
        "n_rings_in_molecule": n_rings_in_molecule,
        "polycyclic": n_rings_in_molecule > 1,
        "has_triple_bond_in_ring": bool(primary["has_triple_bond"]),
        "heteroatoms_adjacent_in_ring": bool(primary["hetero_adjacent"]),
        "extrapolating_beyond_training_ring_sizes": extrapolating,
        "novel_elements_not_in_training_data": novel_elements,
        "model_used": bundle["model_name"],
        "model_loo_mae_kcal_mol": round(bundle["cv_results"][bundle["model_name"]]["loo_mae"], 2),
        "n_training_examples": bundle["n_train"],
        "baeyer_angle_strain_proxy": round(primary["baeyer_strain_proxy"], 1),
        "pitzer_torsional_strain_proxy": round(primary["pitzer_strain_proxy"], 2),
        "sum_baeyer_proxy_all_rings": round(primary["sum_baeyer_proxy_all_rings"], 1),
    }


def predict_many(smiles_list: list[str]) -> list[dict]:
    bundle = _ensure_model()
    out = []
    for smi in smiles_list:
        try:
            out.append(predict_strain(smi, bundle=bundle))
        except Exception as e:
            out.append({"smiles": smi, "error": str(e)})
    return out


def _main():
    ap = argparse.ArgumentParser(description="Predict ring strain energy from SMILES.")
    ap.add_argument("smiles", nargs="+", help="One or more SMILES strings")
    ap.add_argument("--json", action="store_true", help="Print raw JSON")
    args = ap.parse_args()

    results = predict_many(args.smiles)
    if args.json:
        print(json.dumps(results, indent=2))
        return

    for r in results:
        if "error" in r:
            print(f"\n{r['smiles']}: ERROR - {r['error']}")
            continue
        flags = []
        if r["extrapolating_beyond_training_ring_sizes"]:
            flags.append("EXTRAPOLATION: ring size outside training range")
        if r["novel_elements_not_in_training_data"]:
            flags.append(f"EXTRAPOLATION: ring element(s) {r['novel_elements_not_in_training_data']} not in training data")
        if r["polycyclic"]:
            flags.append("polycyclic: primary-ring features augmented with whole-molecule ring-count/proxy sums, but still approximate")
        if r["has_triple_bond_in_ring"]:
            flags.append("contains a ring alkyne (sp center) — high-strain cyclic-alkyne regime")
        print(f"\n{r['smiles']}  ({r['ring_class']}, {r['primary_ring_size']}-membered primary ring)")
        print(f"  Predicted strain energy: {r['predicted_strain_kcal_mol']:.2f} kcal/mol "
              f"({r['predicted_strain_kJ_mol']:.1f} kJ/mol)")
        print(f"  Model: {r['model_used']}  (LOO-CV MAE ~{r['model_loo_mae_kcal_mol']:.2f} kcal/mol, "
              f"n={r['n_training_examples']} training rings)")
        print(f"  Baeyer angle-strain proxy: {r['baeyer_angle_strain_proxy']}   "
              f"Pitzer torsional-strain proxy: {r['pitzer_torsional_strain_proxy']}")
        for f in flags:
            print(f"  ! {f}")


if __name__ == "__main__":
    _main()
