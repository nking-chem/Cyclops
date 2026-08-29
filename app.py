"""
app.py — local web platform for the ring strain energy predictor.

Run with:
    python app.py
then open http://localhost:5050
"""
from flask import Flask, request, jsonify, send_from_directory
import os

from ringstrain.predict import predict_strain, _ensure_model
from ringstrain.model import DEFAULT_DATA_PATH
from ringstrain.reactions import compare_schemes
import csv
import os as _os

PHARMA_SCAFFOLDS_PATH = _os.path.join(_os.path.dirname(DEFAULT_DATA_PATH), "pharma_scaffolds.csv")

app = Flask(__name__, static_folder="webapp", static_url_path="")

# Warm the model once at startup (trains + caches if not already saved).
_BUNDLE = _ensure_model()


@app.route("/")
def index():
    return send_from_directory("webapp", "index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    smiles = (data.get("smiles") or "").strip()
    if not smiles:
        return jsonify({"error": "No SMILES provided."}), 400
    try:
        result = predict_strain(smiles, bundle=_BUNDLE)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/examples")
def api_examples():
    with open(DEFAULT_DATA_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    out = [
        {
            "name": r["name"],
            "smiles": r["smiles"],
            "ring_class": r["ring_class"],
            "known_strain": float(r["strain_kcal_mol"]),
            "confidence": r.get("confidence", ""),
        }
        for r in rows
    ]
    return jsonify(out)


@app.route("/api/model_info")
def api_model_info():
    return jsonify({
        "model_name": _BUNDLE["model_name"],
        "n_train": _BUNDLE["n_train"],
        "cv_results": _BUNDLE["cv_results"],
    })


@app.route("/api/reaction_schemes", methods=["POST"])
def api_reaction_schemes():
    data = request.get_json(force=True, silent=True) or {}
    smiles = (data.get("smiles") or "").strip()
    max_level = int(data.get("max_level", 4))
    if not smiles:
        return jsonify({"error": "No SMILES provided."}), 400
    try:
        result = compare_schemes(smiles, max_level=max_level)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/pharma_scaffolds")
def api_pharma_scaffolds():
    with open(PHARMA_SCAFFOLDS_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    return jsonify(rows)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
