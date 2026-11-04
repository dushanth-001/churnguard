import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from retrain import compute_metrics, decide, promote_candidate, schema_matches


def _m(recall, auc):
    return {"recall": recall, "roc_auc": auc}


def test_rejects_when_recall_drops_too_much():
    r = decide(_m(0.70, 0.84), _m(0.79, 0.84))
    assert r["decision"] == "rejected" and r["promote"] is False


def test_rejects_when_auc_drops_too_much():
    r = decide(_m(0.79, 0.80), _m(0.79, 0.84))
    assert r["decision"] == "rejected"


def test_no_change_when_equivalent():
    r = decide(_m(0.7895, 0.8405), _m(0.79, 0.84))
    assert r["decision"] == "no_change" and r["promote"] is False


def test_promotes_improvement():
    r = decide(_m(0.82, 0.85), _m(0.79, 0.84))
    assert r["decision"] == "promote" and r["promote"] is True


def test_promotes_small_drop_within_tolerance():
    r = decide(_m(0.78, 0.835), _m(0.79, 0.84))
    assert r["promote"] is True


def test_compute_metrics_perfect_model():
    m = compute_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["recall"] == 1.0 and m["roc_auc"] == 1.0
    assert m["confusion_matrix"]["tp"] == 2 and m["confusion_matrix"]["fn"] == 0


def test_schema_matches_ignores_order_but_not_missing_columns():
    assert schema_matches(["a", "b", "c"], ["c", "a", "b"])
    assert not schema_matches(["a", "b"], ["a", "b", "c"])


def test_promote_candidate_copies_files(tmp_path):
    out, models = tmp_path / "out", tmp_path / "models"
    out.mkdir()
    models.mkdir()
    for name in ("xgboost_model.pkl", "scaler.pkl", "reference.csv", "metrics.json"):
        (out / name).write_text("new " + name)
        (models / name).write_text("old")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("old")
    promote_candidate(str(out), str(models), str(metrics_path))
    assert (models / "xgboost_model.pkl").read_text() == "new xgboost_model.pkl"
    assert (models / "reference.csv").read_text() == "new reference.csv"
    assert metrics_path.read_text() == "new metrics.json"
