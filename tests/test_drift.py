import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from drift import detect_drift, load_prediction_logs, psi


def _frame(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "tenure": rng.normal(0, 1, n),
        "MonthlyCharges": rng.normal(0, 1, n) + shift,
        "TotalCharges": rng.normal(0, 1, n),
        "Contract": rng.choice([0, 1, 2], size=n, p=[0.5, 0.2, 0.3]),
        "probability": rng.uniform(0, 1, n),
    })


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 3000), rng.normal(0, 1, 500)) < 0.1


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 3000), rng.normal(1.5, 1, 500)) > 0.25


def test_detect_drift_ok_when_stable():
    report = detect_drift(_frame(3000, seed=1), _frame(500, seed=2))
    assert report["status"] == "ok"
    assert report["drifted_features"] == []


def test_detect_drift_flags_shifted_feature():
    report = detect_drift(_frame(3000, seed=1), _frame(500, seed=2, shift=1.5))
    assert report["status"] == "drift_detected"
    assert "MonthlyCharges" in report["drifted_features"]
    assert "tenure" not in report["drifted_features"]


def test_insufficient_data():
    report = detect_drift(_frame(3000, seed=1), _frame(20, seed=2))
    assert report["status"] == "insufficient_data"


def test_load_prediction_logs_parses_json_lines(tmp_path):
    rec = {"event": "prediction", "prediction": 1, "probability": 0.8, "latency_ms": 5.0,
           "tenure": -1.0, "MonthlyCharges": 1.0, "TotalCharges": -0.5, "Contract": 0}
    err = {"event": "prediction_error", "error": "boom"}
    path = tmp_path / "app.log"
    path.write_text(
        "2026-10-29 10:00:00,000 - INFO - " + json.dumps(rec) + "\n"
        "2026-10-29 10:00:01,000 - ERROR - " + json.dumps(err) + "\n"
        "not a json line\n"
    )
    df = load_prediction_logs(str(path))
    assert len(df) == 1
    assert df["probability"].iloc[0] == 0.8


def test_reference_file_is_valid():
    ref = pd.read_csv(os.path.join(ROOT, "models", "reference.csv"))
    assert list(ref.columns) == ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "probability"]
    assert len(ref) > 500
    assert not ref.isna().any().any()
