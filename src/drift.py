"""Simple data drift detection: compare logged production inputs with a reference sample."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONITORED = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "probability"]
PSI_MODERATE = 0.1
PSI_MAJOR = 0.25


def psi(reference, current, bins=10, eps=1e-4):
    """Population Stability Index between two samples."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if len(np.unique(ref)) <= 10:
        cats = np.union1d(np.unique(ref), np.unique(cur))
        ref_pct = np.array([(ref == c).mean() for c in cats])
        cur_pct = np.array([(cur == c).mean() for c in cats])
    else:
        inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1])
        ref_counts = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=len(inner) + 1)
        cur_counts = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=len(inner) + 1)
        ref_pct = ref_counts / len(ref)
        cur_pct = cur_counts / len(cur)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(reference, current, columns=MONITORED, min_rows=100):
    """Compare current data with the reference sample and report drift per feature."""
    if len(current) < min_rows:
        return {"status": "insufficient_data", "rows": int(len(current)), "min_rows": min_rows}
    features = {}
    for col in columns:
        if col not in reference.columns or col not in current.columns:
            continue
        ref = reference[col].dropna()
        cur = current[col].dropna()
        value = psi(ref, cur)
        ks = ks_2samp(ref, cur)
        level = "major" if value >= PSI_MAJOR else "moderate" if value >= PSI_MODERATE else "stable"
        features[col] = {"psi": round(value, 4), "ks_pvalue": round(float(ks.pvalue), 6), "level": level}
    if not features:
        raise ValueError("No monitored columns found in both datasets")
    drifted = [c for c, v in features.items() if v["level"] == "major"]
    return {
        "status": "drift_detected" if drifted else "ok",
        "rows": int(len(current)),
        "drifted_features": drifted,
        "features": features,
    }


def load_prediction_logs(path):
    """Read prediction events (one JSON object per log line) into a DataFrame."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            start = line.find("{")
            if start == -1:
                continue
            try:
                rec = json.loads(line[start:])
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "prediction":
                rows.append(rec)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Check logged predictions for data drift.")
    parser.add_argument("--reference", default=os.path.join(BASE_DIR, "models", "reference.csv"))
    parser.add_argument("--logs", required=True, help="exported log file with JSON prediction lines")
    parser.add_argument("--min-rows", type=int, default=100)
    args = parser.parse_args()

    report = detect_drift(pd.read_csv(args.reference), load_prediction_logs(args.logs),
                          min_rows=args.min_rows)
    print(json.dumps(report, indent=2))
    sys.exit(1 if report["status"] == "drift_detected" else 0)


if __name__ == "__main__":
    main()
