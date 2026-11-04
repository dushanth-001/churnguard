"""Benchmark single-request inference: the original approach versus faster alternatives.

Run `python src/benchmark.py` to regenerate benchmark.json. Numbers depend on the machine.
"""
import json
import os
import platform
import sys
import time

import joblib
import numpy as np
import xgboost

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from inference import Predictor

MODES = ["legacy", "dataframe_proba", "numpy_booster"]
N_ROWS = 1000
N_CALLS = 1000
TOLERANCE = 1e-5
NUMERIC = ("tenure", "MonthlyCharges", "TotalCharges")


def make_rows(model, n, seed=0):
    """Random but valid-looking inputs, keyed by the API field names."""
    rng = np.random.default_rng(seed)
    names = Predictor(model, "legacy").api_names
    rows = []
    for _ in range(n):
        row = {}
        for name in names:
            if name in NUMERIC:
                row[name] = float(rng.normal(0, 1))
            elif name == "Contract":
                row[name] = int(rng.integers(0, 3))
            elif name == "TenureGroup":
                row[name] = int(rng.integers(0, 4))
            else:
                row[name] = int(rng.integers(0, 2))
        rows.append(row)
    return rows


def time_mode(model, mode, rows, n_calls, warmup=50):
    predictor = Predictor(model, mode)
    for r in rows[:warmup]:
        predictor.predict_one(r)
    times = []
    for i in range(n_calls):
        r = rows[i % len(rows)]
        t0 = time.perf_counter()
        predictor.predict_one(r)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "mean_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
    }


def compare(model, mode, rows):
    base = Predictor(model, "legacy")
    alt = Predictor(model, mode)
    max_diff, mismatches = 0.0, 0
    for r in rows:
        lp, lprob = base.predict_one(r)
        ap, aprob = alt.predict_one(r)
        max_diff = max(max_diff, abs(lprob - aprob))
        mismatches += int(lp != ap)
    return max_diff, mismatches


def choose(results, tolerance=TOLERANCE):
    """Fastest mode that gives the same predictions as legacy and is not slower than it."""
    legacy = results.get("legacy", {})
    eligible = {
        m: r for m, r in results.items()
        if m != "legacy" and "error" not in r
        and r["label_mismatches"] == 0 and r["max_abs_prob_diff"] <= tolerance
        and ("p50_ms" not in legacy or r["p50_ms"] <= legacy["p50_ms"])
    }
    if not eligible:
        return "legacy"
    return min(eligible, key=lambda m: eligible[m]["p50_ms"])


def main():
    t0 = time.perf_counter()
    model = joblib.load(os.path.join(BASE_DIR, "models", "xgboost_model.pkl"))
    load_ms = (time.perf_counter() - t0) * 1000

    rows = make_rows(model, N_ROWS)
    results = {}
    for mode in MODES:
        try:
            stats = time_mode(model, mode, rows, N_CALLS)
            if mode != "legacy":
                diff, mism = compare(model, mode, rows)
                stats["max_abs_prob_diff"] = float(f"{diff:.3g}")
                stats["label_mismatches"] = int(mism)
            results[mode] = stats
        except Exception as e:
            results[mode] = {"error": str(e)[:200]}

    report = {
        "python": platform.python_version(),
        "xgboost": xgboost.__version__,
        "n_calls": N_CALLS,
        "n_rows_compared": N_ROWS,
        "model_load_ms": round(load_ms, 1),
        "modes": results,
        "chosen": choose(results),
    }
    with open(os.path.join(BASE_DIR, "benchmark.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
