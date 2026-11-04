import os
import sys

import joblib

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from benchmark import make_rows
from inference import MODE, Predictor

MODEL = joblib.load(os.path.join(ROOT, "models", "xgboost_model.pkl"))
ROWS = make_rows(MODEL, 300, seed=7)


def _agree(mode):
    base = Predictor(MODEL, "legacy")
    alt = Predictor(MODEL, mode)
    for r in ROWS:
        lp, lprob = base.predict_one(r)
        ap, aprob = alt.predict_one(r)
        assert lp == ap
        assert abs(lprob - aprob) < 1e-5


def test_dataframe_proba_matches_legacy():
    _agree("dataframe_proba")


def test_selected_mode_matches_legacy():
    _agree(MODE)
