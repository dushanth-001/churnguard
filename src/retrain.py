"""Retrain a candidate model and promote it only if it passes a gate against the current model.

The candidate and the current (champion) model are scored on the same held-out test split,
so the comparison is fair even when the input data has changed.
"""
import argparse
import datetime
import json
import os
import shutil
import sys

import joblib
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from data_preprocessing import load_data
from evaluate import DATA_URL, SCALE_COLS, build_features

BEST_PARAMS = {'subsample': 0.8, 'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.01}
MONITORED_INPUTS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract']
MAX_DROP = 0.02
NO_CHANGE = 0.001
MIN_TEST_ROWS = 200


def compute_metrics(y_true, proba, threshold=0.5):
    """Classification metrics at a fixed decision threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'recall': round(float(recall_score(y_true, pred, zero_division=0)), 4),
        'precision': round(float(precision_score(y_true, pred, zero_division=0)), 4),
        'f1': round(float(f1_score(y_true, pred, zero_division=0)), 4),
        'accuracy': round(float(accuracy_score(y_true, pred)), 4),
        'roc_auc': round(float(roc_auc_score(y_true, proba)), 4),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }


def decide(candidate, champion, max_drop=MAX_DROP, no_change=NO_CHANGE):
    """Promotion gate. Compares recall and ROC AUC of the candidate with the current model."""
    reasons = []
    for metric in ('recall', 'roc_auc'):
        drop = round(champion[metric] - candidate[metric], 4)
        if drop > max_drop + 1e-9:
            reasons.append(f'{metric} dropped by {drop} (limit {max_drop})')
    if reasons:
        return {'decision': 'rejected', 'promote': False, 'reasons': reasons}
    if all(abs(candidate[m] - champion[m]) <= no_change + 1e-9 for m in ('recall', 'roc_auc')):
        return {'decision': 'no_change', 'promote': False,
                'reasons': ['candidate is equivalent to the current model']}
    return {'decision': 'promote', 'promote': True, 'reasons': []}


def schema_matches(candidate_cols, champion_cols):
    """The API sends a fixed set of features, so the new model must use exactly the same ones."""
    return set(candidate_cols) == set(champion_cols)


def prepare(X_raw, model, scaler):
    """Scale the numeric columns and order the columns the way the model expects."""
    X = X_raw.copy()
    X[SCALE_COLS] = scaler.transform(X[SCALE_COLS])
    return X.reindex(columns=list(model.feature_names_in_), fill_value=0)


def score(model, scaler, X_raw, y):
    proba = model.predict_proba(prepare(X_raw, model, scaler))[:, 1]
    return compute_metrics(y, proba)


def promote_candidate(out_dir, models_dir, metrics_path):
    """Copy an approved candidate into models/ and refresh metrics.json."""
    for name in ('xgboost_model.pkl', 'scaler.pkl', 'reference.csv'):
        shutil.copy2(os.path.join(out_dir, name), os.path.join(models_dir, name))
    shutil.copy2(os.path.join(out_dir, 'metrics.json'), metrics_path)


def write_summary(result, out_dir):
    lines = ['## Retraining result', '',
             f"**Decision: {result['decision']}** (promote: {result['promote']})", '']
    for reason in result['reasons']:
        lines.append(f'- {reason}')
    if result.get('candidate') and result.get('champion'):
        lines += ['', '| Metric | Current model | Candidate |', '|---|---|---|']
        for m in ('recall', 'precision', 'f1', 'accuracy', 'roc_auc'):
            lines.append(f"| {m} | {result['champion'][m]} | {result['candidate'][m]} |")
    with open(os.path.join(out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Retrain a candidate model and apply the promotion gate.')
    parser.add_argument('--data', default=DATA_URL, help='CSV with the same columns as the training data')
    parser.add_argument('--out', default=os.path.join(BASE_DIR, 'candidate'))
    parser.add_argument('--promote', action='store_true',
                        help='copy an approved candidate into models/ and metrics.json')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    models_dir = os.path.join(BASE_DIR, 'models')
    champion = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))
    champion_scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))

    df = build_features(load_data(args.data))
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    today = datetime.date.today().isoformat()
    cand = champ = None

    if not schema_matches(X.columns, champion.feature_names_in_):
        decision = {'decision': 'rejected', 'promote': False,
                    'reasons': ['feature schema differs from the current model']}
    elif len(y_test) < MIN_TEST_ROWS:
        decision = {'decision': 'rejected', 'promote': False,
                    'reasons': [f'test set too small ({len(y_test)} rows, need {MIN_TEST_ROWS})']}
    else:
        scaler = StandardScaler()
        X_train_s = X_train.copy()
        X_train_s[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
        scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = XGBClassifier(scale_pos_weight=scale_weight, eval_metric='logloss',
                              random_state=42, **BEST_PARAMS)
        model.fit(X_train_s, y_train)

        cand = score(model, scaler, X_test, y_test)
        champ = score(champion, champion_scaler, X_test, y_test)
        decision = decide(cand, champ)

        X_ref = prepare(X_val, model, scaler)
        reference = X_ref[MONITORED_INPUTS].copy()
        reference['probability'] = model.predict_proba(X_ref)[:, 1]
        reference.to_csv(os.path.join(args.out, 'reference.csv'), index=False)
        joblib.dump(model, os.path.join(args.out, 'xgboost_model.pkl'))
        joblib.dump(scaler, os.path.join(args.out, 'scaler.pkl'))
        dataset = (f'Telco Customer Churn (IBM), {len(df)} customers'
                   if args.data == DATA_URL else f'{args.data}, {len(df)} rows')
        with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
            json.dump({'evaluated_on': today, 'dataset': dataset,
                       'split': '70/15/15 stratified, random_state=42',
                       'test_set_size': int(len(y_test)), 'threshold': 0.5, **cand}, f, indent=2)

    result = {'date': today, 'data': args.data, 'candidate': cand, 'champion': champ, **decision}
    with open(os.path.join(args.out, 'decision.json'), 'w') as f:
        json.dump(result, f, indent=2)
    write_summary(result, args.out)

    if decision['promote'] and args.promote:
        promote_candidate(args.out, models_dir, os.path.join(BASE_DIR, 'metrics.json'))
        print('Candidate promoted into models/ and metrics.json')
    print(json.dumps({k: result[k] for k in ('decision', 'promote', 'reasons')}, indent=2))


if __name__ == '__main__':
    main()
