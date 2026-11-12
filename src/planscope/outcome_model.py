'''
Train the outcome model and export it as plain numbers to models/outcome.json.
Logistic regression on TF-IDF text + one-hot authority, type and authority x type, then
calibrated on the most recent decided year so its risks match today's refusal rate.
'''

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'data' / 'processed' / 'outcome.parquet'
OUT_PATH = ROOT / 'models' / 'outcome.json'

GROUPS = ['lpa_name', 'category', 'lpa_category']
C = 0.5  # chosen on validation PR-AUC in notebooks/03_outcome_model.ipynb


def build_model():
    '''TF-IDF words and word pairs + one-hot groups, into one logistic regression.'''
    features = make_column_transformer(
        (TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True), 'text'),
        (OneHotEncoder(handle_unknown='ignore', min_frequency=20), GROUPS),
    )
    return make_pipeline(features, LogisticRegression(C=C, max_iter=2000))


def fit(train, calib):
    '''Fit on train, then Platt-calibrate on calib: risk = sigmoid(a * score + b).'''
    model = build_model().fit(train, train['refused'])
    scores = model.decision_function(calib).reshape(-1, 1)
    platt = LogisticRegression(C=1e6).fit(scores, calib['refused'])  # huge C = effectively no penalty
    return model, platt.coef_[0, 0], platt.intercept_[0]


def calibrated_risk(model, a, b, df):
    '''Calibrated refusal risk for every row of df.'''
    return 1 / (1 + np.exp(-(a * model.decision_function(df) + b)))


def export(model, a, b, calib):
    '''Every weight as plain numbers, with the calibration folded in, plus each group's recent refusal rate.'''
    text = model[0].named_transformers_['tfidfvectorizer']
    onehot = model[0].named_transformers_['onehotencoder']
    weights = a * model[-1].coef_[0]
    n_terms = len(text.idf_)

    named = dict(zip(onehot.get_feature_names_out(), weights[n_terms:]))
    groups = {}
    for col, values, rare in zip(GROUPS, onehot.categories_, onehot.infrequent_categories_):
        rare = set() if rare is None else set(rare)
        groups[col] = {v: round(named[f'{col}_infrequent_sklearn' if v in rare else f'{col}_{v}'], 6) for v in values}

    recent = calib.groupby('lpa_category')['refused'].agg(['mean', 'size'])
    return {
        'intercept': a * model[-1].intercept_[0] + b,
        'terms': text.get_feature_names_out().tolist(),
        'idf': text.idf_.round(6).tolist(),
        'weights': weights[:n_terms].round(6).tolist(),
        'groups': groups,
        'averages': {key: [round(rate, 4), int(n)] for key, (rate, n) in recent.iterrows()},
    }


def main():
    # Same recipe as the notebook, one year later: fit on train + val, calibrate on the latest decided year
    df = pd.read_parquet(DATA_PATH)
    train, calib = df[df['split'].isin(['train', 'val'])], df[df['split'] == 'test']
    model, a, b = fit(train, calib)

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(export(model, a, b, calib), separators=(',', ':')))
    risk = calibrated_risk(model, a, b, calib)
    print(f'Saved {OUT_PATH.name}: {len(model[0].named_transformers_["tfidfvectorizer"].idf_):,} terms, '
          f'{OUT_PATH.stat().st_size / 1e6:.1f} MB')
    print(f'Calibration a={a:.3f} b={b:.3f}; mean risk {risk.mean():.3f} vs refusal rate {calib["refused"].mean():.3f}')


if __name__ == '__main__':
    main()