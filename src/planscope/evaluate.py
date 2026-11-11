'''
Shared evaluation for the category classifiers: accuracy and macro-F1,
each with a 95% bootstrap confidence interval.
'''

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def macro_f1(y_true, y_pred):
    '''F1 per category present in y_true, averaged with equal weight.'''
    return f1_score(y_true, y_pred, labels=np.unique(y_true), average='macro')


def bootstrap(metric, y_true, y_pred, n_boot=1000, seed=42):
    '''The metric on the full set and its 95% confidence interval from resampling rows.'''
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(y_true), len(y_true))
        scores.append(metric(y_true[rows], y_pred[rows]))
    low, high = np.percentile(scores, [2.5, 97.5])
    return metric(y_true, y_pred), low, high


def evaluate(y_true, y_pred):
    '''Accuracy and macro-F1 with confidence intervals, as a small table.'''
    rows = {name: bootstrap(metric, y_true, y_pred) for name, metric in [('accuracy', accuracy_score), ('macro_f1', macro_f1)]}
    return pd.DataFrame(rows, index=['score', 'ci_low', 'ci_high']).T.round(4)