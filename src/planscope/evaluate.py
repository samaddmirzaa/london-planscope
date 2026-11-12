'''
Shared evaluation with 95% bootstrap confidence intervals:
accuracy and macro-F1 for the category classifier, ROC-AUC, PR-AUC and Brier score for the outcome model.
'''

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, roc_auc_score


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


def report(metrics, y_true, y_pred):
    '''Each (name, metric) pair with its confidence interval, as a small table.'''
    rows = {name: bootstrap(metric, y_true, y_pred) for name, metric in metrics}
    return pd.DataFrame(rows, index=['score', 'ci_low', 'ci_high']).T.round(4)


def evaluate(y_true, y_pred):
    '''Category classifier: accuracy and macro-F1.'''
    return report([('accuracy', accuracy_score), ('macro_f1', macro_f1)], y_true, y_pred)


def evaluate_risk(y_true, risk):
    '''Outcome model: ranking (ROC-AUC, PR-AUC) and probability quality (Brier score) of refusal risks.'''
    return report([('roc_auc', roc_auc_score), ('pr_auc', average_precision_score), ('brier', brier_score_loss)], y_true, risk)