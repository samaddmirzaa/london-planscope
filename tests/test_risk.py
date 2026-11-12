import json

import numpy as np
import pandas as pd
import pytest

from planscope.outcome_data import strip_post_submission
from planscope.outcome_model import calibrated_risk, export, fit
from planscope.risk import RiskModel, prepare


def synthetic(n, seed):
    '''Applications whose refusal depends on their words, borough and type.'''
    rng = np.random.default_rng(seed)
    phrases = ['rear extension', 'roof terrace', 'basement', 'loft conversion', 'fell tree', 'new shopfront']
    rows = []
    for _ in range(n):
        words = rng.choice(phrases, size=2, replace=False)
        borough, category = rng.choice(['Camden', 'Brent', 'Barnet']), rng.choice(['householder', 'full_planning'])
        risky = ('roof terrace' in words) + ('basement' in words) + (borough == 'Camden')
        rows.append({'text': ' and '.join(words), 'lpa_name': borough, 'category': category,
                     'lpa_category': f'{borough} | {category}', 'refused': int(rng.random() < 0.15 + 0.25 * risky)})
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    '''A scikit-learn model fitted on synthetic data, and the same model exported and reloaded.'''
    calib = synthetic(300, 1)
    model, a, b = fit(synthetic(600, 0), calib)
    path = tmp_path_factory.mktemp('model') / 'outcome.json'
    path.write_text(json.dumps(export(model, a, b, calib)))
    return model, a, b, RiskModel(path)


def test_exported_model_matches_scikit_learn(trained):
    model, a, b, served = trained

    cases = pd.DataFrame({
        'text': ['Roof terrace and basement', 'rear extension and rear extension and basement',
                 'Fell tree. Linked cases: roof terrace, basement', 'something never seen', ''],
        'lpa_name': ['Camden', 'Brent', 'Barnet', 'Camden', 'Croydon'],
        'category': ['householder', 'full_planning', 'householder', 'full_planning', 'householder'],
    })
    cases['lpa_category'] = cases['lpa_name'] + ' | ' + cases['category']
    expected = calibrated_risk(model, a, b, cases.assign(text=cases['text'].map(prepare)))
    got = [served.predict(t, lpa, cat)['risk'] for t, lpa, cat in cases[['text', 'lpa_name', 'category']].values]
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_prediction_explains_itself(trained):
    result = trained[3].predict('Roof terrace and rear extension, xyzzy', 'Camden', 'householder')
    assert 'roof terrace' in result['raising']
    assert (result['words_known'], result['words_total']) == (5, 6)
    assert result['average'][1] > 0


def test_serving_text_prep_matches_training():
    text = 'Loft conversion (AMENDED DESCRIPTION). Linked cases: APP/U5360/W/22/3313698'
    assert prepare(text) == strip_post_submission(pd.Series([text]))[0].lower() == 'loft conversion .'