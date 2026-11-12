import json

import numpy as np
import pandas as pd
import pytest

from planscope.risk import RiskModel
from planscope.similar import SimilarSearch, build

TERMS = ['convert', 'flats', 'house', 'into', 'fell', 'tree',
         'convert house', 'house into', 'into flats', 'flats into', 'into house']

# Past decisions: B is listed before A, so the ranking cannot come from the row order
APPS = pd.DataFrame({
    'lpa_name': ['Camden', 'Camden', 'Camden', 'Brent', 'Camden'],
    'category': ['householder', 'householder', 'full_planning', 'householder', 'householder'],
    'lpa_app_no': ['B', 'A', 'C', 'D', 'E'],
    'valid_date': pd.to_datetime(['2023-01-15', '2024-03-01', '2024-01-01', '2024-01-01', '2022-06-30']),
    'outcome': ['APPROVED', 'REFUSED', 'APPROVED', 'APPROVED', 'APPROVED'],
    'description': ['Convert flats into house, ' + 'x' * 400, 'Convert house into flats', 'Convert house into flats',
                    'Convert house into flats', 'Fell tree'],
    'url_planning_app': ['/planning/B', 'https://council.gov.uk/A', None, None, None],
})


def risk_model(path, terms=TERMS):
    '''A minimal outcome model: only its vocabulary and idf matter to the search.'''
    path.write_text(json.dumps({'terms': terms, 'idf': [1.0] * len(terms), 'weights': [0.0] * len(terms),
                                'intercept': 0.0, 'groups': {}, 'averages': {}}))
    return RiskModel(path)


@pytest.fixture
def search(tmp_path):
    model = risk_model(tmp_path / 'outcome.json')
    np.savez(tmp_path / 'similar.npz', **build(APPS, model))
    return SimilarSearch(tmp_path / 'similar.npz', model)


def test_search_stays_in_borough_and_type_and_respects_word_order(search):
    found = search.search('Convert my house into flats', 'Camden', 'householder')
    assert [f['reference'] for f in found] == ['A', 'B']  # C: other type, D: other borough, E: no shared words
    assert found[0] == {'reference': 'A', 'date': '2024-03', 'outcome': 'REFUSED', 'description': 'Convert house into flats',
                        'url': 'https://council.gov.uk/A', 'similarity': pytest.approx(0.926, abs=0.002)}
    assert found[1]['similarity'] == pytest.approx(0.667, abs=0.002)  # same words, opposite direction


def test_records_are_trimmed_for_display(search):
    shown = search.search('Convert flats into house', 'Camden', 'householder')[0]
    assert (len(shown['description']), shown['description'][-1], shown['url']) == (300, '…', None)


def test_unknown_group_or_no_shared_words_finds_nothing(search):
    assert search.search('Convert house into flats', 'Camden', 'trees') == []
    assert search.search('Fell tree', 'Brent', 'householder') == []


def test_index_built_for_another_vocabulary_is_refused(tmp_path):
    np.savez(tmp_path / 'similar.npz', **build(APPS, risk_model(tmp_path / 'a.json')))
    with pytest.raises(ValueError, match='rebuild'):
        SimilarSearch(tmp_path / 'similar.npz', risk_model(tmp_path / 'b.json', TERMS[:-1]))