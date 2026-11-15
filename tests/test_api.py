import numpy as np
from fastapi.testclient import TestClient

from planscope.api import app, get_classifier, get_risk_model, get_similar


class FakeClassifier:
    '''Stands in for the real model, so the API is tested without model files.'''
    labels = ['householder', 'trees']

    def logits(self, texts):
        # Softmax gives exactly 0.2 and 0.8; adding 10 to both keeps those probabilities but makes the energy 10,
        # a strong response like a real planning description's
        return np.log(np.array([[0.2, 0.8]])) + 10


class FakeRiskModel:
    '''Stands in for the outcome model; remembers the category it was asked about.'''
    groups = {'lpa_name': {'Camden': 0.1}}
    index = {'prune', 'oak', 'tree', 'roof', 'terrace', 'fell', 'one'}

    def predict(self, description, borough, category):
        self.category = category
        return {'risk': 0.123456, 'average': [0.1, 441], 'raising': ['roof terrace'], 'lowering': ['rear'],
                'words_known': 5, 'words_total': 6}


class FakeSimilar:
    '''Stands in for the similar-applications index.'''

    def search(self, description, borough, category):
        return [{'reference': '2024/0001/P', 'date': '2024-03', 'outcome': 'REFUSED', 'description': 'Roof terrace',
                 'url': None, 'similarity': 0.9}]


risk_model = FakeRiskModel()
app.dependency_overrides[get_classifier] = FakeClassifier
app.dependency_overrides[get_risk_model] = lambda: risk_model
app.dependency_overrides[get_similar] = FakeSimilar
client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_classify_returns_top_category_and_ranked_probabilities():
    response = client.post('/classify', json={'description': 'Prune 1 x oak tree'})
    assert response.status_code == 200
    assert response.json() == {'category': 'trees', 'confidence': 0.8, 'probabilities': {'trees': 0.8, 'householder': 0.2},
                               'familiar': True}


def test_classify_rejects_a_too_short_description():
    assert client.post('/classify', json={'description': 'x'}).status_code == 422


def test_browser_preflight_from_another_site_is_allowed():
    headers = {'Origin': 'https://example.hf.space', 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type'}
    response = client.options('/classify', headers=headers)
    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] == '*'


def test_predict_uses_the_given_category():
    request = {'description': 'Roof terrace', 'borough': 'Camden', 'category': 'householder'}
    response = client.post('/predict', json=request)
    assert response.status_code == 200
    assert response.json() == {
        'risk': 0.1235, 'category': 'householder', 'category_predicted': False,
        'borough_average': 0.1, 'borough_decisions': 441, 'raising': ['roof terrace'], 'lowering': ['rear'],
        'words_known': 5, 'words_total': 6, 'similar': FakeSimilar().search('', '', ''), 'familiar': True,
    }


def test_predict_fills_a_missing_category_with_the_classifier():
    response = client.post('/predict', json={'description': 'Fell one oak', 'borough': 'Camden'})
    assert (response.json()['category'], response.json()['category_predicted']) == ('trees', True)
    assert risk_model.category == 'trees'


def test_predict_rejects_unknown_borough_and_category():
    assert client.post('/predict', json={'description': 'Roof terrace', 'borough': 'Paris'}).status_code == 422
    request = {'description': 'Roof terrace', 'borough': 'Camden', 'category': 'castle'}
    assert client.post('/predict', json=request).status_code == 422


def test_warm_builds_each_model_once(monkeypatch):
    '''/warm loads every model through the cached loaders, so later requests reuse them.'''
    built = []
    monkeypatch.setattr('planscope.api.CategoryClassifier', lambda path: built.append('classifier'))
    monkeypatch.setattr('planscope.api.RiskModel', lambda path: built.append('risk'))
    monkeypatch.setattr('planscope.api.SimilarSearch', lambda path, risk_model: built.append('similar'))
    loaders = (get_classifier, get_risk_model, get_similar)
    for loader in loaders:
        loader.cache_clear()
    try:
        assert client.get('/warm').json() == {'status': 'warm'}
        client.get('/warm')
        assert built == ['classifier', 'risk', 'similar']
    finally:
        for loader in loaders:
            loader.cache_clear()


def test_unfamiliar_text_is_flagged_by_both_endpoints():
    '''Words the risk model has never seen make the coverage check fail.'''
    description = 'asdf qwerty zxcv'
    assert client.post('/classify', json={'description': description}).json()['familiar'] is False
    assert client.post('/predict', json={'description': description, 'borough': 'Camden'}).json()['familiar'] is False