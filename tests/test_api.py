import numpy as np
from fastapi.testclient import TestClient

from planscope.api import app, get_classifier


class FakeClassifier:
    '''Stands in for the real model, so the API is tested without model files.'''
    labels = ['householder', 'trees']

    def predict_proba(self, texts):
        return np.array([[0.2, 0.8]])


app.dependency_overrides[get_classifier] = FakeClassifier
client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_classify_returns_top_category_and_ranked_probabilities():
    response = client.post('/classify', json={'description': 'Prune 1 x oak tree'})
    assert response.status_code == 200
    assert response.json() == {'category': 'trees', 'confidence': 0.8, 'probabilities': {'trees': 0.8, 'householder': 0.2}}


def test_classify_rejects_a_too_short_description():
    assert client.post('/classify', json={'description': 'x'}).status_code == 422

def test_browser_preflight_from_another_site_is_allowed():
    headers = {'Origin': 'https://example.hf.space', 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type'}
    response = client.options('/classify', headers=headers)
    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] == '*'