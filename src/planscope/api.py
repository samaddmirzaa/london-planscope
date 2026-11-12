'''
PlanScope API: classify a planning application, estimate its refusal risk, and find similar past decisions.
'''

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from planscope.classifier import CategoryClassifier
from planscope.risk import RiskModel
from planscope.similar import SimilarSearch

MODELS_DIR = Path(os.environ.get('MODELS_DIR', Path(__file__).resolve().parents[2] / 'models'))

app = FastAPI(title='PlanScope London', version='0.3.0')

# Let web pages on other sites call the API (the browser's preflight check is answered here)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET', 'POST'], allow_headers=['content-type'])


class ClassifyRequest(BaseModel):
    description: str = Field(min_length=3, max_length=5000, examples=['Erection of a single storey rear extension'])


class ClassifyResponse(BaseModel):
    category: str
    confidence: float
    probabilities: dict[str, float]


class PredictRequest(ClassifyRequest):
    borough: str = Field(examples=['Camden'])
    category: str | None = Field(default=None, examples=['householder'])


class SimilarApplication(BaseModel):
    reference: str
    date: str | None
    outcome: str
    description: str
    url: str | None
    similarity: float


class PredictResponse(BaseModel):
    risk: float
    category: str
    category_predicted: bool
    borough_average: float | None
    borough_decisions: int
    raising: list[str]
    lowering: list[str]
    words_known: int
    words_total: int
    similar: list[SimilarApplication]


@lru_cache
def get_classifier():
    '''Load the category model once, on first use, and reuse it for every request.'''
    return CategoryClassifier(MODELS_DIR / 'category-onnx')


@lru_cache
def get_risk_model():
    '''Load the outcome model once, on first use.'''
    return RiskModel(MODELS_DIR / 'outcome.json')


@lru_cache
def get_similar():
    '''Load the similar-applications index once; it shares the outcome model's vocabulary.'''
    return SimilarSearch(MODELS_DIR / 'similar.npz', get_risk_model())


@app.get('/health')
def health():
    '''Liveness check used by the deployment.'''
    return {'status': 'ok'}


@app.post('/classify')
def classify(request: ClassifyRequest, classifier: Annotated[CategoryClassifier, Depends(get_classifier)]) -> ClassifyResponse:
    '''Most likely application category for a description, with every category's probability.'''
    probs = classifier.predict_proba([request.description])[0]
    ranked = sorted(zip(classifier.labels, probs.tolist()), key=lambda pair: pair[1], reverse=True)
    return ClassifyResponse(
        category=ranked[0][0],
        confidence=round(ranked[0][1], 4),
        probabilities={label: round(p, 4) for label, p in ranked},
    )


@app.post('/predict')
def predict(request: PredictRequest,
            classifier: Annotated[CategoryClassifier, Depends(get_classifier)],
            risk_model: Annotated[RiskModel, Depends(get_risk_model)],
            similar: Annotated[SimilarSearch, Depends(get_similar)]) -> PredictResponse:
    '''Refusal risk if decided, the words that raised and lowered it, and the most similar past decisions
    in the same borough and type; the category is predicted if not given.'''
    if request.borough not in risk_model.groups['lpa_name']:
        raise HTTPException(422, f'Unknown borough: {request.borough}')
    if request.category and request.category not in classifier.labels:
        raise HTTPException(422, f'Unknown category: {request.category}')

    category = request.category or classifier.predict([request.description])[0][0]
    result = risk_model.predict(request.description, request.borough, category)
    rate, decisions = result['average'] or (None, 0)
    return PredictResponse(
        risk=round(result['risk'], 4),
        category=category,
        category_predicted=not request.category,
        borough_average=rate,
        borough_decisions=decisions,
        raising=result['raising'],
        lowering=result['lowering'],
        words_known=result['words_known'],
        words_total=result['words_total'],
        similar=similar.search(request.description, request.borough, category),
    )