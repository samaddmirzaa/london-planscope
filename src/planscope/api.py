'''
PlanScope API: classify a planning application from its description.
'''

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from planscope.classifier import CategoryClassifier

MODEL_DIR = Path(os.environ.get('MODEL_DIR', Path(__file__).resolve().parents[2] / 'models' / 'category-onnx'))

app = FastAPI(title='PlanScope London', version='0.1.0')


class ClassifyRequest(BaseModel):
    description: str = Field(min_length=3, max_length=5000, examples=['Erection of a single storey rear extension'])


class ClassifyResponse(BaseModel):
    category: str
    confidence: float
    probabilities: dict[str, float]


@lru_cache
def get_classifier():
    '''Load the model once, on first use, and reuse it for every request.'''
    return CategoryClassifier(MODEL_DIR)


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