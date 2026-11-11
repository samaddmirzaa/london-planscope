'''
Category classifier for CPU, running the exported ONNX model.
Needs only onnxruntime and tokenizers, so it is light enough to serve from AWS Lambda.
'''

import json
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MAX_TOKENS = 128  # must match training

def prepare(text):
    '''Tidy a description exactly as the training text was (clean_text, then normalise).'''
    return re.sub(r'[\s\x00-\x1f\x7f]+', ' ', text).strip().lower().rstrip('.')

class CategoryClassifier:
    '''Predict an application's category from its description.'''

    def __init__(self, model_dir, model_file='model.int8.onnx'):
        model_dir = Path(model_dir)
        self.labels = json.loads((model_dir / 'labels.json').read_text())
        self.tokenizer = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(str(model_dir / model_file), providers=['CPUExecutionProvider'])

    def predict_proba(self, texts):
        '''Probability of every category for each text: an array of shape (texts, categories).'''
        encodings = self.tokenizer.encode_batch([prepare(text) for text in texts])
        inputs = {
            'input_ids': np.array([e.ids for e in encodings], dtype=np.int64),
            'attention_mask': np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        logits = self.session.run(['logits'], inputs)[0]
        scores = np.exp(logits - logits.max(axis=1, keepdims=True))
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, texts):
        '''Most likely category and its probability for each text.

        Texts go one at a time: on CPU that is as fast as batching, and int8 results then never
        depend on which other texts share a batch, so bulk predictions match the API exactly.
        '''
        probs = np.vstack([self.predict_proba([text]) for text in texts])
        return [self.labels[i] for i in probs.argmax(axis=1)], probs.max(axis=1)