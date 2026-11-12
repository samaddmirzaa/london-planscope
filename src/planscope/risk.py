'''
Refusal risk for one application, from the exported outcome model (models/outcome.json).
'''

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

# Text councils add to a description after submission: appeal links and amendment notes
POST_SUBMISSION = r'linked cases:.*|\(?\b(?:amended|revised) (?:plans|drawings|description)\b\)?|\((?:amended|revised)\)'
TOKEN = re.compile(r'(?u)\b\w\w+\b')  # scikit-learn's default: words of 2+ letters or digits
MIN_EFFECT = 0.1  # smallest change in log-odds worth showing as a reason


def prepare(text):
    '''Remove post-submission text and lowercase, exactly as for training.'''
    return re.sub(POST_SUBMISSION, '', text, flags=re.IGNORECASE).strip().lower()


def terms(text):
    '''Single words and adjacent word pairs, the TF-IDF vocabulary's units.'''
    words = TOKEN.findall(text)
    return words + [f'{a} {b}' for a, b in zip(words, words[1:])]


class RiskModel:
    '''Calibrated refusal risk with the words that raised and lowered it.'''

    def __init__(self, path):
        model = json.loads(Path(path).read_text())
        self.terms = model['terms']
        self.index = {term: i for i, term in enumerate(self.terms)}
        self.idf = np.array(model['idf'])
        self.weights = np.array(model['weights'])
        self.intercept = model['intercept']
        self.groups = model['groups']
        self.averages = model['averages']

    def vectorize(self, text):
        '''TF-IDF of prepared text: ids of the known terms and their weights, (1 + log count) x idf, scaled to length 1.'''
        counts = Counter(self.index[term] for term in terms(text) if term in self.index)
        ids = np.fromiter(counts, dtype=int, count=len(counts))
        tfidf = (1 + np.log(np.fromiter(counts.values(), dtype=float, count=len(counts)))) * self.idf[ids]
        return ids, tfidf / (np.linalg.norm(tfidf) or 1.0)

    def predict(self, description, borough, category):
        text = prepare(description)
        ids, tfidf = self.vectorize(text)
        contributions = tfidf * self.weights[ids]  # each term's share of the score

        key = f'{borough} | {category}'
        group_score = sum(self.groups[col].get(value, 0.0)
                          for col, value in [('lpa_name', borough), ('category', category), ('lpa_category', key)])
        score = self.intercept + group_score + contributions.sum()

        ranked = sorted(zip([self.terms[i] for i in ids], contributions.tolist()), key=lambda pair: pair[1])
        words = TOKEN.findall(text)
        return {
            'risk': 1 / (1 + math.exp(-score)),
            'average': self.averages.get(key),
            'raising': [term for term, c in ranked[::-1][:3] if c > MIN_EFFECT],
            'lowering': [term for term, c in ranked[:3] if c < -MIN_EFFECT],
            'words_known': sum(word in self.index for word in words),
            'words_total': len(words),
        }