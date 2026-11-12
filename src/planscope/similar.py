'''
Similar past applications: the decided applications in the same borough and of the same type whose
descriptions share the most weighted words and word pairs with a new one (TF-IDF cosine similarity,
chosen in notebooks/05_similar_applications.ipynb). The index reuses the outcome model's vocabulary,
so it needs no model of its own, and searching it needs only numpy.
'''

import hashlib
import json
from pathlib import Path

import numpy as np

from planscope.risk import RiskModel, prepare

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / 'models' / 'similar.npz'
FIELDS = ['reference', 'date', 'outcome', 'description', 'url']
SHOWN = 300  # characters of each description kept for display


def fingerprint(terms):
    '''Short hash of a vocabulary, so an index is never searched with a different one.'''
    return hashlib.sha256('\n'.join(terms).encode()).hexdigest()[:16]


def build(df, risk_model):
    '''Index arrays: a TF-IDF row per application, sorted so each borough and type is one block of rows.'''
    df = df.assign(key=df['lpa_name'] + ' | ' + df['category']).sort_values('key', kind='stable').reset_index(drop=True)
    vectors = [risk_model.vectorize(prepare(text)) for text in df['description']]

    # What the page shows for each application, as one JSON line per row in a single byte string
    shown = df[['lpa_app_no', 'valid_date', 'outcome', 'description', 'url_planning_app']].assign(
        valid_date=df['valid_date'].dt.strftime('%Y-%m'),
        description=df['description'].map(lambda t: t if len(t) <= SHOWN else t[:SHOWN - 1] + '…'),
        url_planning_app=df['url_planning_app'].where(df['url_planning_app'].str.startswith('http', na=False)),
    )
    shown = shown.astype(object).where(shown.notna(), None)
    records = [json.dumps(row).encode() for row in shown.values.tolist()]

    starts = df['key'].drop_duplicates()  # first row of each block, since rows are sorted by key
    ends = [*starts.index[1:], len(df)]
    return {
        'indptr': np.cumsum([0] + [len(ids) for ids, _ in vectors]),
        'indices': np.concatenate([ids for ids, _ in vectors]).astype(np.int32),
        'weights': np.concatenate([w for _, w in vectors]).astype(np.float16),
        'records': np.frombuffer(b''.join(records), dtype=np.uint8),
        'offsets': np.cumsum([0] + [len(r) for r in records]),
        'groups': np.array(json.dumps({key: [int(s), int(e)] for key, s, e in zip(starts, starts.index, ends)})),
        'vocabulary': np.array(fingerprint(risk_model.terms)),
    }


class SimilarSearch:
    '''The k most similar decided applications in one borough and type, best first.'''

    def __init__(self, path, risk_model):
        with np.load(path) as index:
            if str(index['vocabulary']) != fingerprint(risk_model.terms):
                raise ValueError(f'{path} was built for a different outcome model: rebuild it')
            self.indptr, self.indices, self.weights = index['indptr'], index['indices'], index['weights']
            self.records, self.offsets = index['records'], index['offsets']
            self.groups = json.loads(str(index['groups']))
        self.risk_model = risk_model

    def search(self, description, borough, category, k=10):
        start, end = self.groups.get(f'{borough} | {category}', (0, 0))
        ids, tfidf = self.risk_model.vectorize(prepare(description))
        query = np.zeros(len(self.risk_model.terms), dtype=np.float32)
        query[ids] = tfidf

        # Cosine similarity with every row in the block: rows and query both have length 1,
        # so it is the sum, over shared terms, of row weight x query weight
        a, b = self.indptr[start], self.indptr[end]
        rows = np.repeat(np.arange(end - start), np.diff(self.indptr[start:end + 1]))
        scores = np.bincount(rows, weights=self.weights[a:b] * query[self.indices[a:b]], minlength=end - start)

        best = np.argsort(-scores, kind='stable')[:k]
        return [{**self.record(start + i), 'similarity': round(float(scores[i]), 3)} for i in best if scores[i] > 0]

    def record(self, row):
        return dict(zip(FIELDS, json.loads(self.records[self.offsets[row]:self.offsets[row + 1]].tobytes())))


def main():
    # Build-time only: pandas and the data paths are never needed by the API
    import pandas as pd

    from planscope.outcome_data import CATEGORIES_PATH, CLEAN_PATH
    from planscope.outcome_model import OUT_PATH as RISK_PATH

    columns = ['id', 'lpa_name', 'lpa_app_no', 'valid_date', 'outcome', 'description', 'url_planning_app']
    apps = pd.read_parquet(CLEAN_PATH, columns=columns)
    apps = apps[apps['outcome'].isin(['APPROVED', 'REFUSED']) & apps['description'].notna()]
    df = apps.merge(pd.read_parquet(CATEGORIES_PATH, columns=['id', 'category']), on='id', validate='one_to_one')
    df = df.dropna(subset=['category'])

    np.savez(OUT_PATH, **build(df, RiskModel(RISK_PATH)))
    print(f'Saved {OUT_PATH.name}: {len(df):,} decided applications, {OUT_PATH.stat().st_size / 1e6:.0f} MB')


if __name__ == '__main__':
    main()