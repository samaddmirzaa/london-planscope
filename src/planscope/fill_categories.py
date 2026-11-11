'''
Fill missing categories with the classifier, keeping the form's category where it exists.
Writes data/processed/categories.parquet: id, category, category_source, category_confidence.
'''

from pathlib import Path

import pandas as pd

from planscope.category_data import normalise
from planscope.classifier import CategoryClassifier

ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = ROOT / 'data' / 'processed' / 'applications.parquet'
MODEL_DIR = ROOT / 'models' / 'category-onnx'
OUT_PATH = ROOT / 'data' / 'processed' / 'categories.parquet'


def main():
    df = pd.read_parquet(CLEAN_PATH, columns=['id', 'category', 'description'])
    missing = df['category'].isna() & df['description'].notna()
    labels, confidence = CategoryClassifier(MODEL_DIR).predict(normalise(df.loc[missing, 'description']).tolist())

    df['category_source'] = df['category'].notna().map({True: 'form', False: None})
    df.loc[missing, 'category'] = labels
    df.loc[missing, 'category_source'] = 'model'
    df.loc[missing, 'category_confidence'] = confidence

    df.drop(columns='description').to_parquet(OUT_PATH, index=False)
    print(f'Filled {missing.sum():,} categories; saved {OUT_PATH.name}')
    print(df['category_source'].value_counts(dropna=False))


if __name__ == '__main__':
    main()