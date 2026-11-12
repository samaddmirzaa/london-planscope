'''
Build the outcome dataset: approved or refused applications, described only by what is known
when they are submitted, split by date into train / val / test.
'''

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = ROOT / 'data' / 'processed' / 'applications.parquet'
CATEGORIES_PATH = ROOT / 'data' / 'processed' / 'categories.parquet'
OUT_PATH = ROOT / 'data' / 'processed' / 'outcome.parquet'

# Text councils add to a description after submission: appeal links and amendment notes
POST_SUBMISSION = r'linked cases:.*|\(?\b(?:amended|revised) (?:plans|drawings|description)\b\)?|\((?:amended|revised)\)'

# Bin edges for the application-form numbers (one-hot encoded later; missing gets its own bin)
BINS = {
    'units_proposed': [0, 1, 2, 5, 10, 50],
    'storeys': [1, 2, 3, 5, 10],
    'site_area_ha': [0.01, 0.05, 0.1, 0.5, 1],
}


def strip_post_submission(text):
    '''Remove text added after submission, so the model only sees what an applicant would write.'''
    return text.str.replace(POST_SUBMISSION, '', regex=True, case=False).str.strip()


def to_bins(values, edges):
    '''Numbers -> interval labels such as (2.0, 5.0]; missing values -> 'missing'.'''
    return pd.cut(values, [-np.inf, *edges, np.inf]).astype(str).where(values.notna(), 'missing')


def assign_split(valid_date):
    '''test = the latest 12 months that have had 12 more months to be decided; val = the 12 before; train = earlier.'''
    cut = valid_date.max() - pd.DateOffset(months=12)
    edges = [cut - pd.DateOffset(months=24), cut - pd.DateOffset(months=12), cut]
    split = np.select([valid_date < edge for edge in edges], ['train', 'val', 'test'], 'recent')
    return pd.Series(split, index=valid_date.index)


def build(apps, categories):
    '''One row per approved or refused application with a description, ready for modelling.'''
    df = apps.drop(columns='category').merge(categories[['id', 'category']], on='id', validate='one_to_one')
    df['split'] = assign_split(df['valid_date'])
    df = df[df['outcome'].isin(['APPROVED', 'REFUSED']) & df['description'].notna() & (df['split'] != 'recent')]

    out = df[['id', 'lpa_name', 'category', 'valid_date', 'split']].copy()
    out['text'] = strip_post_submission(df['description'])
    out['lpa_category'] = df['lpa_name'] + ' | ' + df['category'].fillna('none')
    for col, edges in BINS.items():
        out[col] = to_bins(df[col], edges)
    out['refused'] = df['outcome'].eq('REFUSED').astype(int)
    return out.reset_index(drop=True)


def main():
    df = build(pd.read_parquet(CLEAN_PATH), pd.read_parquet(CATEGORIES_PATH))
    df.to_parquet(OUT_PATH, index=False)
    print(f'Saved {OUT_PATH.name}: {len(df):,} rows')
    summary = df.groupby('split').agg(rows=('id', 'size'), refusal_rate=('refused', 'mean'),
                                      first=('valid_date', 'min'), last=('valid_date', 'max'))
    print(summary.loc[['train', 'val', 'test']].round({'refusal_rate': 3}))


if __name__ == '__main__':
    main()