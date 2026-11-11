'''
Build the datasets for the category classifier.

train / val / test: distinct descriptions from labelled applications, each with
its most common category, split 80/10/10 at random.
missing_test: unlabelled applications whose council reference code reliably
means one category, the population the classifier will actually label.
'''

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = ROOT / 'data' / 'processed' / 'applications.parquet'
OUT_DIR = ROOT / 'data' / 'processed' / 'category'
SEED = 42

# A reference code counts as reliable if it appears 30+ times among labelled
# applications of one authority and means one category at least 95% of the time
MIN_COUNT = 30
MIN_PURITY = 0.95


def normalise(text):
    '''Lowercase and drop a trailing full stop, so trivial variants count as one text.'''
    return text.str.lower().str.rstrip('.')


def labelled_texts(df):
    '''One row per distinct description among labelled applications, with its most common category.'''
    lab = df[df['category'].notna() & df['description'].notna() & (df['category'] != 'other')]
    counts = lab.assign(text=normalise(lab['description'])).value_counts(['text', 'category']).reset_index()
    counts = counts.sort_values(['text', 'count', 'category'], ascending=[True, False, True], kind='stable')
    return counts.drop_duplicates('text')[['text', 'category']].reset_index(drop=True)


def reference_code(lpa_app_no):
    '''The letters at the end of a council reference, e.g. 19/3771/HSE -> HSE, 20/01289/FULL6 -> FULL.'''
    return lpa_app_no.str.upper().str.extract(r'([A-Z]+)\d*$')[0]


def code_labels(df):
    '''Label unlabelled applications using reference codes that reliably mean one category.'''
    df = df.assign(code=reference_code(df['lpa_app_no']))
    lab = df[df['category'].notna() & df['code'].notna()]
    rules = lab.groupby(['lpa_name', 'code'])['category'].agg(
        n='size',
        label=lambda s: s.value_counts().index[0],
        purity=lambda s: s.value_counts(normalize=True).iloc[0],
    )
    rules = rules[(rules['n'] >= MIN_COUNT) & (rules['purity'] >= MIN_PURITY)]
    unlabelled = df[df['category'].isna() & df['description'].notna()]
    out = unlabelled.join(rules['label'], on=['lpa_name', 'code'], how='inner')
    return pd.DataFrame({'text': normalise(out['description']), 'category': out['label'], 'lpa_name': out['lpa_name']})


def main():
    df = pd.read_parquet(CLEAN_PATH)
    texts = labelled_texts(df).sample(frac=1, random_state=SEED).reset_index(drop=True)
    cut1, cut2 = int(len(texts) * 0.8), int(len(texts) * 0.9)
    sets = {
        'train': texts[:cut1],
        'val': texts[cut1:cut2],
        'test': texts[cut2:],
        'missing_test': code_labels(df),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in sets.items():
        data.to_parquet(OUT_DIR / f'{name}.parquet', index=False)
        print(f'{name:13} {len(data):>8,} rows')
    print(pd.DataFrame({name: data['category'].value_counts(normalize=True) for name, data in sets.items()}).mul(100).round(1))


if __name__ == '__main__':
    main()