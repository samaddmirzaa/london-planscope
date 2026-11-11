'''
Clean the raw applications into data/processed/applications.parquet.
Implements the decisions at the end of notebooks/01_data_profiling.ipynb.
'''

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Input and output files, located from this script's position in the project
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / 'data' / 'raw' / 'applications.parquet'
OUT_PATH = ROOT / 'data' / 'processed' / 'applications.parquet'

# Text that means "missing" (compared after whitespace is tidied)
PLACEHOLDERS = ['', 'NULL', 'Unknown', 'Missing', 'Not yet decided', 'No description provided by LDMS']

# Decision spellings (lowercase) that map to a clear outcome; anything else becomes OTHER
OUTCOMES = {
    'approved': 'APPROVED', 'approve': 'APPROVED',
    'pdv': 'APPROVED', 'parg': 'APPROVED',  # Ealing codes; their status column says Approved
    'refused': 'REFUSED', 'ref': 'REFUSED',
    'withdrawn': 'WITHDRAWN', 'wdn': 'WITHDRAWN', 'wthdrw': 'WITHDRAWN',
}

# Application type -> category: (pattern, category), first match wins, no match = 'other'
CATEGORY_RULES = [
    ('tree|hedgerow', 'trees'),
    ('^prior approval', 'prior_approval'),
    ('^lawful development', 'lawful_development'),
    ('condition', 'conditions'),
    ('amendment|reserved matters', 'amendment'),
    ('^householder|^enlargement of a dwellinghouse', 'householder'),
    ('^full planning|^outline', 'full_planning'),
    ('listed building', 'listed_building'),
    ('advert', 'advert'),
]

# Columns in the processed file, in order
COLUMNS = [
    'id', 'lpa_name', 'lpa_app_no', 'valid_date', 'decision_date', 'outcome',
    'decision_process', 'application_type', 'category', 'description',
    'units_proposed', 'units_existing', 'storeys', 'site_area_ha',
    'lon', 'lat', 'uprn', 'url_planning_app',
]


def clean_text(text):
    '''Collapse whitespace and control characters to single spaces, trim, and null placeholders.'''
    text = text.str.replace(r'[\s\x00-\x1f\x7f]+', ' ', regex=True).str.strip()
    return text.where(~text.isin(PLACEHOLDERS))


def parse_date(text):
    '''Parse day/month/year text into dates; anything unreadable becomes NaT.'''
    return pd.to_datetime(text, format='%d/%m/%Y', errors='coerce')


def parse_centroid(centroid):
    '''Turn centroid JSON text into lon/lat numbers; points outside London become NaN.'''
    coords = pd.DataFrame(centroid.map(json.loads).tolist(), index=centroid.index).astype(float)
    in_london = coords['lon'].between(-0.52, 0.34) & coords['lat'].between(51.28, 51.70)
    return coords.where(in_london)


def to_outcome(decision):
    '''Map decision spellings to APPROVED / REFUSED / WITHDRAWN / OTHER; no decision stays null.'''
    outcome = decision.str.lower().map(OUTCOMES).fillna('OTHER')
    return outcome.where(decision.notna())


def to_category(app_type):
    '''Group the detailed application types into a few categories; missing stays null.'''
    text = app_type.str.lower()
    matches = [text.str.contains(pattern, na=False) for pattern, _ in CATEGORY_RULES]
    names = [name for _, name in CATEGORY_RULES]
    category = pd.Series(np.select(matches, names, default='other'), index=app_type.index)
    return category.where(app_type.notna())


def parse_form(details):
    '''Pull homes, storeys and site area out of the application-form JSON; impossible values become NaN.'''
    filled = details.dropna()
    form = pd.json_normalize(filled.map(json.loads).tolist()).set_index(filled.index)
    buildings = form['building_details'].explode().dropna()
    storeys = pd.to_numeric(buildings.map(lambda b: b.get('no_storeys')), errors='coerce').groupby(level=0).max()
    site = pd.to_numeric(form['site_area'], errors='coerce')
    out = pd.DataFrame(index=details.index)
    out['units_proposed'] = pd.to_numeric(form['residential_details.total_no_proposed_residential_units'], errors='coerce')
    out['units_existing'] = pd.to_numeric(form['residential_details.total_no_existing_residential_units'], errors='coerce')
    out['storeys'] = storeys.where(storeys.between(1, 100))
    out['site_area_ha'] = site.where((site > 0) & (site <= 100))
    return out


def clean(raw):
    '''Apply every cleaning decision to the raw table and return the processed table.'''
    snapshot = pd.to_datetime(raw['last_updated'], utc=True).max().tz_localize(None)

    # Keep applications submitted from 2019 up to the snapshot
    valid = parse_date(raw['valid_date'])
    df = raw[valid.between('2019-01-01', snapshot)].copy()
    df['valid_date'] = valid

    # Decision dates must fall between the valid date and the snapshot
    decided = parse_date(df['decision_date'])
    df['decision_date'] = decided.where(decided.between(df['valid_date'], snapshot))

    # Tidy text columns and turn placeholders into nulls
    for col in ['lpa_app_no', 'description', 'decision', 'decision_process', 'application_type_full', 'uprn', 'url_planning_app']:
        df[col] = clean_text(df[col])

    # Standard values for outcome, decision route and category
    df['outcome'] = to_outcome(df['decision'])
    df['decision_process'] = df['decision_process'].where(df['decision_process'].isin(['Delegated', 'Committee']))
    df['category'] = to_category(df['application_type_full'])

    # Numbers from the JSON columns
    df = df.join(parse_centroid(df['centroid'])).join(parse_form(df['application_details']))
    return df[COLUMNS].reset_index(drop=True)


def main():
    df = clean(pd.read_parquet(RAW_PATH))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f'Saved {OUT_PATH.name}: {len(df):,} rows, {len(df.columns)} columns')
    print(df['outcome'].value_counts(dropna=False))


if __name__ == '__main__':
    main()