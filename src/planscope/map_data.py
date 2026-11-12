'''
Build the front end's borough map: space/boroughs.geojson.
ONS boundaries of the 33 London boroughs, each with its applications, approvals and refusals per category.
'''

import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = ROOT / 'data' / 'processed' / 'applications.parquet'
CATEGORIES_PATH = ROOT / 'data' / 'processed' / 'categories.parquet'
OUT_PATH = ROOT / 'space' / 'boroughs.geojson'

# ONS local authority boundaries (May 2024, generalised to 20 m); E09 codes are the London boroughs
BOUNDARIES_URL = ('https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/'
                  'Local_Authority_Districts_May_2024_Boundaries_UK_BGC/FeatureServer/0/query')
BOUNDARIES_QUERY = {'where': "LAD24CD LIKE 'E09%'", 'outFields': 'LAD24NM', 'outSR': 4326,
                    'geometryPrecision': 4, 'f': 'geojson'}


def authority_name(ons_name):
    '''ONS borough name -> the Datahub's lpa_name, e.g. Kingston upon Thames -> Kingston.'''
    return ons_name.replace(' upon Thames', '').replace(' and ', ' & ')


def borough_stats(df):
    '''{authority: {category: [applications, approved, refused]}}, with 'all' covering every application.'''
    df = df.assign(approved=df['outcome'].eq('APPROVED'), refused=df['outcome'].eq('REFUSED'))
    df = pd.concat([df, df.assign(category='all')])
    table = df.groupby(['lpa_name', 'category']).agg(
        applications=('id', 'size'), approved=('approved', 'sum'), refused=('refused', 'sum'))
    return {lpa: group.droplevel(0).T.to_dict('list') for lpa, group in table.groupby(level=0)}


def main():
    apps = pd.read_parquet(CLEAN_PATH, columns=['id', 'lpa_name', 'valid_date', 'outcome'])
    categories = pd.read_parquet(CATEGORIES_PATH, columns=['id', 'category'])
    stats = borough_stats(apps.merge(categories, on='id', validate='one_to_one'))

    response = requests.get(BOUNDARIES_URL, params=BOUNDARIES_QUERY, timeout=60)
    response.raise_for_status()
    boroughs = response.json()
    for feature in boroughs['features']:
        name = feature['properties']['LAD24NM']
        feature['properties'] = {'name': name, 'stats': stats[authority_name(name)]}
    boroughs['period'] = [f'{apps["valid_date"].min():%b %Y}', f'{apps["valid_date"].max():%b %Y}']

    OUT_PATH.write_text(json.dumps(boroughs, separators=(',', ':')))
    print(f'Saved {OUT_PATH.name}: {len(boroughs["features"])} boroughs, '
          f'{OUT_PATH.stat().st_size / 1024:.0f} KB, applications validated {" to ".join(boroughs["period"])}')


if __name__ == '__main__':
    main()