'''
Download the Planning London Datahub 'applications' index to data/raw/.

Step 1, download: page through every record with the scroll API and save
each page to applications.jsonl. Skipped if that file already exists.
Step 2, convert: turn the JSONL into applications.parquet.

To re-download, delete data/raw/applications.jsonl first.
Run: python src/planscope/ingest.py
'''

import json
import time
from pathlib import Path

import pandas as pd
import requests

# API address and the access header the guest gateway requires
BASE = 'https://planningdata.london.gov.uk/api-guest'
HEADERS = {'X-API-AllowRequest': 'be2rmRnt&', 'Content-Type': 'application/json'}
PAGE_SIZE = 1000

# Paths built from this file's location, so the script works from any folder
RAW_DIR = Path(__file__).resolve().parents[2] / 'data' / 'raw'
JSONL_PATH = RAW_DIR / 'applications.jsonl'
PART_PATH = RAW_DIR / 'applications.jsonl.part'
PARQUET_PATH = RAW_DIR / 'applications.parquet'

# Fields to download (everything else in the record is skipped)
FIELDS = [
    'id', 'lpa_app_no', 'lpa_name', 'borough', 'ward', 'postcode', 'uprn',
    'valid_date', 'decision', 'decision_date', 'decision_process', 'decision_agency',
    'application_type', 'application_type_full', 'development_type', 'description',
    'status', 'appeal_status', 'appeal_decision', 'appeal_decision_date',
    'centroid', 'url_planning_app', 'last_updated',
    # Structured facts from the application form
    'application_details.residential_details.total_no_proposed_residential_units',
    'application_details.residential_details.total_no_existing_residential_units',
    'application_details.building_details.no_storeys',
    'application_details.site_area',
]


def post(url, body, tries=5):
    '''POST a JSON body and return the JSON reply. Retry network and server errors.'''
    for attempt in range(1, tries + 1):
        try:
            resp = requests.post(url, headers=HEADERS, json=body, timeout=60)
        except requests.RequestException as err:
            print(f'  network error ({err}), retry {attempt}/{tries}')
        else:
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp.json()
            print(f'  server error {resp.status_code}, retry {attempt}/{tries}')
        time.sleep(2 ** attempt)
    raise RuntimeError(f'Gave up after {tries} tries: {url}')


def flatten(record):
    '''Store nested values (dicts, lists) as JSON text, so each column holds one simple type.'''
    return {key: json.dumps(value) if isinstance(value, (dict, list)) else value
            for key, value in record.items()}


def download():
    '''Scroll through the whole index, writing one record per line to a .part file.'''
    body = {'size': PAGE_SIZE, '_source': FIELDS, 'sort': ['_doc']}
    page = post(f'{BASE}/applications/_search?scroll=5m', body)
    total = 0
    with open(PART_PATH, 'w', encoding='utf-8') as f:
        while page['hits']['hits']:
            for hit in page['hits']['hits']:
                f.write(json.dumps(flatten(hit['_source'])) + '\n')
            total += len(page['hits']['hits'])
            if total % 50_000 == 0:
                print(f'  {total:,} records')
            time.sleep(0.3)
            page = post(f'{BASE}/_search/scroll', {'scroll': '5m', 'scroll_id': page['_scroll_id']})
    PART_PATH.replace(JSONL_PATH)
    print(f'Downloaded {total:,} records')


def convert():
    '''Read the JSONL in chunks and save it as one parquet file, every column as text.'''
    chunks = pd.read_json(JSONL_PATH, lines=True, chunksize=100_000, dtype=False)
    df = pd.concat(chunks, ignore_index=True).astype(str)
    df.to_parquet(PARQUET_PATH, index=False)
    print(f'Saved {PARQUET_PATH.name}: {len(df):,} rows, {len(df.columns)} columns')


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if JSONL_PATH.exists():
        print(f'{JSONL_PATH.name} exists, skipping download (delete it to re-download)')
    else:
        download()
    convert()


if __name__ == '__main__':
    main()