"""
ingest.py: Download the Planning London Datahub 'applications' index to disk.


       - "scroll"        : Elasticsearch's built-in bulk-export mode (preferred)
       - "search_after"  : page through sorted results (fallback if scroll is blocked)
     The guest gateway blocks the /_mapping endpoint (403), so if we need
     search_after we PROBE for sortable fields empirically instead of
     reading the schema.
  2. Appends every page to data/raw/applications.jsonl as it goes
     (crash-safe: if the run dies, you keep everything downloaded so far).
  3. Converts the JSONL to data/raw/applications.parquet at the end.

First run: keep MAX_RECORDS = 5_000 below and confirm everything works.
Then set MAX_RECORDS = None and run again for the full dataset

Run from the PROJECT ROOT:  python src/planscope/ingest.py

Requires:  pip install requests pandas pyarrow
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ----------------------------- configuration -----------------------------

BASE = "https://planningdata.london.gov.uk/api-guest"
INDEX = "applications"
HEADERS = {"X-API-AllowRequest": "be2rmRnt&", "Content-Type": "application/json"}

PAGE_SIZE = 1000          # records per request (1000 is the polite maximum)
SLEEP_BETWEEN_PAGES = 0.3 # seconds; be gentle with a free public API
MAX_RECORDS = None       # <<< set to None for the FULL download after the test run

OUT_DIR = Path("data/raw")
JSONL_PATH = OUT_DIR / "applications.jsonl"
PARQUET_PATH = OUT_DIR / "applications.parquet"

# Only pull the fields we actually need (faster + smaller).
FIELDS = [
    "id", "lpa_app_no", "lpa_name", "borough", "ward", "postcode", "uprn",
    "valid_date", "decision", "decision_date", "decision_process",
    "decision_agency", "application_type", "application_type_full",
    "development_type", "description", "status",
    "appeal_status", "appeal_decision", "appeal_decision_date",
    "centroid", "url_planning_app", "last_updated",
]

# Fields we will try, in order, when probing for a sortable field.
CANDIDATE_SORT_FIELDS = [
    "last_updated", "valid_date", "id", "pp_id", "lpa_app_no",
    "last_updated.keyword", "id.keyword", "pp_id.keyword",
]

# ------------------------- small helper functions ------------------------


def request_with_retry(method: str, url: str, max_tries: int = 5, **kwargs):
    """Send a request; on server errors / network blips, wait and retry.

    Waits 2s, 4s, 8s, 16s between attempts (exponential backoff) — the
    standard polite pattern for talking to any API.
    """
    for attempt in range(1, max_tries + 1):
        try:
            resp = requests.request(method, url, headers=HEADERS, timeout=60, **kwargs)
            if resp.status_code < 500:      # 2xx/3xx/4xx: return to caller
                return resp
            print(f"  server error {resp.status_code}, attempt {attempt}/{max_tries}")
        except requests.RequestException as exc:
            print(f"  network error ({exc}), attempt {attempt}/{max_tries}")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Gave up after {max_tries} attempts: {method} {url}")


def probe_sort_fields() -> list:
    """The guest gateway blocks /_mapping (403), so we can't read the schema.

    Instead, discover sortable fields empirically: request 1 record sorted
    by each candidate field and keep the ones the server accepts. We want
    two (a primary + a tiebreaker) so search_after never skips records that
    share an identical primary sort value across a page boundary.
    """
    working = []
    print("Probing for sortable fields ...")
    for field in CANDIDATE_SORT_FIELDS:
        body = {
            "size": 1,
            "_source": False,
            "query": {"match_all": {}},
            "sort": [{field: "asc"}],
        }
        resp = request_with_retry("POST", f"{BASE}/{INDEX}/_search", json=body)
        if resp.status_code == 200:
            hits = resp.json().get("hits", {}).get("hits", [])
            if hits and "sort" in hits[0]:
                working.append(field)
                print(f"  sortable: {field}")
        if len(working) == 2:
            break
    if not working:
        raise RuntimeError("No sortable field accepted — investigate manually.")
    return [{f: "asc"} for f in working]


def write_page(hits: list, fh) -> None:
    """Append one page of records to the JSONL file (one JSON object per line)."""
    for hit in hits:
        fh.write(json.dumps(hit["_source"]) + "\n")


def reached_limit(count: int) -> bool:
    return MAX_RECORDS is not None and count >= MAX_RECORDS


# --------------------------- download strategies -------------------------


def download_with_scroll(fh) -> int:
    """Strategy A: the scroll API — Elasticsearch's bulk-export mode.

    You open a 'scroll' (a temporary snapshot cursor on the server) and keep
    asking 'next page' with the scroll_id it gives you. Designed for exactly
    this job. The guest gateway may block it; if so we raise and fall back.
    """
    body = {
        "size": PAGE_SIZE,
        "_source": FIELDS,
        "query": {"match_all": {}},
        "sort": ["_doc"],   # '_doc' = fastest order for a full export
    }
    resp = request_with_retry("POST", f"{BASE}/{INDEX}/_search?scroll=5m", json=body)
    if resp.status_code != 200:
        raise RuntimeError(f"scroll not available (HTTP {resp.status_code})")

    data = resp.json()
    scroll_id = data.get("_scroll_id")
    if scroll_id is None:
        raise RuntimeError("scroll not available (no scroll_id returned)")

    total = 0
    page = 0
    while True:
        hits = data["hits"]["hits"]
        if not hits:
            break
        write_page(hits, fh)
        total += len(hits)
        page += 1
        if page % 10 == 0:
            print(f"  [scroll] {total:,} records downloaded...")
        if reached_limit(total):
            break
        time.sleep(SLEEP_BETWEEN_PAGES)
        resp = request_with_retry(
            "POST", f"{BASE}/_search/scroll",
            json={"scroll": "5m", "scroll_id": scroll_id},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"scroll page failed (HTTP {resp.status_code})")
        data = resp.json()
        scroll_id = data.get("_scroll_id", scroll_id)
    return total


def download_with_search_after(fh, sort_spec: list) -> int:
    """Strategy B: search_after pagination.

    Sort all records in a stable order; each request says 'give me the next
    1,000 AFTER this bookmark'. The bookmark is the sort value of the last
    record of the previous page. Unlike simple offsets, this has no 10,000-
    record ceiling.
    """
    total = 0
    page = 0
    search_after = None
    while True:
        body = {
            "size": PAGE_SIZE,
            "_source": FIELDS,
            "query": {"match_all": {}},
            "sort": sort_spec,
        }
        if search_after is not None:
            body["search_after"] = search_after

        resp = request_with_retry("POST", f"{BASE}/{INDEX}/_search", json=body)
        if resp.status_code != 200:
            raise RuntimeError(
                f"search_after failed (HTTP {resp.status_code}): {resp.text[:300]}"
            )
        hits = resp.json()["hits"]["hits"]
        if not hits:
            break
        write_page(hits, fh)
        search_after = hits[-1]["sort"]   # bookmark for the next page
        total += len(hits)
        page += 1
        if page % 10 == 0:
            print(f"  [search_after] {total:,} records downloaded...")
        if reached_limit(total):
            break
        time.sleep(SLEEP_BETWEEN_PAGES)
    return total


# --------------------------------- main ----------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if JSONL_PATH.exists():
        print(f"{JSONL_PATH} already exists — delete it to re-download. Aborting.")
        sys.exit(1)

    started = time.time()
    with open(JSONL_PATH, "w", encoding="utf-8") as fh:
        try:
            print("Trying strategy A: scroll ...")
            total = download_with_scroll(fh)
            strategy = "scroll"
        except RuntimeError as exc:
            print(f"Scroll unavailable ({exc}) — falling back to search_after.")
            total = download_with_search_after(fh, probe_sort_fields())
            strategy = "search_after"

    mins = (time.time() - started) / 60
    print(f"\nDownloaded {total:,} records via {strategy} in {mins:.1f} min")
    print(f"Raw file: {JSONL_PATH}")

    print("Converting to parquet ...")
    chunks = pd.read_json(JSONL_PATH, lines=True, chunksize=100_000)
    df = pd.concat(chunks, ignore_index=True)

    def to_json_string(val):
        """Convert any Python object into a JSON-safe string."""
        if val is None or pd.isna(val):
            return None

        if isinstance(val, (dict, list)):
            return json.dumps(val)

        return str(val)

    # Convert centroid explicitly
    df["centroid"] = df["centroid"].apply(to_json_string)

    # Convert any remaining object columns to strings so PyArrow
    # doesn't fail on mixed Python types.
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(to_json_string)

    df.to_parquet(
        PARQUET_PATH,
        engine="pyarrow",
        index=False,
    )

    print(f"Done: {PARQUET_PATH} ({len(df):,} rows, {len(df.columns)} columns)")
    print(f"Done: {PARQUET_PATH}  ({len(df):,} rows, {len(df.columns)} columns)")
    print("\nQuick sanity check:")
    print(df["borough"].value_counts().head(10))


if __name__ == "__main__":
    main()