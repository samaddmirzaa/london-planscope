import pandas as pd

from planscope.clean import (
    COLUMNS, clean, clean_text, parse_centroid, parse_date, parse_form, to_category, to_outcome,
)


def values(series):
    '''Series as a plain list, with every kind of missing value shown as None.'''
    return [None if pd.isna(v) else v for v in series]


def test_clean_text_tidies_whitespace_and_nulls_placeholders():
    text = pd.Series(['  Rear \n dormer\x1f window ', 'Unknown', '   ', None])
    assert values(clean_text(text)) == ['Rear dormer window', None, None, None]


def test_parse_date_reads_day_first_and_rejects_other_formats():
    dates = parse_date(pd.Series(['03/07/2025', '2025-07-03', None]))
    assert values(dates) == [pd.Timestamp('2025-07-03'), None, None]


def test_parse_centroid_handles_text_numbers_and_nulls_points_outside_london():
    coords = parse_centroid(pd.Series([
        '{"lon": -0.1, "lat": 51.5}',
        '{"lon": "8.6e-05", "lat": "51.5"}',
        '{"lon": -7.55716, "lat": 49.766807}',
    ]))
    assert coords.loc[0].tolist() == [-0.1, 51.5]
    assert coords.loc[1].tolist() == [8.6e-05, 51.5]
    assert coords.loc[2].isna().all()


def test_to_outcome_maps_spellings_and_keeps_undecided_null():
    decision = pd.Series(['Approved', 'REFUSED', 'ref', 'WDN', 'Split Decision', None])
    assert values(to_outcome(decision)) == ['APPROVED', 'REFUSED', 'REFUSED', 'WITHDRAWN', 'OTHER', None]


def test_to_category_uses_first_matching_rule():
    app_type = pd.Series([
        'Full planning & listed building consent',
        'Householder planning & listed building consent',
        'Listed building consent',
        'Tree works: Trees in conservation areas/subject to TPOs',
        'Legal Agreement',
        None,
    ])
    expected = ['full_planning', 'householder', 'listed_building', 'trees', 'other', None]
    assert values(to_category(app_type)) == expected


def test_parse_form_extracts_numbers_and_nulls_impossible_values():
    form = parse_form(pd.Series([
        '{"site_area": 0.03, "building_details": [{"no_storeys": "3"}, {"no_storeys": 5}], '
        '"residential_details": {"total_no_proposed_residential_units": 4, "total_no_existing_residential_units": 1}}',
        '{"site_area": 250, "building_details": [{"no_storeys": 909}]}',
        None,
    ]))
    assert form.loc[0].tolist() == [4, 1, 5, 0.03]
    assert form.loc[1, ['storeys', 'site_area_ha']].isna().all()
    assert form.loc[2].isna().all()


def test_clean_applies_window_and_date_rules():
    raw = pd.DataFrame({
        'id': ['a', 'b', 'c'],
        'lpa_name': ['Camden'] * 3,
        'lpa_app_no': ['1', '2', '3'],
        'last_updated': ['2026-11-11T03:25:41.822Z'] * 3,
        'valid_date': ['01/06/2018', '01/06/2020', '01/06/2021'],
        'decision_date': ['01/07/2018', '01/05/2020', '01/08/2021'],
        'decision': ['Approved', 'Refused', None],
        'decision_process': ['Delegated', '100041', 'Committee'],
        'application_type': ['Householder'] * 3,
        'application_type_full': ['Householder planning permission', None, 'Full planning permission'],
        'description': ['Rear dormer', 'New front porch', 'Unknown'],
        'centroid': ['{"lon": -0.1, "lat": 51.5}'] * 3,
        'application_details': [None, None, '{"site_area": 0.03}'],
        'uprn': [None] * 3,
        'url_planning_app': [None] * 3,
    })
    df = clean(raw)
    assert list(df.columns) == COLUMNS
    assert df['id'].tolist() == ['b', 'c']
    assert values(df['decision_date']) == [None, pd.Timestamp('2021-08-01')]
    assert values(df['decision_process']) == [None, 'Committee']
    assert values(df['description']) == ['New front porch', None]
    assert values(df['outcome']) == ['REFUSED', None]