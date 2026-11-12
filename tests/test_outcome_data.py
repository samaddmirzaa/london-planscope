import pandas as pd

from planscope.outcome_data import assign_split, build, strip_post_submission, to_bins


def test_post_submission_text_is_removed():
    text = pd.Series([
        'Rear extension. Linked cases: APP/U5360/W/22/3313698',
        'Loft conversion (AMENDED DESCRIPTION)',
        'Two storey side extension - amended plans received',
        'Fell one oak tree',
    ])
    assert strip_post_submission(text).tolist() == [
        'Rear extension.', 'Loft conversion', 'Two storey side extension -  received', 'Fell one oak tree',
    ]


def test_bins_label_intervals_and_missing():
    labels = to_bins(pd.Series([0, 3, None, 80]), [0, 1, 2, 5, 10, 50])
    assert labels.tolist() == ['(-inf, 0.0]', '(2.0, 5.0]', 'missing', '(50.0, inf]']


def test_split_leaves_the_last_year_out_and_keeps_order():
    dates = pd.Series(pd.to_datetime(['2020-01-01', '2023-06-01', '2024-06-01', '2025-01-01', '2025-12-31']))
    assert assign_split(dates).tolist() == ['train', 'val', 'test', 'recent', 'recent']


def test_build_keeps_only_decided_applications_with_text():
    apps = pd.DataFrame({
        'id': ['a', 'b', 'c', 'd'],
        'lpa_name': ['Camden'] * 4,
        'valid_date': pd.to_datetime(['2019-01-01'] * 3 + ['2024-12-31']),
        'outcome': ['REFUSED', 'WITHDRAWN', 'APPROVED', 'APPROVED'],
        'description': ['Roof terrace', 'Shed', None, 'Porch'],
        'category': [None] * 4,
        'units_proposed': [float('nan')] * 4, 'storeys': [2, None, None, None], 'site_area_ha': [float('nan')] * 4,
    })
    categories = pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'category': ['householder'] * 4})
    out = build(apps, categories)
    assert out['id'].tolist() == ['a']
    assert out.loc[0, ['refused', 'lpa_category', 'storeys']].tolist() == [1, 'Camden | householder', '(1.0, 2.0]']