import pandas as pd

from planscope.map_data import authority_name, borough_stats


def test_ons_names_match_datahub_names():
    names = ['Kingston upon Thames', 'Richmond upon Thames', 'Barking and Dagenham', 'City of London']
    assert [authority_name(n) for n in names] == ['Kingston', 'Richmond', 'Barking & Dagenham', 'City of London']


def test_borough_stats_counts_each_category_and_all():
    df = pd.DataFrame({
        'id': ['a', 'b', 'c', 'd'],
        'lpa_name': ['Camden'] * 3 + ['Brent'],
        'outcome': ['APPROVED', 'REFUSED', None, 'WITHDRAWN'],
        'category': ['householder', 'householder', None, 'trees'],
    })
    stats = borough_stats(df)
    assert stats['Camden'] == {'all': [3, 1, 1], 'householder': [2, 1, 1]}
    assert stats['Brent'] == {'all': [1, 0, 0], 'trees': [1, 0, 0]}