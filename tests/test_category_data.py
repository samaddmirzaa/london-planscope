import pandas as pd

from planscope.category_data import code_labels, labelled_texts, reference_code


def test_reference_code_takes_trailing_letters():
    refs = pd.Series(['19/3771/HSE', '20/01289/FULL6', '185575FUL', '2019/1464'])
    assert reference_code(refs).tolist()[:3] == ['HSE', 'FULL', 'FUL']
    assert pd.isna(reference_code(refs)[3])


def test_labelled_texts_merges_variants_and_keeps_majority_category():
    df = pd.DataFrame({
        'description': ['Rear dormer', 'rear dormer.', 'Rear dormer', 'Legal agreement'],
        'category': ['householder', 'householder', 'full_planning', 'other'],
    })
    assert labelled_texts(df).to_dict('records') == [{'text': 'rear dormer', 'category': 'householder'}]


def test_code_labels_uses_only_reliable_codes_on_unlabelled_rows():
    labelled = pd.DataFrame({
        'lpa_name': ['Barnet'] * 60,
        'lpa_app_no': ['1/HSE'] * 30 + ['2/MIX'] * 30,
        'category': ['householder'] * 30 + ['householder'] * 15 + ['full_planning'] * 15,
        'description': ['x'] * 60,
    })
    unlabelled = pd.DataFrame({
        'lpa_name': ['Barnet'] * 2,
        'lpa_app_no': ['3/HSE', '4/MIX'],
        'category': [None, None],
        'description': ['Single storey rear extension', 'New shopfront'],
    })
    out = code_labels(pd.concat([labelled, unlabelled], ignore_index=True))
    assert out[['text', 'category']].to_dict('records') == [{'text': 'single storey rear extension', 'category': 'householder'}]