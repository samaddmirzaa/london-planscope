import pandas as pd

from planscope.category_data import normalise
from planscope.classifier import prepare
from planscope.clean import clean_text


def test_prepare_matches_how_training_text_was_prepared():
    texts = ['  Rear \n dormer\x1f window. ', 'SINGLE STOREY REAR EXTENSION.', 'Prune 1 x Oak (T1)']
    assert [prepare(text) for text in texts] == normalise(clean_text(pd.Series(texts))).tolist()