from planscope.evaluate import bootstrap, macro_f1


def test_macro_f1_ignores_categories_absent_from_the_truth():
    y_true = ['trees', 'trees', 'advert', 'advert']
    y_pred = ['trees', 'listed_building', 'advert', 'advert']
    # listed_building never appears in y_true, so it is not averaged in as a 0
    assert round(macro_f1(y_true, y_pred), 4) == round((2 / 3 + 1) / 2, 4)


def test_bootstrap_interval_contains_the_score_and_is_repeatable():
    y_true = ['a', 'b'] * 50
    y_pred = ['a', 'b'] * 40 + ['b', 'a'] * 10
    score, low, high = bootstrap(macro_f1, y_true, y_pred, n_boot=200)
    assert low <= score <= high
    assert bootstrap(macro_f1, y_true, y_pred, n_boot=200) == (score, low, high)