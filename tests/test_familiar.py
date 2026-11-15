import numpy as np

from planscope.familiar import MIN_COVERAGE, MIN_ENERGY, coverage, energy, is_familiar


def test_energy_is_logsumexp_even_for_logits_too_large_for_exp():
    assert np.isclose(energy(np.array([1.0, 2.0, 3.0])), np.log(np.exp([1.0, 2.0, 3.0]).sum()))
    assert np.isclose(energy(np.array([1000.0, 1000.0])), 1000 + np.log(2))


def test_coverage_counts_known_words_and_is_zero_without_words():
    vocabulary = {'rear', 'extension'}
    assert coverage('Rear extension to the garden', vocabulary) == 2 / 5
    assert coverage('😀 1 x', vocabulary) == 0.0


def test_familiar_only_when_both_checks_pass():
    vocabulary = {'rear', 'extension', 'roof'}
    strong, weak = np.array([MIN_ENERGY, -5.0]), np.array([MIN_ENERGY - 1, -5.0])
    assert MIN_COVERAGE <= 2 / 3
    assert is_familiar('rear roof extension', strong, vocabulary)
    assert not is_familiar('rear roof extension', weak, vocabulary)
    assert not is_familiar('rear asdf qwerty', strong, vocabulary)