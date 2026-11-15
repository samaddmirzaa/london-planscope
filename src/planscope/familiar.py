'''
Does a description look like the planning applications the models learned from?

Each threshold is the level that 99.5% of held-out real descriptions reach, so about 1% of real ones are flagged, against 100% of random strings
and 96% of off-topic news.
'''

import numpy as np

from planscope.risk import TOKEN, prepare

MIN_ENERGY = 3.113  # logsumexp of the classifier's logits
MIN_COVERAGE = 2 / 3  # share of words in the risk model's vocabulary


def energy(logits):
    '''logsumexp of one text's logits: high when the classifier responds strongly to something it knows.'''
    top = logits.max()
    return float(top + np.log(np.exp(logits - top).sum()))


def coverage(text, vocabulary):
    '''Share of the text's words seen in past applications (0 when there are no words).'''
    words = TOKEN.findall(prepare(text))
    return sum(word in vocabulary for word in words) / len(words) if words else 0.0


def is_familiar(text, logits, vocabulary):
    '''True unless either check falls below what 99.5% of real descriptions reach.'''
    return energy(logits) >= MIN_ENERGY and coverage(text, vocabulary) >= MIN_COVERAGE