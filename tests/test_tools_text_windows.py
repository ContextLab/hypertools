#!/usr/bin/env python
"""Tests for `hypertools.tools.text_windows` (GH #285).

The two reference implementations below are copied VERBATIM from the
hand-written windowers `text_windows` replaces --
`examples/animate_painting_embeddings.py` (``windows``, ~line 149) and
`examples/animate_conversation.py` (``windows``, ~line 137) -- and the
parity tests pin that converting those examples cannot change a single
window. `chunk` is likewise verbatim from `docs/tutorials/text.ipynb`.
"""

import numpy as np
import pytest

import hypertools as hyp
from hypertools.tools import text_windows


# --- the code text_windows replaces, copied verbatim -----------------------
def _painting_windows(text, size=10, step=1):
    """examples/animate_painting_embeddings.py"""
    words = text.split()
    return [' '.join(words[i:i + size])
            for i in range(0, max(1, len(words) - size + 1), step)]


def _conversation_windows(text, size=6, step=2, min_windows=3):
    """examples/animate_conversation.py"""
    words = text.split()
    n = len(words)
    size = max(1, min(size, n - min_windows + 1))
    step = step if (n - size) // step + 1 >= min_windows else 1
    return [' '.join(words[i:i + size]) for i in range(0, n - size + 1, step)]


def _chunk(s, count):
    """docs/tutorials/text.ipynb"""
    size = max(1, len(s) // count)
    return [s[i * size:(i + 1) * size] for i in range(count)]


CORPUS = ([' '.join(f'w{i}' for i in range(k)) for k in range(1, 40)]
          + ['one', '  spaced   out   words  here ',
             'the quick brown fox jumps over the lazy dog'])


@pytest.mark.parametrize('text', CORPUS)
def test_matches_the_painting_examples_windows(text):
    assert text_windows(text, size=10, step=1) == _painting_windows(text)


@pytest.mark.parametrize('text', CORPUS)
def test_matches_the_conversation_examples_windows(text):
    assert (text_windows(text, size=6, step=2, min_windows=3)
            == _conversation_windows(text))


def test_matches_the_text_tutorials_equal_chunks():
    article = 'abcdefghij ' * 137
    size = len(article) // 5
    assert (text_windows(article, size=size, step=size, unit='chars')
            == _chunk(article, 5))


def test_basic_word_windows_and_step():
    assert text_windows('the quick brown fox jumps', size=3) == [
        'the quick brown', 'quick brown fox', 'brown fox jumps']
    assert text_windows('the quick brown fox jumps', size=3, step=2) == [
        'the quick brown', 'brown fox jumps']
    assert text_windows('a b c d', size=2, step=2) == ['a b', 'c d']


def test_size_none_is_one_window_per_document():
    assert text_windows('a b c') == ['a b c']
    assert text_windows(['a b c', 'd e']) == [['a b c'], ['d e']]


def test_min_windows_shrinks_the_window_then_the_step():
    # a document too short for the requested window still yields a path
    assert text_windows('one two three', size=6, min_windows=3) == [
        'one', 'two', 'three']
    # the window fits, but the step would stride past min_windows
    assert text_windows('a b c d e', size=3, step=3, min_windows=3) == [
        'a b c', 'b c d', 'c d e']


def test_min_windows_is_best_effort_on_very_short_text():
    # only two words exist, so only two windows can: no padding is invented
    assert text_windows('one two', size=4, min_windows=5) == ['one', 'two']


def test_one_word_and_empty_text():
    assert text_windows('solo', size=10) == ['solo']
    assert text_windows('', size=3) == []
    assert text_windows('   \n  ', size=3) == []
    assert text_windows('', size=3, min_windows=4) == []
    assert text_windows(['', 'a b'], size=2) == [[], ['a b']]


def test_sentence_windows():
    text = ('First sentence here. Second one follows! Third one? '
            'And a fourth.')
    assert text_windows(text, size=1, unit='sentences') == [
        'First sentence here.', 'Second one follows!', 'Third one?',
        'And a fourth.']
    assert text_windows(text, size=3, unit='sentences') == [
        'First sentence here. Second one follows! Third one?',
        'Second one follows! Third one? And a fourth.']


def test_sentence_splitter_limits_are_the_documented_ones():
    # abbreviations split -- the regex knows no abbreviation dictionary
    assert text_windows('Dr. Smith went home. He slept.', size=1,
                        unit='sentences') == ['Dr.', 'Smith went home.',
                                              'He slept.']
    # a terminator followed by a closing quote does NOT split there
    assert text_windows('"Stop." He left.', size=1, unit='sentences') == [
        '"Stop." He left.']
    # no whitespace after the terminator, no split
    assert text_windows('a.b.c', size=1, unit='sentences') == ['a.b.c']


def test_char_windows_preserve_the_raw_string():
    assert text_windows('abcdef', size=2, step=2, unit='chars') == [
        'ab', 'cd', 'ef']
    assert text_windows('a  b', size=2, unit='chars') == ['a ', '  ', ' b']


def test_max_chars_truncates_before_windowing():
    article = 'word ' * 100
    assert text_windows(article, max_chars=10) == ['word word']
    assert text_windows(article, size=2, max_chars=25) == [
        'word word', 'word word', 'word word', 'word word']
    # the Wikipedia tutorial's 2,000-character cut
    long_article = 'x' * 5000
    assert text_windows(long_article, max_chars=2000) == ['x' * 2000]


def test_list_input_gives_one_trajectory_per_document():
    docs = text_windows(['one two three four', 'five six'], size=2)
    assert docs == [['one two', 'two three', 'three four'], ['five six']]
    assert text_windows(('a b', 'c d'), size=1) == [['a', 'b'], ['c', 'd']]


@pytest.mark.parametrize('kwargs', [
    {'size': 0}, {'size': -1}, {'step': 0}, {'min_windows': 0},
    {'max_chars': 0}, {'size': 2.5}, {'step': 'two'},
])
def test_invalid_numeric_arguments(kwargs):
    with pytest.raises((ValueError, TypeError)):
        text_windows('a b c d', **kwargs)


def test_invalid_unit_and_input_type():
    with pytest.raises(ValueError, match='unit='):
        text_windows('a b c', size=2, unit='paragraphs')
    with pytest.raises(TypeError, match='string'):
        text_windows(42, size=2)
    with pytest.raises(TypeError, match='string'):
        text_windows([['a b'], ['c d']], size=1)


def test_windows_plot_as_one_trajectory_per_document():
    text = ('the quick brown fox jumps over the lazy dog while the dog '
            'sleeps under a warm summer sun and the fox runs across the '
            'wide green field chasing shadows through the tall grass') * 3
    windows = text_windows(text, size=5, step=2)
    assert len(windows) > 10

    fig = hyp.plot(windows, vectorizer='CountVectorizer', semantic=None,
                   reduce='PCA', ndims=3, show=False)
    assert len(fig.axes[0].lines) == 1

    docs = text_windows([text, text[:200]], size=5, step=2)
    fig = hyp.plot(docs, vectorizer='CountVectorizer', semantic=None,
                   reduce='PCA', ndims=3, show=False)
    assert len(fig.axes[0].lines) == 2
    assert all(np.asarray(len(d)) > 1 for d in docs)
