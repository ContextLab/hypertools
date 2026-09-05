"""Per-dataset `labels=` with `label_anchor=` (GH #285).

Two examples build an all-None per-observation label list with one string
dropped in per dataset -- `animate_painting_embeddings` at the MIDDLE
observation, `plot_labels` at the FIRST point. `labels=` of one entry per
dataset plus `label_anchor=` says that directly.

Positions are read back off the real annotation artists, so `center` is
verified to land on the same observation the hand-built list did.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import hypertools as hyp


def _datasets(n=3, rows=21, cols=5, seed=3):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.normal(size=(rows, cols)), axis=0) for _ in range(n)]


def _annotations(fig):
    """(text, anchor xy) for every point annotation, in draw order."""
    return [(t.get_text(), tuple(np.round(t.xy, 9)))
            for ax in fig.axes for t in ax.texts]


# --- the per-dataset form -----------------------------------------------

def test_one_label_per_dataset_annotates_each_dataset_once():
    fig = hyp.plot(_datasets(3), labels=['a', 'b', 'c'], reduce='PCA',
                   show=False)
    assert [t for t, _ in _annotations(fig)] == ['a', 'b', 'c']


def test_none_entries_annotate_nothing():
    fig = hyp.plot(_datasets(3), labels=['a', None, 'c'], reduce='PCA',
                   show=False)
    assert [t for t, _ in _annotations(fig)] == ['a', 'c']


@pytest.mark.parametrize('anchor,index', [
    (None, 0), ('first', 0), ('center', 10), ('last', 20), (7, 7), (-1, 20),
])
def test_label_anchor_matches_the_hand_built_list(anchor, index):
    """`label_anchor` must land on exactly the observation the equivalent
    hand-built per-observation list labels."""
    data = _datasets(3, rows=21)
    short = hyp.plot(data, labels=['a', 'b', 'c'], label_anchor=anchor,
                     reduce='PCA', show=False)
    hand = []
    for name in ('a', 'b', 'c'):
        column = [None] * 21
        column[index] = name
        hand.append(column)
    long = hyp.plot(data, labels=hand, reduce='PCA', show=False)
    assert _annotations(short) == _annotations(long)
    assert len(_annotations(short)) == 3


def test_center_is_the_middle_observation():
    data = _datasets(2, rows=11)
    fig = hyp.plot(data, labels=['x', 'y'], label_anchor='center',
                   reduce='PCA', show=False)
    # 11 rows -> index 5 (the true middle)
    hand = [[None] * 11, [None] * 11]
    hand[0][5], hand[1][5] = 'x', 'y'
    ref = hyp.plot(data, labels=hand, reduce='PCA', show=False)
    assert _annotations(fig) == _annotations(ref)


# --- the per-observation form is untouched ------------------------------

def test_flat_per_observation_labels_unchanged():
    data = _datasets(2, rows=10)
    labels = [None] * 20
    labels[3], labels[15] = 'p', 'q'
    fig = hyp.plot(data, labels=labels, reduce='PCA', show=False)
    assert [t for t, _ in _annotations(fig)] == ['p', 'q']


def test_nested_per_dataset_lists_unchanged():
    data = _datasets(2, rows=10)
    labels = [[None] * 10, [None] * 10]
    labels[0][0], labels[1][9] = 'start', 'end'
    fig = hyp.plot(data, labels=labels, reduce='PCA', show=False)
    assert [t for t, _ in _annotations(fig)] == ['start', 'end']


def test_one_row_per_dataset_keeps_the_per_observation_reading():
    data = [np.arange(5, dtype=float).reshape(1, 5),
            np.arange(5, 10, dtype=float).reshape(1, 5)]
    fig = hyp.plot(data, '.', labels=['a', 'b'], reduce='PCA', show=False)
    assert [t for t, _ in _annotations(fig)] == ['a', 'b']


# --- validation ---------------------------------------------------------

def test_label_anchor_with_per_observation_labels_raises():
    data = _datasets(2, rows=10)
    with pytest.raises(ValueError, match='label_anchor'):
        hyp.plot(data, labels=[None] * 20, label_anchor='center',
                 reduce='PCA', show=False)


def test_bad_label_anchor_raises():
    with pytest.raises(ValueError, match='label_anchor'):
        hyp.plot(_datasets(3), labels=['a', 'b', 'c'],
                 label_anchor='middle-ish', reduce='PCA', show=False)


def test_out_of_range_label_anchor_raises():
    with pytest.raises(ValueError, match='out of range'):
        hyp.plot(_datasets(3, rows=10), labels=['a', 'b', 'c'],
                 label_anchor=99, reduce='PCA', show=False)


def test_wrong_length_labels_still_raise_the_length_error():
    with pytest.raises(ValueError, match='one entry per observation'):
        hyp.plot(_datasets(3, rows=10), labels=['a', 'b'], reduce='PCA',
                 show=False)


# --- plotly parity ------------------------------------------------------

def test_per_dataset_labels_under_plotly():
    pytest.importorskip('plotly')
    data = _datasets(3, rows=21)
    fig = hyp.plot(data, labels=['a', 'b', 'c'], label_anchor='center',
                   reduce='PCA', backend='plotly', show=False)
    texts = sorted(a.text for a in fig.layout.scene.annotations)
    assert texts == ['a', 'b', 'c']


def test_plotly_anchor_matches_the_hand_built_list():
    pytest.importorskip('plotly')
    data = _datasets(2, rows=11)
    short = hyp.plot(data, labels=['x', 'y'], label_anchor='last',
                     reduce='PCA', backend='plotly', show=False)
    hand = [[None] * 11, [None] * 11]
    hand[0][10], hand[1][10] = 'x', 'y'
    long = hyp.plot(data, labels=hand, reduce='PCA', backend='plotly',
                    show=False)
    got = [(a.text, a.x, a.y, a.z) for a in short.layout.scene.annotations]
    want = [(a.text, a.x, a.y, a.z) for a in long.layout.scene.annotations]
    assert got == want
