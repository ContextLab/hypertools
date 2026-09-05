"""Per-dataset `hue=` broadcast (GH #285).

`hue=[[speaker] * len(window) for speaker, window in ...]` repeats one label
per row so that datasets sharing a speaker share a colour. `hue=speakers` --
one scalar per DATASET -- now means the same thing.

The disambiguation rule this file pins down: the broadcast fires only for a
list of exactly `len(x)` SCALAR entries, with more than one dataset, when
that length is not also the total observation count. Every other shape keeps
its historical meaning exactly.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import hypertools as hyp


def _datasets(n=3, rows=20, cols=5, seed=2):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.normal(size=(rows, cols)), axis=0) for _ in range(n)]


def _legend_texts(fig):
    leg = fig.axes[0].get_legend()
    return [] if leg is None else [t.get_text() for t in leg.get_texts()]


def _line_colors(fig):
    return [matplotlib.colors.to_hex(line.get_color())
            for line in fig.axes[0].lines]


# --- the broadcast ------------------------------------------------------

def test_scalar_per_dataset_hue_groups_by_category():
    data = _datasets(3)
    fig = hyp.plot(data, hue=['alice', 'bob', 'alice'], legend=True,
                   reduce='PCA', show=False)
    assert _legend_texts(fig) == ['alice', 'bob']


def test_broadcast_matches_the_explicit_nested_form_exactly():
    """The whole claim: `hue=speakers` must draw the same figure as the
    hand-rolled `hue=[[s] * len(d) for s, d in zip(speakers, data)]`."""
    data = _datasets(3, rows=20)
    speakers = ['alice', 'bob', 'alice']
    short = hyp.plot(data, hue=speakers, reduce='PCA', show=False)
    long = hyp.plot(data, hue=[[s] * len(d) for s, d in zip(speakers, data)],
                    reduce='PCA', show=False)
    assert _line_colors(short) == _line_colors(long)
    got = [np.asarray(line.get_data_3d()) for line in short.axes[0].lines]
    want = [np.asarray(line.get_data_3d()) for line in long.axes[0].lines]
    assert len(got) == len(want)
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b)


def test_datasets_sharing_a_value_share_a_colour():
    data = _datasets(4)
    fig = hyp.plot(data, hue=['x', 'y', 'x', 'y'], reduce='PCA', show=False)
    colors = _line_colors(fig)
    # two categories -> two drawn traces, two distinct colours
    assert len(set(colors)) == 2


def test_numeric_scalar_per_dataset_hue_broadcasts():
    data = _datasets(3)
    fig = hyp.plot(data, hue=[0.0, 5.0, 10.0], colorbar=True, reduce='PCA',
                   show=False)
    # a continuous hue over 60 observations, spanning the given values
    cbar_ax = [a for a in fig.axes if a is not fig.axes[0]]
    assert cbar_ax, 'no colorbar axes were created'


# --- the disambiguation rule --------------------------------------------

def test_nested_per_dataset_hue_is_untouched():
    data = _datasets(3, rows=20)
    hue = [['a'] * 20, ['b'] * 20, ['c'] * 20]
    fig = hyp.plot(data, hue=hue, legend=True, reduce='PCA', show=False)
    assert _legend_texts(fig) == ['a', 'b', 'c']


def test_flat_per_observation_hue_is_untouched():
    data = _datasets(3, rows=20)
    hue = ['a'] * 20 + ['b'] * 20 + ['c'] * 20
    fig = hyp.plot(data, hue=hue, legend=True, reduce='PCA', show=False)
    assert _legend_texts(fig) == ['a', 'b', 'c']


def test_single_dataset_length_one_hue_keeps_its_old_meaning():
    """`hyp.plot(x, hue=['a'])` on a many-row dataset stays the historical
    per-observation error, NOT a silent broadcast."""
    one = _datasets(1, rows=20)
    with pytest.raises(ValueError, match='one value'):
        hyp.plot(one, hue=['a'], reduce='PCA', show=False)


def test_one_row_per_dataset_keeps_the_per_observation_reading():
    """When len(hue) == n_datasets == n_observations the two readings are
    identical, so the existing flat path handles it and nothing changes."""
    data = [np.arange(5, dtype=float).reshape(1, 5),
            np.arange(5, 10, dtype=float).reshape(1, 5),
            np.arange(10, 15, dtype=float).reshape(1, 5)]
    fig = hyp.plot(data, '.', hue=['a', 'b', 'c'], legend=True,
                   reduce='PCA', show=False)
    assert _legend_texts(fig) == ['a', 'b', 'c']


def test_wrong_length_scalar_hue_still_raises_the_length_error():
    data = _datasets(3, rows=20)
    with pytest.raises(ValueError, match='exactly one'):
        hyp.plot(data, hue=['a', 'b'], reduce='PCA', show=False)


def test_matrix_hue_is_not_broadcast():
    """A (n_obs, k) matrix hue whose row count happens to equal the
    dataset count must stay a matrix, not be read as per-dataset."""
    data = _datasets(3, rows=1, cols=5)
    hue = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    fig = hyp.plot(data, '.', hue=hue, reduce='PCA', show=False)
    assert fig.axes[0].collections, 'matrix hue drew no per-point colours'


# --- plotly parity ------------------------------------------------------

def test_broadcast_hue_under_plotly():
    pytest.importorskip('plotly')
    data = _datasets(3, rows=20)
    fig = hyp.plot(data, hue=['alice', 'bob', 'alice'], legend=True,
                   reduce='PCA', backend='plotly', show=False)
    names = [t.name for t in fig.data if t.showlegend]
    assert names == ['alice', 'bob']
