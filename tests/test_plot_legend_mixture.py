"""Legend under a mixture/matrix hue, plus `legend_kwargs=`/`legend_colors=`.

GH #285. `plot.py` used to answer a `legend=` under a matrix-valued hue with
"legend is not supported for continuous or matrix-valued hue; ignoring
legend", which left `examples/animate_market_sectors.py` hand-building six
`Line2D` proxy handles plus a grey '#666666' "Market" entry. The palette
entries the per-observation blend is made OF are a perfectly good key, so
hypertools now builds one swatch per hue column itself.

Handles and their colours are read back off the real legend artist.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.plot.colors import get_palette_colors


def _datasets(n=2, rows=30, cols=5, seed=4):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.normal(size=(rows, cols)), axis=0) for _ in range(n)]


def _mixture_hue(n_obs, n_columns=3, seed=5):
    rng = np.random.default_rng(seed)
    weights = rng.random((n_obs, n_columns))
    return weights / weights.sum(axis=1, keepdims=True)


def _legend(fig):
    return fig.axes[0].get_legend()


def _entries(fig):
    leg = _legend(fig)
    if leg is None:
        return []
    return [(t.get_text(), matplotlib.colors.to_hex(h.get_color()))
            for t, h in zip(leg.get_texts(), leg.legend_handles)]


# --- mixture / matrix hue gets a legend ---------------------------------

def test_matrix_hue_builds_one_swatch_per_column():
    data = _datasets(2, rows=30)
    hue = _mixture_hue(60, 3)
    fig = hyp.plot(data, hue=hue, legend=True, reduce='PCA', show=False)
    entries = _entries(fig)
    assert [label for label, _ in entries] == ['1', '2', '3']
    want = [matplotlib.colors.to_hex(c)
            for c in get_palette_colors('hls', 3)]
    assert [color for _, color in entries] == want


def test_matrix_hue_legend_list_names_the_columns():
    data = _datasets(2, rows=30)
    hue = _mixture_hue(60, 3)
    fig = hyp.plot(data, hue=hue, legend=['tech', 'energy', 'health'],
                   reduce='PCA', show=False)
    assert [label for label, _ in _entries(fig)] == ['tech', 'energy',
                                                     'health']


def test_matrix_hue_legend_list_length_must_match_the_columns():
    data = _datasets(2, rows=30)
    hue = _mixture_hue(60, 3)
    with pytest.raises(ValueError, match='hue= has 3 columns'):
        hyp.plot(data, hue=hue, legend=['a', 'b'], reduce='PCA', show=False)


def test_dataframe_hue_uses_its_own_column_names():
    data = _datasets(2, rows=30)
    hue = pd.DataFrame(_mixture_hue(60, 3),
                       columns=['tech', 'energy', 'health'])
    fig = hyp.plot(data, hue=hue, legend=True, reduce='PCA', show=False)
    assert [label for label, _ in _entries(fig)] == ['tech', 'energy',
                                                     'health']


def test_marker_style_matrix_hue_also_gets_a_legend():
    """The other matrix-hue branch (markers/animated), which quantized the
    blend into groups and dropped the legend with its own warning."""
    data = _datasets(2, rows=30)
    hue = _mixture_hue(60, 3)
    fig = hyp.plot(data, 'o', hue=hue, legend=True, reduce='PCA',
                   show=False)
    assert [label for label, _ in _entries(fig)] == ['1', '2', '3']


def test_matrix_hue_legend_no_longer_warns():
    import warnings
    data = _datasets(2, rows=30)
    hue = _mixture_hue(60, 3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hyp.plot(data, hue=hue, legend=True, reduce='PCA', show=False)
    assert not [w for w in caught
                if 'legend is not supported' in str(w.message)]


def test_continuous_hue_still_drops_the_legend_with_a_warning():
    """A 1-D continuous hue genuinely has no finite key -- the warning (and
    a pointer at colorbar=) is still the right answer."""
    data = _datasets(2, rows=30)
    with pytest.warns(UserWarning, match='legend is not supported'):
        fig = hyp.plot(data, hue=np.linspace(0, 1, 60), legend=True,
                       reduce='PCA', show=False)
    assert _legend(fig) is None


def test_rgb_matrix_hue_still_drops_the_legend():
    data = _datasets(2, rows=30)
    hue = np.tile(np.array([[0.9, 0.1, 0.1]]), (60, 1))
    with pytest.warns(UserWarning, match='legend is not supported'):
        fig = hyp.plot(data, hue=hue, hue_mode='rgb', legend=True,
                       reduce='PCA', show=False)
    assert _legend(fig) is None


# --- legend_kwargs= -----------------------------------------------------

def test_legend_kwargs_override_the_defaults():
    data = _datasets(3, rows=20)
    fig = hyp.plot(data, names=['a', 'b', 'c'], legend=True, reduce='PCA',
                   show=False,
                   legend_kwargs={'loc': 'upper left', 'frameon': True,
                                  'ncol': 3})
    leg = _legend(fig)
    assert leg.get_frame_on() is True
    assert leg._ncols == 3


def test_legend_kwargs_without_a_legend_do_nothing():
    data = _datasets(3, rows=20)
    fig = hyp.plot(data, reduce='PCA', show=False,
                   legend_kwargs={'loc': 'upper left'})
    assert _legend(fig) is None


def test_legend_kwargs_must_be_a_dict():
    with pytest.raises(TypeError, match='legend_kwargs'):
        hyp.plot(_datasets(2), legend=True, legend_kwargs=['loc'],
                 show=False)


# --- legend_colors= -----------------------------------------------------

def test_legend_colors_recolour_the_entries():
    data = _datasets(3, rows=20)
    fig = hyp.plot(data, names=['a', 'b', 'c'], legend=True, reduce='PCA',
                   show=False, legend_colors=['#ff0000', '#00ff00',
                                              '#0000ff'])
    assert [c for _, c in _entries(fig)] == ['#ff0000', '#00ff00', '#0000ff']


def test_legend_colors_length_must_match_the_entries():
    data = _datasets(3, rows=20)
    with pytest.raises(ValueError, match='legend_colors has 2 entries'):
        hyp.plot(data, names=['a', 'b', 'c'], legend=True, reduce='PCA',
                 show=False, legend_colors=['#ff0000', '#00ff00'])


def test_legend_colors_pairs_replace_the_legend_outright():
    """The market-sectors shape: per-sector swatches plus a grey 'Market'
    entry that no drawn trace corresponds to."""
    data = _datasets(2, rows=30)
    hue = _mixture_hue(60, 2)
    fig = hyp.plot(data, hue=hue, legend=True, reduce='PCA', show=False,
                   legend_colors=[('tech', '#E4572E'), ('energy', '#17BEBB'),
                                  ('Market', '#666666')])
    assert _entries(fig) == [('tech', '#e4572e'), ('energy', '#17bebb'),
                             ('Market', '#666666')]


def test_legend_colors_pairs_work_without_any_hue():
    data = _datasets(2, rows=20)
    fig = hyp.plot(data, reduce='PCA', show=False,
                   legend_colors=[('one', '#111111'), ('two', '#222222')])
    assert _entries(fig) == [('one', '#111111'), ('two', '#222222')]


def test_matrix_hue_swatches_recoloured_by_a_plain_list():
    """`legend_colors=` as plain colours recolours whatever entries would
    have been drawn -- including the matrix-hue palette swatches."""
    data = _datasets(2, rows=30)
    hue = _mixture_hue(60, 2)
    fig = hyp.plot(data, hue=hue, legend=['tech', 'energy'], reduce='PCA',
                   show=False, legend_colors=['#aaaaaa', '#bbbbbb'])
    assert _entries(fig) == [('tech', '#aaaaaa'), ('energy', '#bbbbbb')]


def test_mixing_colours_and_pairs_raises():
    with pytest.raises(ValueError, match='mixes plain colors'):
        hyp.plot(_datasets(2), legend=True, show=False,
                 legend_colors=['red', ('two', '#222222')])


def test_empty_legend_colors_raises():
    with pytest.raises(ValueError, match='empty'):
        hyp.plot(_datasets(2), legend=True, show=False, legend_colors=[])


def test_legend_colors_must_be_a_sequence():
    with pytest.raises(TypeError, match='legend_colors'):
        hyp.plot(_datasets(2), legend=True, show=False, legend_colors='red')


def test_marker_only_style_gets_marker_swatches():
    data = _datasets(2, rows=20)
    fig = hyp.plot(data, 'o', reduce='PCA', show=False,
                   legend_colors=[('one', '#111111'), ('two', '#222222')])
    handles = _legend(fig).legend_handles
    assert all(h.get_marker() == 'o' for h in handles)
    assert all(h.get_linestyle() == 'None' for h in handles)


# --- plotly parity ------------------------------------------------------

def test_matrix_hue_legend_under_plotly():
    pytest.importorskip('plotly')
    data = _datasets(2, rows=30)
    hue = _mixture_hue(60, 3)
    fig = hyp.plot(data, hue=hue, legend=['tech', 'energy', 'health'],
                   reduce='PCA', backend='plotly', show=False)
    names = [t.name for t in fig.data if t.showlegend]
    assert names == ['tech', 'energy', 'health']
    assert fig.layout.showlegend is True


def test_legend_colors_pairs_under_plotly():
    pytest.importorskip('plotly')
    data = _datasets(2, rows=20)
    fig = hyp.plot(data, reduce='PCA', backend='plotly', show=False,
                   legend_colors=[('one', '#111111'), ('two', '#222222')])
    entries = [(t.name, matplotlib.colors.to_hex(
        tuple(int(v) / 255 for v in t.line.color[5:-1].split(',')[:3])))
        for t in fig.data if t.showlegend]
    assert entries == [('one', '#111111'), ('two', '#222222')]


def test_legend_kwargs_under_plotly():
    pytest.importorskip('plotly')
    data = _datasets(3, rows=20)
    fig = hyp.plot(data, names=['a', 'b', 'c'], legend=True, reduce='PCA',
                   backend='plotly', show=False,
                   legend_kwargs={'orientation': 'h', 'x': 0.1})
    assert fig.layout.legend.orientation == 'h'
    assert fig.layout.legend.x == 0.1


def test_plain_legend_colors_are_refused_under_plotly():
    pytest.importorskip('plotly')
    data = _datasets(3, rows=20)
    with pytest.raises(NotImplementedError, match='plotly'):
        hyp.plot(data, names=['a', 'b', 'c'], legend=True, reduce='PCA',
                 backend='plotly', show=False,
                 legend_colors=['#ff0000', '#00ff00', '#0000ff'])
