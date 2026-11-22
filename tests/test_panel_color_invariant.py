"""The `panels=` color invariant (1.2.0 review round 3): every artist a
one-panel-per-dataset grid draws for dataset i has the color the equivalent
single-axes call of ALL the datasets gives that dataset's rows.

Each case draws both real figures and reads the real artists back. The
first data column is the row's global index, so every drawn vertex can be
traced to its row whatever regrouping (`hue=` categories, clusters, runs)
either call applied: hypertools' display rescale is affine per axis, so
the drawn x of a row is an affine function of its index in either figure.
"""
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba, to_rgba_array
from sklearn.base import BaseEstimator, TransformerMixin

import hypertools as hyp
from hypertools.plot.plot import SHARED_COLORBAR_GID, SHARED_LEGEND_GID

ROWS = 12
N_DATASETS = 3
TOTAL = ROWS * N_DATASETS


@pytest.fixture(autouse=True)
def _close_figures():
    """Leave no figure behind."""
    yield
    plt.close('all')


def _datasets(width=2):
    """Three datasets whose first column is the global row index and whose
    other columns separate the rows into clusterable blocks."""
    rng = np.random.default_rng(7)
    out = []
    for i in range(N_DATASETS):
        index = np.arange(i * ROWS, (i + 1) * ROWS, dtype=float)
        columns = [index]
        for k in range(width - 1):
            columns.append(np.sin(index / 3.0 + k)
                           + 0.05 * rng.normal(size=ROWS))
        out.append(np.column_stack(columns))
    return out


def _x_of(artist):
    """The drawn x coordinates of a line or scatter artist (2-D or 3-D)."""
    if hasattr(artist, 'get_data_3d'):
        return np.asarray(artist.get_data_3d()[0], dtype=float)
    if hasattr(artist, '_offsets3d'):
        return np.asarray(artist._offsets3d[0], dtype=float)
    if isinstance(artist, PathCollection):
        return np.asarray(artist.get_offsets(), dtype=float)[:, 0]
    return np.asarray(artist.get_xdata(), dtype=float)


def _segments_of(collection):
    """A line collection's segments as ``(x0, x1)`` pairs (2-D or 3-D)."""
    segments = getattr(collection, '_segments3d', None)
    if segments is None:
        segments = collection.get_segments()
    return [(float(np.asarray(seg)[0][0]), float(np.asarray(seg)[-1][0]))
            for seg in segments if len(seg) >= 2]


def _marks(ax):
    """Every colored mark `ax` draws, as ``(kind, x0, x1, rgb)``: a
    ``'marker'`` at ``x0`` (``x1`` is None) or a line ``'segment'`` from
    ``x0`` to ``x1``."""
    marks = []
    for line in ax.lines:
        if not line.get_visible():
            continue
        xs = _x_of(line)
        if xs.size == 0:
            continue
        if line.get_linestyle() not in ('None', '', ' ', None) \
                and xs.size >= 2:
            rgb = to_rgba(line.get_color())[:3]
            marks.extend(('segment', float(a), float(b), rgb)
                         for a, b in zip(xs[:-1], xs[1:]))
        if line.get_marker() not in ('None', '', ' ', None):
            rgb = to_rgba(line.get_markerfacecolor())[:3]
            marks.extend(('marker', float(a), None, rgb) for a in xs)
    for coll in ax.collections:
        if not coll.get_visible():
            continue
        if isinstance(coll, LineCollection):
            segments = _segments_of(coll)
            if not segments:
                continue
            colors = to_rgba_array(coll.get_colors())
            if len(colors) == 1 and len(
                    {round(abs(x), 9) for seg in segments for x in seg}) == 1:
                # a face of the 3-D cube frame (a wireframe whose every
                # vertex sits on the cube's +/- half-width), not data
                continue
            if len(colors) == 1:
                colors = np.repeat(colors, len(segments), axis=0)
            marks.extend(('segment', a, b, tuple(c[:3]))
                         for (a, b), c in zip(segments, colors))
        elif isinstance(coll, PathCollection):
            xs = _x_of(coll)
            if xs.size == 0:
                continue
            # (a 3-D scatter's `get_facecolor()` is depth-sorted to the
            # current view; the colors it was given are `_facecolors`)
            colors = to_rgba_array(coll._facecolors
                                   if hasattr(coll, '_offsets3d')
                                   else coll.get_facecolor())
            if len(colors) == 1:
                colors = np.repeat(colors, xs.size, axis=0)
            marks.extend(('marker', float(a), None, tuple(c[:3]))
                         for a, c in zip(xs, colors))
    return marks


def _row_colors(ax, first_row, last_row):
    """``{(kind, row): sorted colors}`` for the rows `ax` draws, which run
    from `first_row` to `last_row` (global indices): a marker belongs to
    the row it sits on, a line segment to every row it starts at or
    passes."""
    marks = _marks(ax)
    assert marks, 'the axes draws nothing'
    xs = [m[1] for m in marks] + [m[2] for m in marks if m[2] is not None]
    lo, hi = min(xs), max(xs)
    assert hi > lo
    scale = (last_row - first_row) / (hi - lo)
    eps = 1e-6
    out = {}
    for kind, x0, x1, rgb in marks:
        a = first_row + (x0 - lo) * scale
        color = tuple(round(float(v), 3) for v in rgb)
        if kind == 'marker':
            if abs(a - round(a)) < 1e-4:
                out.setdefault((kind, int(round(a))), set()).add(color)
            continue
        b = first_row + (x1 - lo) * scale
        if b <= a:
            continue
        for row in range(math.ceil(a - eps), math.ceil(b - eps)):
            out.setdefault((kind, row), set()).add(color)
    return {key: sorted(value) for key, value in out.items()}


def _draw(data, kwargs):
    """`hyp.plot` of `data`, quietly; an animation is advanced to its last
    frame (every row revealed) and its figure returned."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = hyp.plot(data, show=False, **kwargs)
    if hasattr(out, 'draw_frame'):
        out.draw_frame(out.n_frames - 1)
        return out.figure
    return out


def _mixed_categories():
    """Categories spread unevenly over the panels: the first panel holds
    one, the others two each, never the same pair."""
    return (['a'] * ROWS + ['a'] * 5 + ['b'] * (ROWS - 5)
            + ['c'] * 7 + ['b'] * (ROWS - 7))


def _rgb_matrix():
    """An RGB hue whose value range differs between the panels."""
    ramp = np.linspace(0.0, 2.0, TOTAL)
    return np.column_stack([ramp, ramp[::-1] ** 2, np.cos(ramp)])


def _wide_matrix():
    """A five-column hue matrix (reduced to RGB)."""
    rng = np.random.default_rng(3)
    return rng.normal(size=(TOTAL, 5)).cumsum(axis=0)


def _mixture(signed=False):
    """Mixture weights over three components; `signed` puts negative
    entries in the last panel's rows only."""
    rng = np.random.default_rng(5)
    weights = rng.dirichlet([1.0, 1.0, 1.0], size=TOTAL)
    if signed:
        weights[-ROWS:] -= 0.4
    return weights


def _multiindex_frames(width=2):
    """The datasets as row-MultiIndex frames (flattened in a list)."""
    index = pd.MultiIndex.from_product([['u', 'v'], range(ROWS // 2)])
    return [pd.DataFrame(d, index=index) for d in _datasets(width)]


#: ``id -> (kwargs, data factory or None)``; every case is drawn with each
#: of `FMTS` unless it names its own `fmt=`
CASES = {
    'plain': {},
    'palette-name': {'palette': 'viridis'},
    'palette-list': {'palette': ['#111111', '#aa3300', '#0055ff']},
    'color-list': {'color': ['k', 'tab:orange', 'teal']},
    'color-tuple': {'color': (0.2, 0.4, 0.6)},
    'fmt-letters': {'fmt': ['r-', '-', 'g-']},
    'legend': {'legend': True},
    'names': {'names': ['one', 'two', 'three']},
    'hue-continuous': {'hue': np.linspace(-1.0, 4.0, TOTAL) ** 2},
    'hue-continuous-colorbar': {'hue': np.linspace(0.0, 1.0, TOTAL),
                                'colorbar': True, 'palette': 'magma'},
    'hue-continuous-nan': {'hue': np.r_[np.nan, np.linspace(0, 1,
                                                             TOTAL - 1)]},
    'hue-categorical': {'hue': _mixed_categories()},
    'hue-categorical-legend': {'hue': _mixed_categories(), 'legend': True,
                               'colorbar': True},
    'hue-categorical-palette-map': {
        'hue': _mixed_categories(),
        'palette': {'a': 'red', 'b': 'blue', 'c': '#00aa00'}},
    'hue-categorical-duplicate-legend': {
        'hue': _mixed_categories(),
        'palette': {'a': 'red', 'b': 'blue', 'c': 'green'},
        'legend': ['same', 'same', 'other'], 'colorbar': True},
    'hue-categorical-renamed': {
        'hue': _mixed_categories(), 'legend': ['x', 'y', 'z']},
    'hue-partial-labels': {
        'hue': [None if i % 5 == 0 else label
                for i, label in enumerate(_mixed_categories())]},
    'hue-integers': {'hue': np.repeat([3, 1, 2, 1, 3, 2], TOTAL // 6)},
    'hue-many-integers': {'hue': np.arange(TOTAL)},
    'hue-per-dataset': {'hue': ['p', 'q', 'p']},
    'hue-nested': {'hue': [['a'] * ROWS, ['b'] * ROWS,
                           ['a'] * 6 + ['c'] * (ROWS - 6)]},
    'hue-rgb': {'hue': _rgb_matrix(), 'hue_mode': 'rgb'},
    'hue-wide-matrix': {'hue': _wide_matrix()},
    'hue-wide-matrix-pca': {'hue': _wide_matrix(), 'color_reduce': 'PCA'},
    'hue-mixture': {'hue': _mixture(), 'palette': ['r', 'g', 'b']},
    'hue-mixture-signed': {'hue': _mixture(signed=True)},
    'hue-mixture-legend': {'hue': _mixture(), 'legend': True},
    'cluster-kmeans': {'cluster': 'KMeans', 'n_clusters': 4,
                       'random_state': 0},
    'cluster-kmeans-legend': {'n_clusters': 3, 'random_state': 1,
                              'legend': ['k1', 'k2', 'k3'],
                              'colorbar': True},
    'cluster-mixture': {'cluster': 'GaussianMixture', 'n_clusters': 3,
                        'random_state': 0},
    'multiindex-frames': ({'legend': True, 'colorbar': True},
                          _multiindex_frames),
    'font': {'hue': np.linspace(0.0, 1.0, TOTAL), 'colorbar': True,
             'font': 'DejaVu Sans Mono'},
}

FMTS = ('-', 'o', 'o-')
MODES = {
    'static': {},
    'static-raw': {'antialias': False},
    'spin': {'animate': 'spin', 'duration': 1, 'frame_rate': 4,
             '_width': 3},
    'reveal': {'animate': True, 'duration': 1, 'frame_rate': 4,
               'tail_duration': 1},
    'independent': {'panel_fit': 'independent'},
    '3d': {'_width': 3},
}


def _grid_cases():
    """Every (case, fmt, mode) drawn by `test_panel_colors_are_the_single`
    `_axes_call_s`: all cases statically with every fmt; the animated,
    independent-fit and 3-D modes with lines and with markers."""
    params = []
    for name, spec in CASES.items():
        kwargs, factory = spec if isinstance(spec, tuple) else (spec, None)
        fmts = (None,) if 'fmt' in kwargs else FMTS
        for mode in MODES:
            for fmt in fmts:
                if mode != 'static' and fmt == 'o-':
                    continue
                if mode == 'independent' and ('cluster' in kwargs
                                              or 'n_clusters' in kwargs):
                    # every panel then fits clusters of its own: there is
                    # no single-axes call the grid is equivalent to
                    continue
                params.append(pytest.param(
                    kwargs, factory, fmt, mode,
                    id=f'{name}/{fmt or "own"}/{mode}'))
    return params


@pytest.mark.parametrize('kwargs,factory,fmt,mode', _grid_cases())
def test_panel_colors_are_the_single_axes_calls(kwargs, factory, fmt, mode):
    mode_kwargs = dict(MODES[mode])
    width = mode_kwargs.pop('_width', 2)
    data = (factory or _datasets)(width)
    call = dict(kwargs, reduce=None, ndims=width, **mode_kwargs)
    if fmt is not None:
        call['fmt'] = fmt
    single_kwargs = {k: v for k, v in call.items() if k != 'panel_fit'}
    single = _draw(data, single_kwargs)
    grid = _draw(data, dict(call, panels=True))

    # (the data axes come first; a colorbar's axes follow them)
    want = _row_colors(single.axes[0], 0, TOTAL - 1)
    panels = grid.axes[:N_DATASETS]
    assert all(ax.get_gid() != SHARED_COLORBAR_GID for ax in panels)
    for i, ax in enumerate(panels):
        first, last = i * ROWS, (i + 1) * ROWS - 1
        got = _row_colors(ax, first, last)
        expected = {key: value for key, value in want.items()
                    if first <= key[1] <= last}
        assert expected, f'the single-axes call draws nothing of dataset {i}'
        assert got == expected, (
            f'panel {i}: ' + '; '.join(
                f'{key}: grid {got.get(key)} != single {expected.get(key)}'
                for key in sorted(set(got) | set(expected))
                if got.get(key) != expected.get(key))[:600])


# --------------------------------------------------------------------------
# 1.2.0 review round 4: a matrix `hue=` is reduced ONCE per grid call, and
# its column names reach the shared legend.
# --------------------------------------------------------------------------

class _CountingRandomProjection(BaseEstimator, TransformerMixin):
    """A small real reducer (sklearn API): projects onto `n_components`
    random directions drawn from numpy's GLOBAL random state at fit time --
    so two fits under one seed differ, as a stochastic embedding's do --
    and counts every fit made by any instance in `fits`."""

    fits = 0

    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, X, y=None):
        """Draw the projection (one more fit)."""
        type(self).fits += 1
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.components_ = np.random.normal(
            size=(self.n_components, X.shape[1]))
        return self

    def transform(self, X):
        """Project the rows of `X`."""
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T


def _seeded_draw(data, kwargs, seed=11):
    """`_draw` with numpy's global random state reset to `seed` first, and
    the number of reducer fits the call made."""
    _CountingRandomProjection.fits = 0
    np.random.seed(seed)
    figure = _draw(data, kwargs)
    return figure, _CountingRandomProjection.fits


def _assert_panels_match_single(single, grid):
    """Every panel of `grid` draws its dataset's rows in the colors
    `single` (the single-axes figure of all the datasets) gives them."""
    want = _row_colors(single.axes[0], 0, TOTAL - 1)
    for i, ax in enumerate(grid.axes[:N_DATASETS]):
        first, last = i * ROWS, (i + 1) * ROWS - 1
        expected = {key: value for key, value in want.items()
                    if first <= key[1] <= last}
        assert expected
        assert _row_colors(ax, first, last) == expected, f'panel {i}'


REDUCED_HUES = {
    'counting-class': {'color_reduce': _CountingRandomProjection},
    'counting-spec': {'color_reduce': {
        'model': _CountingRandomProjection,
        'kwargs': {'n_components': 3}}},
}


@pytest.mark.parametrize('fit', ['shared', 'independent'])
@pytest.mark.parametrize('fmt', ['-', 'o'])
@pytest.mark.parametrize('mode', ['static', 'reveal'])
@pytest.mark.parametrize('spec', list(REDUCED_HUES))
def test_matrix_hue_is_reduced_once_per_grid(spec, mode, fmt, fit):
    call = dict(REDUCED_HUES[spec], hue=_wide_matrix(), reduce=None, ndims=2,
                fmt=fmt, **MODES[mode])
    single, single_fits = _seeded_draw(_datasets(), call)
    assert single_fits == 1
    grid, grid_fits = _seeded_draw(
        _datasets(), dict(call, panels=True, panel_fit=fit))
    assert grid_fits == 1, (
        f'the grid reduced its hue matrix {grid_fits} times')
    # ...so, under one seed, the grid's colors are the single-axes call's
    _assert_panels_match_single(single, grid)


@pytest.mark.parametrize('fit', ['shared', 'independent'])
def test_seeded_tsne_hue_colors_are_the_single_axes_calls(fit):
    call = dict(hue=_wide_matrix(), reduce=None, ndims=2, fmt='o',
                color_reduce={'model': 'TSNE',
                              'kwargs': {'perplexity': 5, 'init': 'random'}})
    single, _ = _seeded_draw(_datasets(), call)
    grid, _ = _seeded_draw(_datasets(),
                           dict(call, panels=True, panel_fit=fit))
    _assert_panels_match_single(single, grid)


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_reducer_list_grid_reduces_its_matrix_hue_once(backend):
    """A one-panel-per-REDUCER grid draws every dataset in every panel:
    one reduction of the hue matrix colors all of them alike."""
    call = dict(hue=_wide_matrix(), color_reduce=_CountingRandomProjection,
                reduce=['PCA', 'IncrementalPCA'], ndims=2, fmt='o',
                panels=True, backend=backend)
    grid, fits = _seeded_draw(_datasets(3), call)
    assert fits == 1
    if backend == 'matplotlib':
        first, second = (
            sorted(tuple(np.round(c, 6)) for coll in ax.collections
                   for c in to_rgba_array(coll.get_facecolor()))
            for ax in grid.axes[:2])
        assert first and first == second


def _legend_entries(figure):
    """The labels of the legend(s) `figure` shows: a matplotlib figure's
    one legend (the grid's shared one, or the single axes') -- asserting
    there is exactly one -- or a plotly figure's legend entries."""
    if hasattr(figure, 'data'):
        return [trace.name for trace in figure.data
                if trace.showlegend is not False and trace.name is not None]
    legends = list(figure.legends) + [
        ax.get_legend() for ax in figure.axes if ax.get_legend() is not None]
    assert len(legends) == 1, f'{len(legends)} legends'
    return [text.get_text() for text in legends[0].get_texts()]


def _named_mixture():
    """Mixture weights as a DataFrame whose columns are named."""
    return pd.DataFrame(_mixture(), columns=['signal', 'noise', 'drift'])


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
@pytest.mark.parametrize('fit', ['shared', 'independent'])
@pytest.mark.parametrize('fmt', ['-', 'o'])
def test_mixture_hue_column_names_reach_the_shared_legend(fmt, fit, backend):
    call = dict(hue=_named_mixture(), hue_mode='mixture', legend=True,
                reduce=None, ndims=2, fmt=fmt, backend=backend)
    single = _legend_entries(_draw(_datasets(), call))
    assert single == ['signal', 'noise', 'drift']
    grid = _draw(_datasets(), dict(call, panels=True, panel_fit=fit))
    assert _legend_entries(grid) == single
    if backend == 'matplotlib':
        assert grid.legends[0].get_gid() == SHARED_LEGEND_GID


def test_mixture_hue_column_names_reach_an_animated_grids_legend():
    call = dict(hue=_named_mixture(), hue_mode='mixture', legend=True,
                reduce=None, ndims=2, **MODES['reveal'])
    single = _legend_entries(_draw(_datasets(), call))
    assert single == ['signal', 'noise', 'drift']
    assert _legend_entries(_draw(_datasets(), dict(call, panels=True))) \
        == single


def test_mixture_hue_names_are_in_the_grids_color_scale():
    call = dict(hue=_named_mixture(), hue_mode='mixture', legend=True,
                reduce=None, ndims=2, return_model=True, show=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        single = hyp.plot(_datasets(), **call)
        grid = hyp.plot(_datasets(), panels=True, **call)
    assert list(single['colors']['categories']) == ['signal', 'noise',
                                                    'drift']
    assert list(grid['colors']['categories']) \
        == list(single['colors']['categories'])
