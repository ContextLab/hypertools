"""Palette ORDER and DATA-MATRIX palettes (Jeremy, 2026-09-08): colors
extracted from an image are put in a deterministic order (by value, dark to
bright) when they become a plot palette, and a t x k data matrix passed as a
palette is reduced to 3-D with `hypertools.reduce`, scaled, sorted and
resampled to however many colors the plot needs. Real images, real reducers,
real figures on both backends; no mocks."""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import Colormap, rgb_to_hsv, to_hex
from PIL import Image

import hypertools as hyp
from hypertools.plot.colors import (
    MatrixColormap, PALETTE_SORT_KEYS, get_palette_colors, image_palette,
    is_palette_matrix, luminance, matrix_palette, palette_lead_color,
    sort_colors)
from hypertools.plot.plotly_backend import _rgb_triplet


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close('all')


# --- fixtures ---------------------------------------------------------------

COLORS = {                        # name: RGB in 0..255
    'navy': (20, 30, 120), 'gold': (240, 200, 30), 'teal': (20, 150, 140),
    'crimson': (200, 30, 60), 'ivory': (245, 240, 220), 'coal': (25, 25, 25),
}


def painting_png(tmp_path, name='painting.png'):
    """Six blocks of very different size: a salience order that is NOT a
    value order, so the two contracts are distinguishable."""
    rng = np.random.default_rng(0)
    canvas = np.zeros((120, 120, 3), dtype=np.uint8)
    canvas[:] = COLORS['ivory']                     # big muted background
    canvas[:40, :40] = COLORS['coal']
    canvas[40:60, :60] = COLORS['navy']
    canvas[60:120, :30] = COLORS['teal']
    canvas[100:120, 100:120] = COLORS['crimson']    # small and vivid
    canvas[10:20, 100:110] = COLORS['gold']         # tiny and vivid
    canvas = np.clip(canvas.astype(int) + rng.integers(-2, 3, canvas.shape),
                     0, 255).astype(np.uint8)
    path = tmp_path / name
    Image.fromarray(canvas).save(path)
    return str(path)


def matrix(t=40, k=12, seed=0):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(size=(t, 3)), axis=0)   # a 3-D trajectory
    mix = rng.normal(size=(3, k))
    return base @ mix + 0.05 * rng.normal(size=(t, k))


def _walk(n=40, seed=1):
    return np.cumsum(np.random.default_rng(seed).normal(size=(n, 3)), axis=0)


def _mpl_segment_colors(fig):
    """The per-segment colors of a continuous-hue trajectory (matplotlib
    draws it as Line3DCollections, one color per segment)."""
    # the axis lines are one-segment black Line3DCollections; the
    # trajectory is the collection with one color per (antialiased) segment
    cols = [np.asarray(c.get_edgecolor())[:, :3] for c in fig.axes[0].collections
            if type(c).__name__.startswith('Line') and len(c.get_edgecolor()) >= 10]
    assert cols, 'a continuous hue draws multicolored line collections'
    return np.vstack(cols)


# --- sort_colors --------------------------------------------------------------

def test_sort_colors_value_is_dark_to_bright_with_hue_as_the_tie_break():
    cols = np.array([[1, 0, 0], [0, 0, .2], [0, .5, 0], [.2, .2, .2], [0, 0, 1]],
                    dtype=float)
    out = sort_colors(cols, 'value')
    v = rgb_to_hsv(out)[:, 2]
    assert np.all(np.diff(v) >= 0)
    # equal value (1.0): red (hue 0) before blue (hue 2/3)
    top = out[v == 1.0]
    assert to_hex(top[0]) == '#ff0000' and to_hex(top[-1]) == '#0000ff'


def test_sort_colors_keys_are_deterministic_and_are_permutations():
    cols = np.random.default_rng(3).random((30, 3))
    for key in PALETTE_SORT_KEYS:
        a, b = sort_colors(cols, key), sort_colors(cols, key)
        assert np.array_equal(a, b)
        assert sorted(map(tuple, a)) == sorted(map(tuple, cols))
    assert np.array_equal(sort_colors(cols, None), cols)
    assert np.array_equal(sort_colors(cols, 'original'), cols)
    h = rgb_to_hsv(sort_colors(cols, 'hue'))[:, 0]
    assert np.all(np.diff(np.round(h, 6)) >= 0)
    L = luminance(sort_colors(cols, 'lightness'))
    assert np.all(np.diff(np.round(L, 6)) >= 0)
    c = sort_colors(cols, 'columns')
    assert np.all(np.diff(np.round(c[:, 0], 6)) >= 0)
    with pytest.raises(ValueError, match='key must be one of'):
        sort_colors(cols, 'brightness')


# --- image palettes: extraction order vs palette order -------------------------

def test_image_palette_keeps_salience_order_unless_asked_to_sort(tmp_path):
    path = painting_png(tmp_path)
    salient = image_palette(path)
    assert np.allclose(salient[0], np.array(COLORS['teal']) / 255, atol=0.03)
    by_value = image_palette(path, sort='value')
    assert sorted(map(tuple, np.round(by_value, 3))) == \
        sorted(map(tuple, np.round(salient, 3)))
    assert np.all(np.diff(rgb_to_hsv(by_value)[:, 2]) >= 0)


def test_an_image_used_as_a_plot_palette_is_sorted_by_value(tmp_path):
    path = painting_png(tmp_path)
    cols = get_palette_colors(f'image:{path}', 6)
    assert np.all(np.diff(rgb_to_hsv(cols)[:, 2]) >= 0)
    assert np.allclose(cols[0], np.array(COLORS['coal']) / 255, atol=0.03)
    # the spec's own sort wins, and 'original' keeps the salience order
    original = get_palette_colors(f'image:{path}?sort=original', 6)
    assert np.allclose(original, image_palette(path), atol=1e-9)
    by_hue = get_palette_colors(f'image:{path}?sort=hue', 6)
    assert np.all(np.diff(np.round(rgb_to_hsv(by_hue)[:, 0], 6)) >= 0)
    with pytest.raises(ValueError, match='sort'):
        get_palette_colors(f'image:{path}?sort=brightness', 6)


def test_the_lead_color_of_an_image_is_still_its_most_salient(tmp_path):
    path = painting_png(tmp_path)
    assert np.allclose(palette_lead_color(f'image:{path}'), image_palette(path)[0])
    # a continuous gradient from the image runs dark to bright
    grad = get_palette_colors(f'image:{path}', 50)
    assert np.all(np.diff(rgb_to_hsv(grad)[:, 2]) >= -1e-9)


# --- matrix palettes -----------------------------------------------------------

def test_is_palette_matrix_distinguishes_data_from_color_lists():
    assert not is_palette_matrix([(1, 0, 0), (0, 1, 0)])             # colors
    assert not is_palette_matrix(np.array([[.1, .2, .3], [.4, .5, .6]]))
    assert not is_palette_matrix('viridis')
    assert not is_palette_matrix({'a': 'red'})
    assert is_palette_matrix(np.array([[1.5, 0, 0], [0, 1, 0]]))       # > 1
    assert is_palette_matrix(matrix())                                  # k=12
    assert is_palette_matrix(pd.DataFrame(np.array([[.1, .2, .3], [.4, .5, .6]])))
    assert is_palette_matrix([[1.0, 2.0], [3.0, 4.0]])                  # k=2
    assert not is_palette_matrix(pd.DataFrame({'a': ['x', 'y']}))


def test_matrix_palette_reduces_scales_sorts_and_resamples():
    data = matrix()
    cmap = matrix_palette(data)
    assert isinstance(cmap, Colormap)
    rows = get_palette_colors(cmap, 40)
    assert rows.shape == (40, 3) and rows.min() >= 0 and rows.max() <= 1
    # the SAME rows as reducing by hand: PCA to 3, min-max per column, sorted
    # along the first component
    reduced = hyp.reduce(data, reduce='PCA', ndims=3, random_state=0)
    lo, hi = reduced.min(axis=0), reduced.max(axis=0)
    expected = sort_colors((reduced - lo) / (hi - lo), 'columns')
    assert np.allclose(get_palette_colors(cmap, 40), expected, atol=1e-6)
    assert np.all(np.diff(expected[:, 0]) >= 0)
    # resampled to any count by interpolation, endpoints kept
    five = get_palette_colors(cmap, 5)
    assert np.allclose(five[0], expected[0], atol=1e-6)
    assert np.allclose(five[-1], expected[-1], atol=1e-6)
    # deterministic
    assert np.allclose(get_palette_colors(matrix_palette(data), 40), expected,
                       atol=1e-6)


def test_matrix_palette_honours_the_reduce_spec_and_sort_key():
    data = matrix()
    by_hue = get_palette_colors(matrix_palette(data, sort='hue'), 40)
    assert np.all(np.diff(np.round(rgb_to_hsv(by_hue)[:, 0], 6)) >= 0)
    ipca = get_palette_colors(matrix_palette(data, reduce='IncrementalPCA'), 40)
    pca = get_palette_colors(matrix_palette(data), 40)
    assert ipca.shape == pca.shape
    # a different reducer is genuinely applied (its columns are not PCA's)
    ica = get_palette_colors(matrix_palette(data, reduce='FastICA'), 40)
    assert not np.allclose(ica, pca, atol=1e-3)
    with pytest.raises(ValueError, match='at least two rows'):
        matrix_palette(np.ones((1, 5)))
    with pytest.raises(ValueError, match='finite'):
        bad = data.copy()
        bad[3, 4] = np.nan
        matrix_palette(bad)


def test_narrow_matrices_are_not_reduced_and_missing_channels_are_neutral():
    two = np.array([[0, 10], [5, 0], [10, 5]], dtype=float)
    rows = get_palette_colors(matrix_palette(two, sort='original'), 3)
    assert np.allclose(rows[:, 2], 0.5)
    assert np.allclose(rows[:, 0], [0, .5, 1]) and np.allclose(rows[:, 1], [1, 0, .5])
    one = np.array([[3.0], [1.0], [2.0]])
    rows = get_palette_colors(matrix_palette(one, sort='original'), 3)
    assert np.allclose(rows[:, 1:], 0.5) and np.allclose(rows[:, 0], [1, 0, .5])
    # a constant column is neutral too
    const = np.column_stack([np.arange(4.0), np.ones(4), np.arange(4.0) * 2])
    rows = get_palette_colors(matrix_palette(const * 10, sort='original'), 4)
    assert np.allclose(rows[:, 1], 0.5)


def test_normalize_and_manip_reach_the_reducer():
    data = matrix()
    plain = get_palette_colors(matrix_palette(data), 40)
    smoothed = get_palette_colors(matrix_palette(data, manip='Smooth'), 40)
    assert plain.shape == smoothed.shape and not np.allclose(plain, smoothed)
    normalized = get_palette_colors(matrix_palette(data, normalize='across'), 40)
    assert normalized.shape == plain.shape
    # z-scoring the columns before PCA re-weights them, so the palette
    # differs from the raw-matrix one and equals a by-hand normalize+PCA
    assert not np.allclose(normalized, plain, atol=1e-3)
    by_hand = hyp.reduce(hyp.normalize(data), reduce='PCA', ndims=3, random_state=0)
    lo, hi = by_hand.min(axis=0), by_hand.max(axis=0)
    assert np.allclose(normalized, sort_colors((by_hand - lo) / (hi - lo), 'columns'), atol=1e-6)


# --- through hyp.plot ----------------------------------------------------------

@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_a_matrix_palette_colors_a_continuous_hue_along_the_reduced_axis(backend):
    x = _walk()
    weights = matrix(t=len(x))
    hue = np.arange(len(x), dtype=float)
    fig = hyp.plot(x, hue=hue, palette=weights, show=False, backend=backend)
    expected = get_palette_colors(matrix_palette(weights), 100)   # n_bins
    if backend == 'matplotlib':
        got = _mpl_segment_colors(fig)
    else:
        tr = [t for t in fig.data if (t.meta or {}).get('hyp_trace_index') is not None]
        got = np.array([_rgb_triplet(c) for c in tr[0].line.color]) / 255.0
    # the trajectory starts and ends on the palette's ends, and its red
    # channel (the first reduced component, the sort key) rises along the
    # way -- antialiased segments blend neighbouring palette colors, so the
    # check is on the ends and the monotone channel, not on exact membership
    assert np.linalg.norm(got[0] - expected[0]) < 0.05
    assert np.linalg.norm(got[-1] - expected[-1]) < 0.05
    assert np.all(np.diff(got[:, 0]) >= -1e-6)


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_per_dataset_matrix_palettes_give_each_dataset_its_most_saturated_color(backend):
    x = _walk()
    mats = [matrix(seed=1), matrix(seed=2)]
    fig = hyp.plot([x, x + 3], palette=mats, show=False, backend=backend)
    def most_saturated(m):
        anchors = matrix_palette(m).anchors
        return anchors[np.argmax(anchors.max(axis=1) - anchors.min(axis=1))]
    leads = [most_saturated(m) for m in mats]
    assert all(np.allclose(palette_lead_color(m), lead) for m, lead in zip(mats, leads))
    if backend == 'matplotlib':
        got = [np.asarray(matplotlib.colors.to_rgb(ln.get_color()))
               for ln in fig.axes[0].lines[:2]]
    else:
        tr = [t for t in fig.data if (t.meta or {}).get('hyp_trace_index') is not None]
        got = [np.array(_rgb_triplet(t.line.color)) / 255.0 for t in tr[:2]]
    for g, lead in zip(got, leads):
        assert np.linalg.norm(g - lead) < 0.02


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_palette_sort_orders_an_image_palette_on_both_backends(backend, tmp_path):
    x = _walk()
    path = painting_png(tmp_path)
    hue = ['a'] * 10 + ['b'] * 10 + ['c'] * 10 + ['d'] * 10
    def drawn(**kw):
        fig = hyp.plot(x, hue=hue, palette=f'image:{path}', show=False,
                       backend=backend, **kw)
        if backend == 'matplotlib':
            return np.array([matplotlib.colors.to_rgb(ln.get_color())
                             for ln in fig.axes[0].lines[:4]])
        tr = [t for t in fig.data if (t.meta or {}).get('hyp_trace_index') is not None]
        return np.array([_rgb_triplet(t.line.color) for t in tr[:4]]) / 255.0
    by_value, by_hue, original = drawn(), drawn(palette_sort='hue'), \
        drawn(palette_sort='original')
    assert np.all(np.diff(rgb_to_hsv(by_value)[:, 2]) >= -1e-6)
    assert np.all(np.diff(np.round(rgb_to_hsv(by_hue)[:, 0], 6)) >= 0)
    # plotly stores 8-bit channels
    assert np.allclose(original, image_palette(path, n_colors=4),
                       atol=1e-6 if backend == 'matplotlib' else 3e-3)
    with pytest.raises(ValueError, match='palette_sort= must be one of'):
        hyp.plot(x, palette=f'image:{path}', palette_sort='brightness',
                 show=False, backend=backend)


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_palette_reduce_and_stage_kwargs_reach_the_matrix(backend):
    x = _walk()
    weights = matrix(t=len(x))
    hue = np.arange(len(x), dtype=float)
    a = hyp.plot(x, hue=hue, palette=weights, show=False, backend=backend)
    b = hyp.plot(x, hue=hue, palette=weights, palette_reduce='FastICA',
                 palette_sort='hue', show=False, backend=backend)
    def pick(f):
        if backend == 'matplotlib':
            return _mpl_segment_colors(f)
        tr = [t for t in f.data if (t.meta or {}).get('hyp_trace_index') is not None][0]
        return np.array([_rgb_triplet(c) for c in tr.line.color]) / 255.0
    assert not np.allclose(pick(a), pick(b), atol=1e-3)
    # each stage kwarg reaches hyp.reduce: the drawn colors equal those of a
    # palette built BY HAND from the same staged reduction, and differ from
    # the unstaged palette (Codex round 12: the test's name promised the
    # stage kwargs but only exercised palette_reduce/palette_sort)
    tol = 1e-6 if backend == 'matplotlib' else 3e-3
    for stage, value in (('manip', 'Smooth'), ('normalize', 'across'),
                         ('align', 'hyper')):
        staged = hyp.plot(x, hue=hue, palette=weights, show=False,
                          backend=backend, **{f'palette_{stage}': value})
        reduced = hyp.reduce([weights, weights] if stage == 'align' else weights,
                             reduce='PCA', ndims=3, random_state=0, **{stage: value})
        reduced = np.asarray(reduced[0] if stage == 'align' else reduced)
        lo, hi = reduced.min(axis=0), reduced.max(axis=0)
        anchors = sort_colors((reduced - lo) / np.where(hi > lo, hi - lo, 1.0), 'columns')
        by_hand = hyp.plot(x, hue=hue, palette=MatrixColormap('by-hand', anchors),
                           show=False, backend=backend)
        assert np.allclose(pick(staged), pick(by_hand), atol=tol), stage
        if stage != 'align':
            # (aligning ONE matrix -- against itself -- changes nothing; the
            # by-hand equality above is the check that the kwarg arrived)
            assert not np.allclose(pick(staged), pick(a), atol=1e-3), stage


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_forecast_palette_matrix_colors_match_between_single_and_panel_calls(backend):
    """A matrix `forecast_palette=` colours the forecasts the same way in a
    single-axes call and inside `panels=`, on both backends (Codex round
    11 asked for colour assertions, not an existence check)."""
    x, y = _walk(seed=4), _walk(seed=5) + 2
    fp = matrix(t=8)
    kw = dict(predict='Kalman', t=3, forecast_palette=fp, show=False, backend=backend)

    def forecast_colors(fig):
        if backend == 'matplotlib':
            return [to_hex(ln.get_color()) for ax in fig.axes for ln in ax.lines
                    if getattr(ln, '_hyp_forecast_role', None) == 'static']
        return [_rgb_triplet(tr.line.color) for tr in fig.data
                if (tr.meta or {}).get('hyp_forecast_role') == 'static']
    single = forecast_colors(hyp.plot([x, y], **kw))
    panels = forecast_colors(hyp.plot([x, y], panels=True, **kw))
    assert len(single) == 2 and panels == single
    expected = get_palette_colors(matrix_palette(fp), 2)
    if backend == 'matplotlib':
        assert [to_hex(c) for c in expected] == single
    else:
        assert [tuple(int(round(v * 255)) for v in c) for c in expected] == single


# --- the matrix colormap honours matplotlib's Colormap contract (Codex round 10)

def test_matrix_colormap_supports_the_inherited_colormap_operations():
    """`MatrixColormap` built its parent from a bare anchor list, so integer
    sampling, `resampled()`, `reversed()`, an alpha array and masked input
    all failed while float sampling (the only path the tests used) worked."""
    cmap = matrix_palette(np.random.default_rng(0).normal(size=(8, 5)))
    anchors = cmap.anchors
    # exact float sampling at the anchors, and the parent's LUT agrees
    assert np.allclose(cmap(np.linspace(0, 1, 8))[:, :3], anchors)
    lut = cmap(np.arange(cmap.N))                       # integer indices
    assert lut.shape == (cmap.N, 4) and np.allclose(lut[0, :3], anchors[0], atol=1e-6)
    assert np.allclose(cmap(0)[:3], anchors[0]) and np.allclose(cmap(cmap.N - 1)[:3], anchors[-1])
    # resampled() and reversed() are the inherited colormaps, consistent with the sampler
    small = cmap.resampled(8)
    assert np.allclose(small(np.linspace(0, 1, 8))[:, :3], anchors, atol=2e-3)
    rev = cmap.reversed()
    # the reversed map samples through the parent's 256-entry table, so
    # intermediate points carry its interpolation error (~ slope / 255)
    assert np.allclose(rev(np.linspace(0, 1, 8))[:, :3], anchors[::-1], atol=0.03)
    assert np.allclose(rev(0.0)[:3], anchors[-1]) and np.allclose(rev(1.0)[:3], anchors[0])
    # an alpha array, bytes, masked and NaN input follow matplotlib's rules
    out = cmap(np.array([0.0, 0.5, 1.0]), alpha=np.array([0.2, 0.5, 1.0]))
    assert np.allclose(out[:, 3], [0.2, 0.5, 1.0])
    assert cmap(np.array([0.0, 1.0]), bytes=True).dtype == np.uint8
    masked = cmap(np.ma.masked_array([0.0, 0.5], mask=[False, True]))
    assert np.allclose(masked[1], cmap.get_bad())
    assert np.allclose(cmap(np.array([np.nan]))[0], cmap.get_bad())
    from matplotlib.colors import Colormap
    assert isinstance(cmap, Colormap)
    with pytest.raises(ValueError, match='at least two'):
        from hypertools.plot.colors import MatrixColormap
        MatrixColormap('x', np.ones((1, 3)))


# --- Codex round 11 ------------------------------------------------------------

def test_matrix_colormap_applies_under_over_and_bad_per_element():
    """The exact float sampler clipped out-of-range values to the ends
    (ignoring set_under/set_over) and one NaN sent the whole array through
    the quantized table, changing the other entries' colors."""
    cmap = matrix_palette(np.random.default_rng(0).normal(size=(8, 5)))
    cmap.set_under('red')
    cmap.set_over('blue')
    out = cmap(np.array([-0.1, 0.25, 1.1]))
    assert to_hex(out[0][:3]) == '#ff0000' and to_hex(out[2][:3]) == '#0000ff'
    clean = cmap(np.array([0.25, 0.75]))
    with_nan = cmap(np.array([0.25, np.nan, 0.75]))
    assert np.allclose(with_nan[[0, 2]], clean)              # unchanged neighbours
    assert np.allclose(with_nan[1], cmap.get_bad())
    assert np.allclose(cmap(0.25), clean[0])                 # scalar == vector entry


def test_interpolated_image_palettes_stay_distinct_above_256_categories(tmp_path):
    """`sns.blend_palette` sampled a 256-entry table, so 257 categories got
    256 colors (two categories shared one). Exact interpolation now."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :5] = (200, 30, 30)
    img[:, 5:] = (30, 30, 200)
    path = tmp_path / 'two.png'
    Image.fromarray(img).save(path)
    from hypertools.plot.colors import interpolate_colors
    for n in (2, 17, 255, 257, 400):
        cols = get_palette_colors(f'image:{path}', n)
        assert len(cols) == n
        assert len({tuple(np.round(c, 9)) for c in cols}) == n
    # the continuous short-list path uses the same exact interpolation
    assert len({tuple(np.round(c, 9)) for c in interpolate_colors([(1, 0, 0), (0, 0, 1)], 300)}) == 300
    x = _walk(n=300, seed=7)
    fig = hyp.plot(x, '.', hue=[str(i) for i in range(300)], palette=f'image:{path}',
                   show=False)
    # the artists carry the float colours the library assigned (a hex
    # rendering would quantize a two-anchor gradient to ~170 values)
    drawn = {tuple(np.round(matplotlib.colors.to_rgb(ln.get_color()), 9))
             for ln in fig.axes[0].lines}
    assert len(drawn) == 300


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_polars_forecast_hue_is_partitioned_under_panels(backend):
    """A polars `forecast_hue` Series failed under panels= on both backends
    while the equivalent pandas Series and the ordinary call succeeded."""
    import polars as pl
    x, y = _walk(seed=4), _walk(seed=5) + 2
    hue = ['a', 'b']
    kw = dict(predict='Kalman', t=3, forecast_palette=['red', 'blue'], show=False, backend=backend)
    fig_pl = hyp.plot([x, y], panels=True, forecast_hue=pl.Series('h', hue), **kw)
    fig_pd = hyp.plot([x, y], panels=True, forecast_hue=pd.Series(hue), **kw)

    def forecast_colors(fig):
        if backend == 'matplotlib':
            return [to_hex(ln.get_color()) for ax in fig.axes for ln in ax.lines
                    if getattr(ln, '_hyp_forecast_role', None) == 'static']
        return [_rgb_triplet(tr.line.color) for tr in fig.data
                if (tr.meta or {}).get('hyp_forecast_role') == 'static']
    got = forecast_colors(fig_pl)
    assert got == forecast_colors(fig_pd) and len(got) == 2
    assert got[0] != got[1]
