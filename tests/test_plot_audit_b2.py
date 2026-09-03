"""Regression tests for release-1.0 audit batch B2 (hue/colors/fonts).

Covers CONFIRMED findings from units F02-plot-hue and
F24-colors-fonts-interactive (plus F01-plot-static-core-013), all
reproduced against the real plotting pipeline (MPLBACKEND=Agg, real data,
fixed seeds, no mocks). Each test names the finding(s) it locks in.
"""

import warnings

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.plot.colors import (colors2groups, get_palette_colors,
                                    mat2colors)

GRAY = (0.75, 0.75, 0.75)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


# ---------------------------------------------------------------------------
# F02-plot-hue-001 / F24-colors-fonts-interactive-004: NaN in hue
# ---------------------------------------------------------------------------

def test_nan_in_continuous_hue_keeps_gradient_unit():
    h = np.linspace(0, 1, 20)
    h[5] = np.nan
    with pytest.warns(UserWarning, match='non-finite'):
        cols = np.asarray(mat2colors(h, palette='hls'))
    # the NaN point is neutral gray...
    assert np.allclose(cols[5], GRAY)
    # ...and the 19 finite points keep their full gradient (19 distinct
    # colors, exactly matching the same values passed without the NaN)
    finite = np.delete(np.arange(20), 5)
    uniq = np.unique(np.round(cols[finite], 4), axis=0)
    assert len(uniq) == 19
    control = np.asarray(mat2colors(np.delete(h, 5), palette='hls'))
    assert np.allclose(cols[finite], control, atol=1e-12)


def test_nan_in_continuous_hue_keeps_gradient_end_to_end():
    np.random.seed(3)
    pts = np.random.randn(20, 4)
    h = np.linspace(0, 1, 20)
    h[5] = np.nan
    with pytest.warns(UserWarning, match='non-finite'):
        fig = hyp.plot(pts, fmt='.', hue=h, ndims=2, show=False)
    ax = fig.axes[0]
    colls = [c for c in ax.collections if hasattr(c, 'get_facecolor')]
    assert colls, 'expected a per-point-colored scatter'
    face = np.vstack([c.get_facecolor()[:, :3] for c in colls])
    # gradient survives: many distinct rendered colors, not 2
    assert len(np.unique(np.round(face, 4), axis=0)) >= 15
    # the NaN observation renders neutral gray
    assert np.any(np.all(np.isclose(face, GRAY, atol=1e-6), axis=1))


def test_nan_matrix_hue_row_neutral_not_uniform_blend():
    with pytest.warns(UserWarning, match='non-finite'):
        cols = np.asarray(mat2colors(np.array([[0.5, 0.5], [np.nan, 0.5]])))
    # the finite row keeps the true 50/50 blend of the 2 hls components
    expected_mid = get_palette_colors('hls', 2).mean(axis=0)
    assert np.allclose(cols[0], expected_mid, atol=1e-12)
    # the NaN row is neutral gray, NOT silently identical to a real
    # [0.5, 0.5] mixture row
    assert np.allclose(cols[1], GRAY)
    assert not np.allclose(cols[1], cols[0])


# ---------------------------------------------------------------------------
# F02-plot-hue-002: singleton hue category + line fmt cannot crash
# ---------------------------------------------------------------------------

def test_singleton_last_category_static_line_fmt():
    np.random.seed(3)
    with pytest.warns(UserWarning, match='only one observation'):
        fig = hyp.plot(np.random.randn(20, 4), hue=['a'] * 19 + ['b'],
                       show=False)
    assert fig is not None


def test_singleton_last_category_animated_line_fmt():
    np.random.seed(3)
    # animated path went through scipy pchip with a 1-point group and
    # crashed with "ValueError: `x` must contain at least 2 elements."
    with pytest.warns(UserWarning, match='only one observation'):
        result = hyp.plot(np.random.randn(20, 3), hue=['a'] * 19 + ['b'],
                          animate=True, show=False, duration=1, frame_rate=5)
    assert result is not None


def test_singleton_first_category_still_works():
    np.random.seed(3)
    fig = hyp.plot(np.random.randn(20, 4), hue=['b'] + ['a'] * 19,
                   show=False)
    assert fig is not None


def test_singleton_warning_names_the_category_not_the_legend_sentinel():
    """The warning must name the user's own category.

    It used to read `hue category '_nolegend_' has only one observation`.
    `'_nolegend_'` is matplotlib's sentinel for "keep this artist out of the
    legend", which `_regroup_categorical_lines` assigns to every REPEAT run
    of a category so each category gets exactly one legend entry. The
    warning was reading that legend-label list, so any singleton after the
    first run of its category was reported under a name no user could ever
    have supplied -- and there is no category called `_nolegend_` to go
    looking for.

    Alternating labels under a pure line format make every run a singleton,
    so the runs that get the sentinel are exactly the ones warned about.
    """
    np.random.seed(3)
    with pytest.warns(UserWarning, match='only one observation') as record:
        hyp.plot(np.random.randn(60, 3), '-', hue=['a', 'b'] * 30,
                 show=False)
    messages = [str(w.message) for w in record
                if 'only one observation' in str(w.message)]
    assert messages, 'expected the singleton-category warning'
    joined = ' '.join(messages)
    assert '_nolegend_' not in joined, (
        f"the warning leaked matplotlib's legend sentinel to the user: "
        f'{joined!r}')
    # it names a category the caller actually passed
    assert "'a'" in joined or "'b'" in joined, (
        f'the warning names no real category: {joined!r}')
    # ...and not a numpy scalar repr like np.str_('b')
    assert 'np.str_' not in joined, (
        f'numpy scalar repr leaked into a user-facing message: {joined!r}')


# ---------------------------------------------------------------------------
# F02-plot-hue-003: pandas Series hue with ANY index (positional semantics)
# ---------------------------------------------------------------------------

def test_series_hue_nondefault_index_matches_list_hue():
    np.random.seed(3)
    pts = np.random.randn(20, 4)
    labels = ['a', 'b'] * 10
    fig_series = hyp.plot(pts, fmt='.', ndims=2, show=False,
                          hue=pd.Series(labels, index=range(100, 120)))
    fig_list = hyp.plot(pts, fmt='.', hue=labels, ndims=2, show=False)
    colors_series = sorted(matplotlib.colors.to_hex(ln.get_color())
                           for ln in fig_series.axes[0].lines)
    colors_list = sorted(matplotlib.colors.to_hex(ln.get_color())
                         for ln in fig_list.axes[0].lines)
    assert colors_series == colors_list
    assert len(fig_series.axes[0].lines) == 2


def test_categorical_index_and_categorical_hue_work():
    np.random.seed(3)
    pts = np.random.randn(12, 4)
    fig = hyp.plot(pts, fmt='.', ndims=2, show=False,
                   hue=pd.Categorical(['x', 'y', 'z'] * 4))
    assert len(fig.axes[0].lines) == 3
    fig2 = hyp.plot(pts, fmt='.', ndims=2, show=False,
                    hue=pd.Index(['x', 'y', 'z'] * 4))
    assert len(fig2.axes[0].lines) == 3


# ---------------------------------------------------------------------------
# F02-plot-hue-004: combo fmt ('o-') + continuous hue keeps the line and
# marks only the REAL sample points
# ---------------------------------------------------------------------------

def test_combo_fmt_continuous_hue_line_plus_raw_markers():
    from matplotlib.collections import LineCollection, PathCollection
    np.random.seed(3)
    pts = np.random.randn(20, 4)
    fig = hyp.plot(pts, fmt='o-', hue=np.linspace(0, 1, 20), ndims=2,
                   show=False)
    ax = fig.axes[0]
    line_colls = [c for c in ax.collections if isinstance(c, LineCollection)]
    marker_colls = [c for c in ax.collections
                    if isinstance(c, PathCollection)]
    assert line_colls, 'combo fmt must keep a multicolored connecting line'
    assert marker_colls, 'combo fmt must keep its markers'
    n_marked = sum(c.get_offsets().shape[0] for c in marker_colls)
    assert n_marked == 20, (
        f'markers must sit at the 20 TRUE sample points, not every '
        f'interpolated point; got {n_marked}')


def test_pure_marker_fmt_continuous_hue_unchanged():
    np.random.seed(3)
    pts = np.random.randn(20, 4)
    fig = hyp.plot(pts, fmt='.', hue=np.linspace(0, 1, 20), ndims=2,
                   show=False)
    ax = fig.axes[0]
    n_pts = sum(c.get_offsets().shape[0] for c in ax.collections
                if hasattr(c, 'get_offsets'))
    assert n_pts == 20


# ---------------------------------------------------------------------------
# F02-plot-hue-005 / F24-006: list palettes (hex/named/RGBA) work uniformly
# ---------------------------------------------------------------------------

def test_hex_list_palette_categorical_returns_rgb_floats():
    cols = np.asarray(mat2colors(['a', 'b'], palette=['#ff0000', '#0000ff']))
    assert cols.shape == (2, 3)
    assert cols.dtype.kind == 'f'
    assert np.allclose(cols, [[1, 0, 0], [0, 0, 1]])


def test_rgba_list_palette_consistent_rgb_shape():
    cats = np.asarray(mat2colors(['a', 'b'],
                                 palette=[(1, 0, 0, 1), (0, 0, 1, 1)]))
    blend = np.asarray(mat2colors(np.array([[0.5, 0.5]]),
                                  palette=[(1, 0, 0, 1), (0, 0, 1, 1)]))
    assert cats.shape == (2, 3)
    assert blend.shape == (1, 3)
    assert np.allclose(blend[0], [0.5, 0, 0.5])


def test_hex_list_palette_matrix_hue_end_to_end():
    np.random.seed(7)
    pts = np.random.randn(20, 4)
    mix = np.column_stack([np.linspace(0, 1, 20), 1 - np.linspace(0, 1, 20)])
    fig = hyp.plot(pts, fmt='.', hue=mix, palette=['#ff0000', '#0000ff'],
                   ndims=2, show=False)
    assert fig is not None


def test_hex_list_palette_colorbar_end_to_end():
    np.random.seed(7)
    pts = np.random.randn(20, 4)
    fig = hyp.plot(pts, fmt='.', hue=['a'] * 10 + ['b'] * 10,
                   palette=['#ff0000', '#0000ff'], colorbar=True, ndims=2,
                   show=False)
    assert len(fig.axes) == 2


def test_named_color_list_palette():
    cols = np.asarray(mat2colors(['a', 'b'], palette=['red', 'blue']))
    assert np.allclose(cols, [[1, 0, 0], [0, 0, 1]])


# ---------------------------------------------------------------------------
# F02-plot-hue-006 / F24-017: short list palettes blend for continuous hue
# ---------------------------------------------------------------------------

def test_short_list_palette_blends_for_continuous_hue():
    vals = np.linspace(0, 1, 30)
    cols = np.asarray(mat2colors(vals, palette=[(1, 0, 0), (0, 0, 1)]))
    assert cols.shape == (30, 3)
    # endpoints match the supplied anchor colors
    assert np.allclose(cols[0], [1, 0, 0], atol=0.05)
    assert np.allclose(cols[-1], [0, 0, 1], atol=0.05)
    # a real gradient in between
    assert len(np.unique(np.round(cols, 4), axis=0)) >= 25


def test_short_list_palette_continuous_hue_end_to_end():
    np.random.seed(2)
    fig = hyp.plot(np.random.randn(30, 4), fmt='.', hue=np.linspace(0, 1, 30),
                   palette=[(1, 0, 0), (0, 0, 1)], ndims=2, show=False)
    assert fig is not None


def test_short_list_palette_categorical_error_names_palette():
    with pytest.raises(ValueError, match='palette'):
        mat2colors(['a', 'b', 'c'], palette=['#ff0000', '#0000ff'])


def test_colormap_palette_supported():
    cmap = plt.get_cmap('viridis')
    cats = np.asarray(mat2colors(['a', 'b', 'c'], palette=cmap))
    assert cats.shape == (3, 3)
    cont = np.asarray(mat2colors(np.linspace(0, 1, 10), palette=cmap))
    assert len(np.unique(np.round(cont, 4), axis=0)) == 10
    np.random.seed(2)
    fig = hyp.plot(np.random.randn(12, 4), fmt='.',
                   hue=['a', 'b', 'c'] * 4, palette=cmap, ndims=2,
                   show=False)
    assert len(fig.axes[0].lines) == 3


# ---------------------------------------------------------------------------
# F02-plot-hue-007: colorbar + categorical hue shows category names without
# requiring legend=True
# ---------------------------------------------------------------------------

def test_colorbar_categorical_hue_names_without_legend():
    np.random.seed(2)
    pts = np.random.randn(30, 4)
    cats = ['a'] * 10 + ['b'] * 10 + ['c'] * 10
    fig = hyp.plot(pts, fmt='.', hue=cats, colorbar=True, ndims=2,
                   show=False)
    ticklabels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert ticklabels == ['a', 'b', 'c']


def test_colorbar_categorical_hue_names_with_legend_unchanged():
    np.random.seed(2)
    pts = np.random.randn(30, 4)
    cats = ['a'] * 10 + ['b'] * 10 + ['c'] * 10
    fig = hyp.plot(pts, fmt='.', hue=cats, colorbar=True, legend=True,
                   ndims=2, show=False)
    ticklabels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert ticklabels == ['a', 'b', 'c']


# ---------------------------------------------------------------------------
# F02-plot-hue-008: color_reduce error names color_reduce= and the value
# ---------------------------------------------------------------------------

def test_color_reduce_error_names_kwarg_and_value():
    np.random.seed(2)
    with pytest.raises(ValueError) as excinfo:
        hyp.plot(np.random.randn(30, 4), fmt='.', hue=np.random.randn(30, 6),
                 color_reduce='NotAModelXYZ', ndims=2, show=False)
    msg = str(excinfo.value)
    assert 'color_reduce' in msg
    assert 'NotAModelXYZ' in msg
    assert '  ' not in msg, 'no whitespace runs'
    assert 'http://hypertools.readthedocs.io' not in msg


# ---------------------------------------------------------------------------
# F02-plot-hue-009: names= + categorical hue -> explanatory error
# ---------------------------------------------------------------------------

def test_names_with_categorical_hue_explains_regrouping():
    np.random.seed(3)
    d = [np.random.randn(10, 3), np.random.randn(10, 3)]
    with pytest.raises(ValueError, match='regroup'):
        hyp.plot(d, fmt='.', hue=['a'] * 7 + ['b'] * 7 + ['c'] * 6,
                 names=['first', 'second'], ndims=2, show=False)


def test_names_with_coincident_hue_group_count_also_explains():
    # 2 datasets, 2 hue categories: previously names silently labeled the
    # CATEGORY groups instead of the datasets the user named
    np.random.seed(3)
    d = [np.random.randn(10, 3), np.random.randn(10, 3)]
    with pytest.raises(ValueError, match='regroup'):
        hyp.plot(d, fmt='.', hue=['a'] * 10 + ['b'] * 10,
                 names=['first', 'second'], ndims=2, show=False)


def test_names_without_hue_still_works():
    np.random.seed(3)
    d = [np.random.randn(10, 3), np.random.randn(10, 3)]
    fig = hyp.plot(d, fmt='.', names=['first', 'second'], ndims=2,
                   show=False)
    legend_texts = [t.get_text()
                    for t in fig.axes[0].get_legend().get_texts()]
    assert legend_texts == ['first', 'second']


# ---------------------------------------------------------------------------
# F02-plot-hue-010: unhashable hue -> TypeError naming hue=
# ---------------------------------------------------------------------------

def test_unhashable_hue_raises_clear_typeerror():
    np.random.seed(3)
    # the scalar (dict) hue provokes the one-group broadcast notice before
    # its unhashability raises the TypeError -- assert both
    with pytest.warns(UserWarning, match='single scalar value'), \
         pytest.raises(TypeError, match='hue'):
        hyp.plot(np.random.randn(20, 4), fmt='.', hue={'a': 1}, ndims=2,
                 show=False)


# ---------------------------------------------------------------------------
# F02-plot-hue-013: None hue entries -> de-emphasized gray, stable palette
# ---------------------------------------------------------------------------

def test_none_hue_entries_render_gray_and_keep_named_palette_order():
    np.random.seed(3)
    fig = hyp.plot(np.random.randn(20, 4), fmt='.',
                   hue=['a', None, 'b', 'a'] * 5, legend=True, ndims=2,
                   show=False)
    ax = fig.axes[0]
    assert len(ax.lines) == 3
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_texts == ['a', 'b']
    line_colors = [tuple(np.round(matplotlib.colors.to_rgb(ln.get_color()), 4))
                   for ln in ax.lines]
    # the None group (second in first-appearance order) is neutral gray
    assert line_colors[1] == GRAY
    # named categories take the first palette slots, in order, unaffected
    # by where the None entries appear
    expected = [tuple(np.round(c, 4)) for c in get_palette_colors('hls', 2)]
    assert line_colors[0] == expected[0]
    assert line_colors[2] == expected[1]


# ---------------------------------------------------------------------------
# F01-plot-static-core-013 / F24-013: cyclic default palette endpoints
# ---------------------------------------------------------------------------

def test_hls_continuous_endpoints_distinguishable():
    c = np.asarray(mat2colors(np.arange(100).astype(float), palette='hls'))
    assert np.linalg.norm(c[0] - c[-1]) > 0.1
    # husl is cyclic too
    c2 = np.asarray(mat2colors(np.arange(100).astype(float), palette='husl'))
    assert np.linalg.norm(c2[0] - c2[-1]) > 0.1


def test_categorical_hls_palette_unchanged_by_cyclic_trim():
    import seaborn as sns
    cols = np.asarray(mat2colors(['a', 'b', 'c'], palette='hls'))
    assert np.allclose(cols, np.asarray(sns.color_palette('hls', 3)))


# ---------------------------------------------------------------------------
# F24-001: font='sans-serif' (hyphenated family names)
# ---------------------------------------------------------------------------

def test_sans_serif_font_family_works():
    np.random.seed(0)
    fig = hyp.plot(np.random.randn(10, 3), font='sans-serif', reduce=None,
                   show=False)
    assert fig is not None


def test_unknown_font_family_still_raises_helpful_error():
    from hypertools.plot.fonts import resolve_font
    with pytest.raises(ValueError, match='nosuchfontxyz'):
        resolve_font('nosuchfontxyz', 'x')


# ---------------------------------------------------------------------------
# F24-003: existing non-font file -> clear error at resolve time
# ---------------------------------------------------------------------------

def test_nonfont_file_rejected_at_resolve_time(tmp_path):
    """A file that EXISTS but is not a loadable font must be rejected.

    The file has to exist for this to test anything: `resolve_font` branches
    on `os.path.exists` (fonts.py:401-406), so a non-existent path takes the
    *installed-font-lookup* branch and raises a different error. The previous
    version of this test passed a hardcoded absolute path that existed only
    on one developer's machine, and `match='font='` matched BOTH messages --
    so everywhere else it passed without exercising this branch at all.
    """
    from hypertools.plot.fonts import resolve_font
    not_a_font = tmp_path / 'README.md'
    not_a_font.write_text('# not a font file\n')
    with pytest.raises(ValueError,
                       match='exists but is not a loadable font file'):
        resolve_font(str(not_a_font), 'x')


# ---------------------------------------------------------------------------
# F24-008: CJK auto-font prefers a regular weight over ultra-light
# ---------------------------------------------------------------------------

def test_cjk_font_not_ultralight_weight():
    import matplotlib.font_manager as fm
    from hypertools.plot.fonts import (_covering_font_cache,
                                       find_covering_font)
    _covering_font_cache.clear()
    fp = find_covering_font('日本語のラベル')
    if fp is None:
        pytest.skip('no installed font covers CJK on this machine')
    fname = fp.get_file()
    # find the matching fontManager entry to read its weight
    entries = [e for e in fm.fontManager.ttflist if e.fname == fname]
    if not entries:
        pytest.skip('resolved font not in fontManager.ttflist')
    weights = []
    for e in entries:
        w = e.weight
        if isinstance(w, str):
            w = fm.weight_dict.get(w, 400)
        weights.append(int(w))
    family = {e.name for e in entries}
    same_family = [e for e in fm.fontManager.ttflist if e.name in family]
    if len(same_family) < 3:
        pytest.skip('resolved family has no weight variety to choose from')
    assert min(weights) >= 300, (
        f'auto-detected CJK font {fname} is ultra-light (weight '
        f'{min(weights)}); a regular weight from the same family should '
        f'be preferred')


# ---------------------------------------------------------------------------
# F24-014: pandas Series/Index/dict text containers reach the font scan
# ---------------------------------------------------------------------------

def test_font_scan_sees_pandas_series_and_dict():
    from hypertools.plot.fonts import _non_ascii_codepoints
    cps_list = _non_ascii_codepoints(['日本語'])
    assert _non_ascii_codepoints(pd.Series(['日本語'])) == cps_list
    assert _non_ascii_codepoints(pd.Index(['日本語'])) == cps_list
    assert _non_ascii_codepoints({'k': '日本語'}) == cps_list
    assert _non_ascii_codepoints(pd.Categorical(['日本語'])) == cps_list


def test_series_cjk_hue_end_to_end_no_tofu_warnings():
    from hypertools.plot.fonts import find_covering_font
    if find_covering_font('実験統制') is None:
        pytest.skip('no installed font covers CJK on this machine')
    np.random.seed(0)
    pts = np.random.randn(12, 3)
    labels = pd.Series(['実験', '統制'] * 6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fig = hyp.plot(pts, fmt='.', hue=labels, legend=True, reduce=None,
                       show=False)
        fig.canvas.draw()
    missing = [x for x in w if 'missing from font' in str(x.message)]
    assert not missing, f'tofu warnings: {missing[:3]}'


# ---------------------------------------------------------------------------
# F24-010 / F24-011: colors helpers misuse errors name the argument
# ---------------------------------------------------------------------------

def test_mat2colors_empty_input_error():
    with pytest.raises(ValueError, match='empty'):
        mat2colors([])


def test_mat2colors_scalar_input_error():
    with pytest.raises(ValueError, match='scalar'):
        mat2colors(5)


def test_mat2colors_generator_input_materialized():
    cols = np.asarray(mat2colors(iter(['a', 'b', 'a'])))
    assert cols.shape == (3, 3)


def test_mat2colors_palette_none_error():
    with pytest.raises(ValueError, match='palette'):
        mat2colors(['a', 'b'], palette=None)


def test_mat2colors_bad_n_bins_errors():
    with pytest.raises(ValueError, match='n_bins'):
        mat2colors([1., 2.], n_bins=0)
    with pytest.raises(ValueError, match='n_bins'):
        mat2colors([1., 2.], n_bins=-5)
    with pytest.raises(ValueError, match='n_bins'):
        mat2colors([1., 2.], n_bins=2.5)


def test_get_palette_colors_zero_is_empty():
    cols = get_palette_colors('hls', 0)
    assert cols.shape == (0, 3)


def test_colors2groups_res_lt_2_error():
    with pytest.raises(ValueError, match='res'):
        colors2groups(np.array([[0.5, 0.5, 0.5]]), res=1)


# ---------------------------------------------------------------------------
# F24-012: plotly_draw input validation
# ---------------------------------------------------------------------------

def test_plotly_draw_empty_data_error():
    from hypertools.plot.plotly_backend import plotly_draw
    with pytest.raises(ValueError, match='data'):
        plotly_draw([], fmt=[], kwargs_list=[], show=False)


def test_plotly_draw_4col_data_error():
    from hypertools.plot.plotly_backend import plotly_draw
    np.random.seed(0)
    with pytest.raises(ValueError, match='3'):
        plotly_draw([np.random.randn(5, 4)], fmt=['-'], kwargs_list=[{}],
                    show=False)


def test_plotly_draw_fmt_length_mismatch_error():
    from hypertools.plot.plotly_backend import plotly_draw
    np.random.seed(0)
    with pytest.raises(ValueError, match='fmt'):
        plotly_draw([np.random.randn(5, 3)], fmt=['-', '--'],
                    kwargs_list=[{}], show=False)


# ---------------------------------------------------------------------------
# F24-009: MultiIndex + linestyles= emits no redundant-linestyle warnings
# ---------------------------------------------------------------------------

def test_multiindex_linestyles_no_redundant_warnings():
    np.random.seed(0)
    idx = pd.MultiIndex.from_product([['A', 'B'], ['s1', 's2', 's3']],
                                     names=['cond', 'subj'])
    df = pd.DataFrame(np.random.randn(24, 3), index=idx.repeat(4))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hyp.plot(df, linestyles=['-', '--'], reduce=None, show=False)
    redundant = [x for x in w if 'redundantly' in str(x.message)]
    assert not redundant
