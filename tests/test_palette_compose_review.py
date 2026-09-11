"""1.1 release review (2026-09-11): palette / marker / legend / panel / ax=
composition findings, each pinned at the public API on both backends where
the path is shared.

1  a per-dataset list of {category: color} dicts was ignored on the
   marker-only (fmt='o') categorical path; the line path applied it.
2  composing into an ``ax=`` re-sampled an evenly-spaced palette ('hls',
   'viridis') at the new total, so a later call repeated an earlier call's
   colour (hls: 2 datasets then 2 more drew a2 == b1).
3  the return_model bundle's ``'colors'`` (and a colorbar) did not match the
   colours drawn when the call continued the palette on a composed
   axes/figure (or when a fmt= colour letter coloured a dataset).
4  panels= forwarded a plain legend_colors list whole, so every panel
   refused it.
5  two-column data into a 2-D ax= at the default ndims raised "the plot is
   3D".
6  nested per-dataset labels= given as arrays/Series was rejected on one
   axes (accepted with panels=).
A  per-dataset / nested labels= with hue= or cluster= crashed both
   backends.
7  an explicit marker= list lost to the fmt's marker on matplotlib.
8  markers= on a line fmt marked every smoothed vertex, static and
   animated, instead of the samples.

No mocks: every assertion reads the drawn artists or traces.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
from matplotlib.colors import to_rgb

import hypertools as hyp
from tests._plotly_colors import rgba as effective_rgba


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _walks(n=2, rows=40, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.standard_normal((rows, 3)), 0) for _ in range(n)]


def _mpl_data_lines(ax):
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) is None
            and len(ln.get_xdata()) > 1]


def _pl_data(fig):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_trace_index') is not None]


def _pl_rgb(trace, component='line'):
    # 8-bit, the precision a plotly colour string carries
    return tuple(round(v * 255) for v in effective_rgba(trace, component)[:3])


def _r3(c):
    return tuple(round(float(v) * 255) for v in to_rgb(c[:3]
                                                      if not isinstance(c, str)
                                                      else c))


# --- 1: per-dataset {category: color} dicts on the marker path -------------

_HUE = [np.repeat(['a', 'b'], 20), np.repeat(['b', 'c'], 20)]
_DICTS = [{'a': 'red', 'b': 'blue'}, {'b': 'blue', 'c': 'green'}]
_BY_NAME = {'a': _r3('red'), 'b': _r3('blue'), 'c': _r3('green')}


@pytest.mark.parametrize('fmt', ['o', '.', 'o-', '-'])
def test_per_dataset_dict_palettes_colour_markers_matplotlib(fmt):
    fig = hyp.plot(_walks(), hue=_HUE, palette=_DICTS, fmt=fmt,
                   legend=True, show=False)
    legend = fig.axes[0].get_legend()
    got = {t.get_text(): _r3(h.get_color())
           for t, h in zip(legend.get_texts(), legend.legend_handles)}
    assert got == _BY_NAME
    drawn = {_r3(ln.get_color()) for ln in fig.axes[0].lines}
    assert drawn == set(_BY_NAME.values())


@pytest.mark.parametrize('fmt', ['o', 'o-', '-'])
def test_per_dataset_dict_palettes_colour_markers_plotly(fmt):
    fig = hyp.plot(_walks(), hue=_HUE, palette=_DICTS, fmt=fmt,
                   legend=True, show=False, backend='plotly')
    comp = 'marker' if fmt == 'o' else 'line'
    got = {tr.name: _pl_rgb(tr, comp) for tr in _pl_data(fig)
           if tr.showlegend is not False}
    assert got == _BY_NAME


def test_per_dataset_dict_palette_bundle_matches_the_markers():
    out = hyp.plot(_walks(), hue=_HUE, palette=_DICTS, fmt='o',
                   show=False, return_model=True)
    cats = {k: _r3(c) for k, c in out['colors']['categories'].items()}
    assert cats == _BY_NAME


# --- 2: composing continues the palette without repeating a colour --------

def _mpl_compose(pal, counts):
    fig, axes = hyp.subplots(1, 1)
    for i, n in enumerate(counts):
        hyp.plot(_walks(n, seed=i), ax=axes[0], palette=pal, show=False)
    return [_r3(ln.get_color()) for ln in _mpl_data_lines(axes[0])]


def _pl_compose(pal, counts):
    fig = None
    for i, n in enumerate(counts):
        fig = hyp.plot(_walks(n, seed=i), ax=fig, palette=pal, show=False,
                       backend='plotly')
    return [_pl_rgb(tr) for tr in _pl_data(fig)]


def _pl_cell_compose(pal, counts):
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    for i, n in enumerate(counts):
        hyp.plot(_walks(n, seed=i), ax=cells[0], palette=pal, show=False,
                 backend='plotly')
    return [_pl_rgb(tr) for tr in _pl_data(fig)]


_COMPOSERS = {'matplotlib': _mpl_compose, 'plotly': _pl_compose,
              'plotly-cell': _pl_cell_compose}


def _min_pairwise(cols):
    a = np.asarray(cols, float)
    d = np.linalg.norm(a[:, None] - a[None], axis=-1)
    return d[np.triu_indices(len(a), 1)].min()


@pytest.mark.parametrize('where', sorted(_COMPOSERS))
def test_hls_two_then_two_draws_four_distinct_hls_colours(where):
    drawn = _COMPOSERS[where]('hls', [2, 2])
    assert len(drawn) == 4
    # the first call's colours are untouched...
    assert drawn[:2] == [_r3(c) for c in sns.color_palette('hls', 2)]
    # ...and the four together are the palette's four-colour sampling,
    # so the second call filled the gaps instead of repeating a colour
    assert len(set(drawn)) == 4
    assert set(drawn) == {_r3(c) for c in sns.color_palette('hls', 4)}


@pytest.mark.parametrize('where', sorted(_COMPOSERS))
@pytest.mark.parametrize('pal', ['viridis', 'hls', 'husl'])
def test_composed_colours_are_as_far_apart_as_a_fresh_call(where, pal):
    drawn = _COMPOSERS[where](pal, [2, 2])
    assert len(set(drawn)) == 4
    # no closer together than one fresh call drawing one MORE dataset
    fresh = [_r3(c) for c in sns.color_palette(pal, 5)]
    assert _min_pairwise(drawn) >= _min_pairwise(fresh) - 1e-3


@pytest.mark.parametrize('where', sorted(_COMPOSERS))
def test_one_then_one_is_still_what_one_call_draws(where):
    drawn = _COMPOSERS[where]('hls', [1, 1])
    assert drawn == [_r3(c) for c in sns.color_palette('hls', 2)]
    drawn = _COMPOSERS[where]('hls', [1, 2])
    assert drawn == [_r3(c) for c in sns.color_palette('hls', 3)]


@pytest.mark.parametrize('where', sorted(_COMPOSERS))
def test_a_fixed_sequence_palette_continues_in_order(where):
    drawn = _COMPOSERS[where]('deep', [2, 2])
    assert drawn == [_r3(c) for c in sns.color_palette('deep', 4)]
    # a short explicit list still cycles, as one call with four would
    drawn = _COMPOSERS[where](['navy', 'gold'], [2, 2])
    assert drawn == [_r3(c) for c in ['navy', 'gold', 'navy', 'gold']]


@pytest.mark.parametrize('where', sorted(_COMPOSERS))
def test_three_calls_never_repeat_a_colour(where):
    drawn = _COMPOSERS[where]('hls', [2, 2, 1])
    assert len(drawn) == 5
    assert len(set(drawn)) == 5


# --- 3: the bundle's colours are the colours drawn -------------------------

@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_bundle_colours_match_a_composed_call(backend):
    first = hyp.plot(_walks(1), show=False, backend=backend)
    target = first if backend == 'plotly' else first.axes[0]
    res = hyp.plot(_walks(2, seed=1), ax=target, show=False,
                   backend=backend, return_model=True)
    bundle = [_r3(c) for c in res['colors']['colors']]
    if backend == 'plotly':
        drawn = [_pl_rgb(tr) for tr in _pl_data(res['fig'])][1:]
    else:
        drawn = [_r3(ln.get_color())
                 for ln in _mpl_data_lines(res['fig'].axes[0])][1:]
    assert bundle == drawn
    assert {str(k): _r3(c)
            for k, c in res['colors']['categories'].items()} in (
        {}, {str(i + 1): c for i, c in enumerate(drawn)})
    # the colormap the bundle offers is the drawn colours too
    cmap_cols = [_r3(res['colors']['cmap'](i)) for i in range(len(drawn))]
    assert cmap_cols == drawn


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_bundle_colours_follow_a_fmt_colour_letter(backend):
    res = hyp.plot(_walks(2), fmt=['r-', '-'], show=False,
                   backend=backend, return_model=True)
    bundle = [_r3(c) for c in res['colors']['colors']]
    if backend == 'plotly':
        drawn = [_pl_rgb(tr) for tr in _pl_data(res['fig'])]
    else:
        drawn = [_r3(ln.get_color())
                 for ln in _mpl_data_lines(res['fig'].axes[0])]
    assert drawn[0] == _r3('r')
    assert bundle == drawn


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_a_composed_colorbar_shows_the_drawn_colours(backend):
    if backend == 'plotly':
        fig = hyp.plot(_walks(2), show=False, backend=backend)
        fig = hyp.plot(_walks(2, seed=1), ax=fig, show=False,
                       backend=backend, colorbar=True)
        drawn = [_pl_rgb(tr) for tr in _pl_data(fig)][2:]
        scale = [tr.marker.colorscale for tr in fig.data
                 if tr.marker is not None and tr.marker.colorscale]
        assert len(scale) == 1
        swatches = {tuple(int(v) for v in c[c.index('(') + 1:-1].split(','))
                    for _, c in scale[0]}
        assert swatches == set(drawn)
    else:
        fig, axes = hyp.subplots(1, 1)
        hyp.plot(_walks(2), ax=axes[0], show=False)
        hyp.plot(_walks(2, seed=1), ax=axes[0], show=False, colorbar=True)
        drawn = [_r3(ln.get_color()) for ln in _mpl_data_lines(axes[0])][2:]
        mesh = fig.axes[-1].collections[-1]
        swatches = [_r3(mesh.cmap(i)) for i in range(mesh.cmap.N)]
        assert swatches == drawn


# --- 4: panels= splits a plain legend_colors list per panel ----------------

def _legend_colours(ax):
    legend = ax.get_legend()
    return [(t.get_text(), _r3(h.get_color()))
            for t, h in zip(legend.get_texts(), legend.legend_handles)]


@pytest.mark.parametrize('panel_fit', ['shared', 'independent'])
def test_panels_split_a_plain_legend_colors_list(panel_fit):
    data = _walks(2, rows=30)
    single = hyp.plot(data, legend=['a', 'b'], legend_colors=['r', 'b'],
                      show=False)
    assert _legend_colours(single.axes[0]) == [('a', _r3('r')),
                                               ('b', _r3('b'))]
    fig = hyp.plot(data, legend=['a', 'b'], legend_colors=['r', 'b'],
                   panels=True, panel_fit=panel_fit, show=False)
    assert _legend_colours(fig.axes[0]) == [('a', _r3('r'))]
    assert _legend_colours(fig.axes[1]) == [('b', _r3('b'))]


def test_panels_keep_the_shared_legend_colours_after_the_datasets():
    data = _walks(2, rows=30)
    kw = dict(legend=['a', 'b'], predict='Kalman', t=3,
              legend_colors=['r', 'b', 'k'], show=False)
    single = hyp.plot(data, **kw)
    assert _legend_colours(single.axes[0]) == [
        ('a', _r3('r')), ('b', _r3('b')), ('Kalman', _r3('k'))]
    fig = hyp.plot(data, panels=True, **kw)
    assert _legend_colours(fig.axes[0]) == [('a', _r3('r')),
                                            ('Kalman', _r3('k'))]
    assert _legend_colours(fig.axes[1]) == [('b', _r3('b')),
                                            ('Kalman', _r3('k'))]


def test_panels_forward_legend_colour_pairs_whole():
    fig = hyp.plot(_walks(2, rows=30), legend=True,
                   legend_colors=[('Key', 'k')], panels=True, show=False)
    for ax in fig.axes[:2]:
        assert _legend_colours(ax) == [('Key', _r3('k'))]


# --- 5: two-column data into a 2-D ax= at the default ndims ----------------

@pytest.mark.parametrize('kw', [{}, {'reduce': None}])
def test_two_column_data_draws_into_a_2d_axes_at_default_ndims(kw):
    two = _walks(1, rows=20)[0][:, :2]
    fig, ax = plt.subplots()
    out = hyp.plot(two, ax=ax, show=False, **kw)
    assert out is fig
    # the same 2-D drawing a figure of its own gets
    own = hyp.plot(two, show=False, **kw).axes[0]
    assert own.name == ax.name == 'rectilinear'
    np.testing.assert_allclose(ax.lines[0].get_xydata(),
                               own.lines[0].get_xydata())


def test_three_column_data_still_refuses_a_2d_axes():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match='ax must also be 3d'):
        hyp.plot(_walks(1, rows=20)[0], ax=ax, show=False)
    # nothing was drawn into the refused axes
    assert not ax.lines


# --- 6 / A: labels= forms on one axes, with and without a regrouping ------

_LA = ['a%d' % i for i in range(10)]
_LB = ['b%d' % i for i in range(10)]


def _label_positions(fig, backend):
    if backend == 'plotly':
        anns = fig.layout.scene.annotations or fig.layout.annotations
        return {a.text: tuple(round(float(v), 6) for v in (
            (a.x, a.y, a.z) if hasattr(a, 'z') else (a.x, a.y)))
            for a in anns}
    out = {}
    for t in fig.axes[0].texts:
        pos = (t.get_position_3d() if hasattr(t, 'get_position_3d')
               else t.get_position())
        out[t.get_text()] = tuple(round(float(v), 6) for v in pos)
    return out


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
@pytest.mark.parametrize('form', ['arrays', 'series', 'tuples'])
def test_nested_label_sequences_of_any_type_on_one_axes(backend, form):
    import pandas as pd
    wrap = {'arrays': np.array, 'series': pd.Series, 'tuples': tuple}[form]
    data = _walks(2, rows=10)
    ref = _label_positions(hyp.plot(data, labels=[_LA, _LB], show=False,
                                    backend=backend), backend)
    got = _label_positions(hyp.plot(data, labels=[wrap(_LA), wrap(_LB)],
                                    show=False, backend=backend), backend)
    assert len(ref) == 20
    assert got == ref


_HUE_A = np.repeat(['x', 'y'], 5)
_HUE_B = np.repeat(['y', 'z'], 5)


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
@pytest.mark.parametrize('labels', [
    pytest.param([_LA, _LB], id='nested'),
    pytest.param(['A', 'B'], id='per-dataset'),
    pytest.param([np.array(_LA), np.array(_LB)], id='arrays')])
@pytest.mark.parametrize('group', [
    pytest.param(dict(hue=['x', 'y']), id='per-dataset-hue'),
    pytest.param(dict(hue=[_HUE_A, _HUE_B]), id='nested-hue'),
    pytest.param(dict(hue=[_HUE_A, _HUE_B], fmt='o'), id='nested-hue-o'),
    pytest.param(dict(hue=[_HUE_A, _HUE_B], antialias=False),
                 id='nested-hue-raw'),
    pytest.param(dict(hue=['x', 'y'], fmt='o'), id='per-dataset-hue-o'),
    pytest.param(dict(cluster='KMeans', n_clusters=3), id='cluster')])
def test_per_dataset_labels_survive_a_regrouping(backend, labels, group):
    data = _walks(2, rows=10)
    kw = {k: v for k, v in group.items() if k in ('fmt', 'antialias')}
    ref = _label_positions(hyp.plot(data, labels=labels, show=False,
                                    backend=backend, **kw), backend)
    got = _label_positions(hyp.plot(data, labels=labels, show=False,
                                    backend=backend, **group), backend)
    assert ref and sorted(got) == sorted(ref)
    # every label on its own observation: the same layout as the ungrouped
    # figure's. Compared about the labels' centroid, since plotly centres
    # a hue-segmented figure on its per-run smoothed geometry (a shift of
    # the WHOLE figure, traces and labels together, ~0.004 here)
    def centred(pos):
        keys = sorted(pos)
        arr = np.asarray([pos[k] for k in keys], float)
        return arr - arr.mean(axis=0)
    np.testing.assert_allclose(centred(got), centred(ref), atol=1e-5)


# --- 7 / 8: explicit marker= wins; markers only at the true samples -------

def _walk48():
    return _walks(2, rows=48)


def _marked(ax):
    """(label, marker, n marked points) for every artist that draws a
    marker, reading markevery (None = every vertex)."""
    out = []
    for ln in ax.lines:
        mk = ln.get_marker()
        if mk in (None, 'None', '', ' '):
            continue
        me = ln.get_markevery()
        n = len(ln.get_xdata()) if me is None else len(np.atleast_1d(me))
        out.append((ln.get_label(), mk, n))
    return out


def test_an_explicit_marker_list_wins_over_the_fmt_marker():
    fig = hyp.plot(_walk48(), fmt='-o', marker=['o', 's'], legend=['a', 'b'],
                   show=False)
    ax = fig.axes[0]
    drawn = [(lbl, mk) for lbl, mk, n in _marked(ax) if n]
    assert drawn == [('_nolegend_', 'o'), ('_nolegend_', 's')]
    legend = ax.get_legend()
    assert [h.get_marker() for h in legend.legend_handles] == ['o', 's']
    # plotly draws the same symbols
    pfig = hyp.plot(_walk48(), fmt='-o', marker=['o', 's'], show=False,
                    backend='plotly')
    assert [tr.marker.symbol for tr in _pl_data(pfig)] == ['circle',
                                                            'square']


@pytest.mark.parametrize('kw', [dict(markers='o'), dict(marker='o'),
                                dict(fmt='--', marker='s'),
                                dict(fmt='-o')])
def test_markers_on_a_smoothed_line_sit_only_at_the_samples(kw):
    data = _walk48()
    fig = hyp.plot(data, legend=['a', 'b'], show=False, **kw)
    ax = fig.axes[0]
    marked = [(lbl, n) for lbl, _, n in _marked(ax) if n]
    # one markers-only artist per dataset, one marker per SAMPLE
    assert marked == [('_nolegend_', 48), ('_nolegend_', 48)]
    # ...drawn at the samples themselves: every one is a vertex of the
    # smoothed line of the same dataset (antialias keeps each sample)
    lines = [ln for ln in ax.lines if ln.get_label() in ('a', 'b')]
    dots = [ln for ln in ax.lines if ln.get_label() == '_nolegend_']
    for line, dot in zip(lines, dots):
        curve = np.column_stack(line.get_data_3d())
        pts = np.column_stack(dot.get_data_3d())
        assert len(curve) > 5 * len(pts)          # the line IS smoothed
        gap = np.min(np.linalg.norm(curve[:, None] - pts[None], axis=-1),
                     axis=0)
        assert gap.max() < 1e-9
    # the legend glyph still shows the marker with the line
    assert [h.get_marker() for h in ax.get_legend().legend_handles] == \
        [kw.get('marker', 'o')] * 2


@pytest.mark.parametrize('kw', [dict(fmt='o-'), dict(markers='o'),
                                dict(fmt='--', marker='s')])
def test_animated_markers_sit_at_the_samples_not_every_vertex(kw):
    data = _walk48()
    anim = hyp.plot(data, animate='spin', duration=3, frame_rate=20,
                    legend=['a', 'b'], show=False, **kw)
    anim.draw_frame(anim.n_frames - 1)
    ax = anim.figure.axes[0]
    lines = [ln for ln in ax.lines if ln.get_label() in ('a', 'b')]
    assert len(lines) == 2
    # the observations, in the figure's coordinates: a static plot of the
    # same data marks them exactly (see the static test above)
    static = hyp.plot(data, fmt='o-', show=False).axes[0]
    obs = [np.column_stack(ln.get_data_3d()) for ln in static.lines
           if ln.get_label() == '_nolegend_']
    n_grid = anim.n_frames          # the frame grid the line is drawn from
    for line, pts in zip(lines, obs):
        verts = np.column_stack(line.get_data_3d())
        marked = np.atleast_1d(line.get_markevery())
        n_verts = len(verts)
        assert n_verts > 5 * len(pts)             # smoothed, dense curve
        assert len(marked) == len(pts) == 48      # one marker per sample
        # in order along the curve, from its first vertex to its last...
        assert np.all(np.diff(marked) > 0)
        assert marked[0] == 0 and marked[-1] == n_verts - 1
        # ...each within one frame-grid row of its sample's place along it
        per_row = (n_verts - 1) / (n_grid - 1)
        place = np.arange(48) * (n_verts - 1) / 47
        assert np.abs(marked - place).max() <= per_row
        # and, in space, within half a grid row of the sample itself (the
        # static figure's coordinates differ from the animation's by the
        # small offset the curve's exact endpoints show)
        offset = np.linalg.norm(verts[[0, -1]] - pts[[0, -1]], axis=1).max()
        row_len = np.linalg.norm(np.diff(verts, axis=0), axis=1).sum() / (
            n_grid - 1)
        gap = np.linalg.norm(verts[marked] - pts, axis=1)
        assert gap.max() <= 0.5 * row_len + offset
    # the legend handle is the same artist, so it keeps the marker
    assert [h.get_marker() for h in ax.get_legend().legend_handles] == \
        [kw.get('marker', 'o')] * 2


def test_animated_window_marks_only_the_revealed_samples():
    data = _walk48()
    anim = hyp.plot(data, animate=True, fmt='o-', duration=3,
                    frame_rate=20, show=False)
    anim.draw_frame(anim.n_frames // 2)
    for line in anim.figure.axes[0].lines:
        if line.get_marker() in (None, 'None', '', ' '):
            continue
        marked = np.atleast_1d(line.get_markevery())
        n_verts = len(line.get_xdata())
        # a fraction of the 48 samples, and far fewer than the vertices
        assert 0 < len(marked) < 48
        assert len(marked) * 5 < n_verts
        assert marked.max() < n_verts


# --- B: continuous-hue markers carry alpha= (matplotlib) --------------------

def _marker_alphas(ax):
    from matplotlib.collections import PathCollection
    return [np.unique(np.round(c.get_facecolors()[:, 3], 6)).tolist()
            for c in ax.collections if isinstance(c, PathCollection)]


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('fmt', ['o', '-o'])
def test_continuous_hue_markers_honour_alpha(fmt, ndims):
    data = _walks(2, rows=30)
    hue = [np.linspace(0, 1, 30), np.linspace(1, 0, 30)]
    fig = hyp.plot(data, hue=hue, fmt=fmt, alpha=0.7, ndims=ndims,
                   show=False)
    assert _marker_alphas(fig.axes[0]) == [[0.7], [0.7]]
    # the reference: the same plot without hue= honours alpha= too
    plain = hyp.plot(data, fmt=fmt, alpha=0.7, ndims=ndims, show=False)
    assert {ln.get_alpha() for ln in plain.axes[0].lines} == {0.7}
    # a per-dataset alpha list reaches each dataset's markers
    fig = hyp.plot(data, hue=hue, fmt=fmt, alpha=[1.0, 0.4], ndims=ndims,
                   show=False)
    assert _marker_alphas(fig.axes[0]) == [[1.0], [0.4]]
