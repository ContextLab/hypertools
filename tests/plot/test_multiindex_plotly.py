# -*- coding: utf-8 -*-
"""Backend parity for hierarchies: exact counts, styles, hue, forecasts.

The matplotlib expectations these mirror live in test_column_multiindex.py,
test_multiindex_hue.py and test_multiindex_predict.py. `>= 3` was the v2
assertion; every count here is exact, because a duplicated or extra trace
is precisely the failure mode parity work introduces.

Three things about plotly's figure make the obvious helpers wrong, and each
cost a real defect here:

1. `fig.data` is NOT a list of data traces. It also carries the black
   wireframe cube (`_cube_trace`), any 2-D density/surface layer, the
   `predict=` overlays and -- with `colorbar=True` -- a phantom colorbar
   trace. None of them has a `name`, so `name != 'cube'` (and
   `type in ('scatter3d', 'scatter')`) keeps every one of them. Data traces
   are found by their `meta['hyp_trace_index']` tag, the plotly half of
   matplotlib's `coll._hyp_trace_index`.
2. A forecast INHERITS its source trace's dash (`_forecast_style_from`), and
   plotly spells a solid line `dash='solid'` rather than leaving it unset --
   so under `fmt='-'` `getattr(t.line, 'dash', None)` is truthy for *every*
   line, and "the dashed ones" selects the whole figure. Forecasts are found
   by `meta['hyp_forecast_role']`.
3. Line widths are in PIXELS here and in POINTS on matplotlib. The parity
   claim is `plotly_px == matplotlib_pt * PT_TO_PX`, so that constant is
   imported rather than the pt values hard-coded.
"""
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp

pytest.importorskip('plotly')

from hypertools.plot.plotly_backend import PT_TO_PX   # noqa: E402


def market_frame(T=60, seed=0):
    rng = np.random.default_rng(seed)
    tuples = [('Market', sector, m)
              for sector in ('Tech', 'Energy')
              for m in ('return', 'volatility', 'momentum')]
    cols = pd.MultiIndex.from_tuples(tuples,
                                     names=['Market', 'Sector', 'Measure'])
    return pd.DataFrame(rng.normal(size=(T, 6)).cumsum(axis=0) + 100.0, columns=cols)


def two_level_frame(T=40, seed=0):
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_tuples(
        [(g, f) for g in ('A', 'B', 'C') for f in ('f0', 'f1', 'f2')],
        names=['Group', 'Feature'])
    return pd.DataFrame(rng.normal(size=(T, 9)).cumsum(axis=0), columns=cols)


def subject_frame(n_time=10, seed=0):
    """A ROW hierarchy whose innermost index tuple REPEATS, so each leaf
    keeps `n_time` rows: 2 conds x 3 subjs x n_time timepoints.

    The same construction as `tests/plot/test_multiindex_predict.py`'s
    `multirow_row_frame`, restated here because this module's subject is the
    plotly rendering of it, not the grouping: 6 leaves plus 2 top-level
    (per-cond) means = 8 final traces, every one of them `n_time` rows, so
    every trace clears Contract 10's >= 2-row precondition.
    """
    rng = np.random.default_rng(seed)
    tuples, rows = [], []
    for ci, cond in enumerate(['cond1', 'cond2']):
        for si in range(3):
            rows.append(rng.standard_normal((n_time, 3)).cumsum(axis=0)
                        + ci * 5.0)
            tuples.extend([(cond, f'S{si}')] * n_time)
    idx = pd.MultiIndex.from_tuples(tuples, names=['cond', 'subj'])
    return pd.DataFrame(np.vstack(rows), index=idx, columns=['x', 'y', 'z'])


def _data_traces(fig):
    """The OBSERVED data traces -- see this module's docstring, point 1."""
    return [t for t in fig.data
            if (t.meta or {}).get('hyp_trace_index') is not None]


def _forecast_traces(fig):
    """The `predict=` overlays -- see this module's docstring, point 2."""
    return [t for t in fig.data
            if (t.meta or {}).get('hyp_forecast_role') is not None]


def _rgb(rgba):
    """The three colour channels of an ``rgb(...)``/``rgba(...)`` string.

    Parses rather than string-slices: the observed traces print ``rgb(r,g,b)``
    (`_rgb_string`) and the forecasts ``rgba(r,g,b,a)`` (`_to_plotly_color`),
    so `rsplit(',', 1)[0]` yields two strings that can never compare equal
    even when the colour is identical.
    """
    body = str(rgba).split('(', 1)[1].rstrip(')')
    return tuple(int(v) for v in body.split(',')[:3])


def _alpha(rgba):
    return float(str(rgba).rstrip(')').rsplit(',', 1)[1])


def _plot(*args, **kwargs):
    hyp.set_interactive_backend('plotly')
    try:
        return hyp.plot(*args, **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')


def _mpl(*args, **kwargs):
    """The matplotlib half of a parity claim.

    Every "plotly does X" assertion in this module that is really a PARITY
    assertion states it against the figure this returns, rather than against
    a hard-coded expectation copied out of another module -- a copied number
    goes stale silently, and both backends drifting together is exactly the
    failure the parity directive is about.
    """
    import matplotlib
    matplotlib.use('Agg')
    return hyp.plot(*args, **kwargs)


def _mpl_rgb(color):
    import matplotlib.colors as mcolors
    return tuple(int(round(255 * v)) for v in mcolors.to_rgb(color))


def _mpl_forecasts(fig, role='static'):
    """{dataset -> artist} for the matplotlib forecast overlays.

    Keyed by `_hyp_forecast_dataset` -- the matplotlib spelling of plotly's
    `meta['hyp_dataset']` -- so the two backends are compared series by
    series rather than by list position.
    """
    return {ln._hyp_forecast_dataset: ln for ln in fig.axes[0].lines
            if getattr(ln, '_hyp_forecast_role', None) == role}


def _mpl_hue_collections(fig):
    """The per-trace `LineCollection`s a continuous hue draws, in trace order.

    NOT `ax.collections`: the 3-D bounding cube is six `Line3DCollection`
    wireframe faces (`matplotlib_backend._draw_cube`), so an unfiltered list
    is 6 longer than the trace list. `_hyp_trace_index` is the tag that tells
    them apart -- the matplotlib half of plotly's `meta['hyp_trace_index']`.
    """
    colls = [c for c in fig.axes[0].collections
             if getattr(c, '_hyp_trace_index', None) is not None]
    return sorted(colls, key=lambda c: c._hyp_trace_index)


def test_three_level_column_hierarchy_exact_trace_count_and_order():
    fig = _plot(market_frame(), '-', show=False)
    traces = _data_traces(fig)
    assert len(traces) == 3, '2 sector leaves + 1 market mean'
    assert traces[-1].name == 'Market'
    # the tag is positional and complete: the leaves come first, in order
    assert [(t.meta or {})['hyp_trace_index'] for t in traces] == [0, 1, 2]


def test_plotly_widths_match_the_documented_formula():
    traces = _data_traces(_plot(market_frame(), '-', show=False))
    # matplotlib draws this hierarchy at 1.0/1.0/2.0 POINTS
    # (test_column_multiindex.py); plotly's line.width is in pixels.
    assert [t.line.width for t in traces] == pytest.approx(
        [1.0 * PT_TO_PX, 1.0 * PT_TO_PX, 2.0 * PT_TO_PX])


def test_plotly_opacities_match_the_documented_formula():
    traces = _data_traces(_plot(market_frame(), '-', show=False))
    assert [_alpha(t.line.color) for t in traces] == pytest.approx(
        [0.7, 0.7, 1.0])


def test_plotly_hue_opacities_match_matplotlib():
    """Step 3 of Task 9 asks for "alpha baked into the `rgba()` string" --
    and a CONTINUOUS hue is the case where that is the ONLY place it can
    live, because the per-point colour array replaces the trace colour.

    matplotlib bakes it into the segment colours it swaps the line artists
    for (`plot._apply_multicolor_lines`: `seg_colors` gains a 4th channel
    from `tkwargs['alpha']`); plotly serialized the same colours through
    `_rgb_string`, which drops the 4th channel, and set no trace `opacity` --
    so this hierarchy's 0.7 leaves, and a plain `alpha=`, rendered FULLY
    OPAQUE on plotly alone. `test_plotly_opacities_match_the_documented_
    formula` could not see it: it plots no `hue=`.
    """
    df = market_frame()
    hues = [np.linspace(0.0, 1.0, len(df)), np.linspace(9.0, 10.0, len(df))]
    kw = dict(hue=hues, palette='viridis', antialias=False, show=False)
    mpl_alpha = [float(c.get_colors()[-1][3])
                 for c in _mpl_hue_collections(_mpl(df, '-', **kw))]
    assert mpl_alpha == pytest.approx([0.7, 0.7, 1.0]), (
        'the hierarchy alphas themselves regressed, so the parity assertion '
        'below would be comparing two wrong numbers')
    traces = _data_traces(_plot(df, '-', **kw))
    assert [_alpha(t.line.color[-1]) for t in traces] == pytest.approx(
        mpl_alpha)


def test_plotly_hue_honours_a_plain_alpha_kwarg():
    """The same defect, off the hierarchy path: `hue=` + `alpha=` on a plain
    array. Kept separate so a hierarchy-only regression and a general one
    are distinguishable from the failure report alone."""
    x = np.random.default_rng(0).normal(size=(50, 3)).cumsum(axis=0)
    kw = dict(hue=np.linspace(0, 1, 50), alpha=0.3, antialias=False,
              show=False)
    mpl_alpha = [float(c.get_colors()[-1][3])
                 for c in _mpl_hue_collections(_mpl(x, '-', **kw))]
    assert mpl_alpha == pytest.approx([0.3])
    traces = _data_traces(_plot(x, '-', **kw))
    assert [_alpha(t.line.color[-1]) for t in traces] == pytest.approx(
        mpl_alpha)


def test_plotly_legend_labels_only_the_top_level_mean():
    traces = _data_traces(_plot(market_frame(), '-', show=False))
    assert [t.showlegend for t in traces] == [False, False, True]
    assert len(set(_rgb(t.line.color) for t in traces)) == 1


def test_two_level_column_hierarchy_labels_every_trace():
    traces = _data_traces(_plot(two_level_frame(), '-', show=False))
    assert len(traces) == 3
    assert [t.name for t in traces] == ['A', 'B', 'C']
    assert [t.showlegend for t in traces] == [True, True, True]
    assert len(set(_rgb(t.line.color) for t in traces)) == 3


def test_plotly_colours_are_the_matplotlib_colours():
    """Parity down to the byte. Both backends resolve the same float RGB;
    plotly then serializes it, and `_to_plotly_color` used to TRUNCATE each
    channel while `_rgb_string` rounded -- so the same colour printed
    rgb(219,94,86) here and rounded to (219,95,87) on matplotlib, and a
    forecast anchored to a per-point colour disagreed with the very trace it
    was copied from."""
    import matplotlib
    matplotlib.use('Agg')
    df = market_frame()
    mpl_fig = hyp.plot(df, '-', show=False)
    mpl_rgb = [tuple(int(round(255 * v))
                     for v in matplotlib.colors.to_rgb(ln.get_color()))
               for ln in mpl_fig.axes[0].lines]
    ply_rgb = [_rgb(t.line.color) for t in _data_traces(_plot(df, '-',
                                                             show=False))]
    assert len(mpl_rgb) == len(ply_rgb) == 3
    assert ply_rgb == mpl_rgb


def test_continuous_price_hue_renders_per_point_colours():
    df = market_frame()
    hues = [df['Market'][s].mean(axis=1).to_numpy() for s in ('Tech', 'Energy')]
    traces = _data_traces(_plot(df, '-', hue=hues, palette='viridis',
                                show=False))
    assert len(traces) == 3
    per_point = [t for t in traces
                 if not isinstance(t.line.color, str)
                 and t.line.color is not None]
    assert len(per_point) == 3
    # one colour per DRAWN vertex. Not `len(df)`: `antialias=True` (the
    # default) densifies each line and resamples the colours onto it
    # (`_aa_resample_colors`), so this frame draws 945 vertices, not 60.
    assert all(len(t.line.color) == len(t.x) for t in per_point)
    # ...and they really vary along each trajectory, rather than being a
    # constant colour repeated once per vertex
    assert all(len(set(t.line.color)) > 1 for t in per_point)

    # with the densification off, it is exactly one colour per input row
    raw = _data_traces(_plot(df, '-', hue=hues, palette='viridis',
                             antialias=False, show=False))
    assert all(len(t.line.color) == len(df) for t in raw)


def test_colorbar_renders_on_plotly():
    """`marker.colorbar` is NOT the discriminator: plotly instantiates a
    `ColorBar` object on every trace, so `is not None` holds with
    `colorbar=False` too (measured). `marker.showscale` is what actually
    turns one on, and the `colorbar=False` half of this test proves the
    check discriminates."""
    df = market_frame()
    fig = _plot(df, '-', hue=np.linspace(0, 1, len(df)), colorbar=True,
                show=False)
    has_bar = [t for t in fig.data
               if getattr(getattr(t, 'marker', None), 'showscale', None)
               is True]
    assert len(has_bar) == 1, 'expected exactly one colorbar-bearing trace'
    assert has_bar[0].marker.colorscale, 'the colorbar carries no colorscale'

    off = _plot(df, '-', hue=np.linspace(0, 1, len(df)), colorbar=False,
                show=False)
    assert not [t for t in off.data
                if getattr(getattr(t, 'marker', None), 'showscale', None)
                is True]


def test_predict_draws_one_forecast_trace_per_drawn_trace():
    fig = _plot(market_frame(), '-', predict='Kalman', t=1, show=False)
    traces = _data_traces(fig)
    forecasts = _forecast_traces(fig)
    assert len(traces) == 3, '2 sector leaves + 1 market mean'
    assert len(forecasts) == 3, 'one forecast per FINAL trace, means included'
    assert sorted((t.meta or {})['hyp_dataset'] for t in forecasts) == [0, 1, 2]
    # Every forecast starts where the trace IT continues ends -- a MAP keyed
    # by identity, not a set of endpoints. `start in ends` reads like the
    # same claim and is not: measured, permuting the three forecasts'
    # coordinates by [1, 2, 0] (so every one of them continues the wrong
    # trajectory) leaves every start inside the set and the assertion green.
    # Without regrouping a forecast's dataset IS its trace index -- pinned
    # just above -- so the two tags are the same key here.
    ends = {(t.meta or {})['hyp_trace_index']:
            tuple(np.round([t.x[-1], t.y[-1], t.z[-1]], 9)) for t in traces}
    for f in forecasts:
        ds = (f.meta or {})['hyp_dataset']
        assert tuple(np.round([f.x[0], f.y[0], f.z[0]], 9)) == ends[ds], (
            f'forecast {ds} does not continue trace {ds}')


def test_plotly_forecast_takes_the_final_observed_hue_colour():
    """F14, plotly half -- the same rule as matplotlib.

    The two leaves are given DISJOINT hue ranges so their tail colours are
    far apart and the derived mean's lands between them: with one shared hue
    all three traces end in the same colour and a mis-paired (or entirely
    palette-derived) forecast would pass vacuously."""
    df = market_frame()
    hues = [np.linspace(0.0, 1.0, len(df)), np.linspace(9.0, 10.0, len(df))]
    fig = _plot(df, '-', hue=hues, palette='viridis', predict='Kalman', t=1,
                show=False)
    traces = _data_traces(fig)
    forecasts = _forecast_traces(fig)
    assert len(traces) == len(forecasts) == 3
    tails = [_rgb(t.line.color[-1]) for t in traces]
    assert len(set(tails)) == 3, (
        'the three traces must end in visibly different colours for this '
        f'test to be able to detect a mis-pairing (got {tails})')
    for f in forecasts:
        ds = (f.meta or {})['hyp_dataset']
        assert _rgb(f.line.color) == tails[ds], (
            f'forecast for trace {ds} is coloured {_rgb(f.line.color)}, but '
            f'that trace ends at {tails[ds]}')


@pytest.mark.parametrize('fmt', ['-', '--', 'o', 'o-'])
def test_plotly_and_matplotlib_agree_on_the_forecast_hue_colour(fmt):
    """The parity statement itself. plotly styled the forecast from
    `kwargs_list[src]['color']`, which under a continuous hue is the
    per-dataset PALETTE colour `plot.py` fills in for plotly and NOT a
    colour the trace is drawn in -- so the two backends drew the same
    forecast in unrelated colours.

    Parametrized over `fmt` because the anchor is applied unconditionally on
    plotly (`_hue_anchor_color` at every trace's build) while matplotlib
    applied it only on the LINE path (`_apply_multicolor_lines`) and not on
    the MARKER one (`_apply_multicolor_markers`) -- so a marker-only `'o'`
    drew the forecast in the hue tail on plotly and in the palette cycle on
    matplotlib. Measured: `'o'` disagreed at every `ndims`, `'-'`/`'--'`/
    `'o-'` agreed, which is precisely why a single-`fmt` test could not see
    it.
    """
    df = market_frame()
    hues = [np.linspace(0.0, 1.0, len(df)), np.linspace(9.0, 10.0, len(df))]
    kw = dict(hue=hues, palette='viridis', predict='Kalman', t=1,
              antialias=False, show=False)
    mpl = {ds: _mpl_rgb(ln.get_color())
           for ds, ln in _mpl_forecasts(_mpl(df, fmt, **kw)).items()}
    ply = {(t.meta or {})['hyp_dataset']: _rgb(t.line.color)
           for t in _forecast_traces(_plot(df, fmt, **kw))}
    assert sorted(mpl) == sorted(ply) == [0, 1, 2]
    assert mpl == ply, f'matplotlib {mpl} vs plotly {ply}'
    # ...and it is the hue TAIL both are wearing, not a shared palette
    # colour: without this, both backends drifting together would pass.
    tails = [_rgb(t.line.color[-1])
             for t in _data_traces(_plot(df, fmt, **kw))]
    assert ply == dict(enumerate(tails))


def test_row_hierarchy_forecasts_cross_to_plotly():
    """Task 9: "one dashed trace per plotted trajectory, ON EITHER AXIS, in
    the final observed colour."

    Every other forecast test in this module plots a COLUMN hierarchy, and
    no plotly test anywhere pairs `predict=` with a ROW one
    (`tests/plot/test_multiindex_predict.py` never touches plotly;
    `tests/test_multiindex.py::test_plot_2level_plotly_parity` never passes
    `predict=`), so the axis the task names explicitly was the one with no
    coverage. Measured correct -- this pins it.
    """
    df = subject_frame()
    kw = dict(predict='Kalman', t=1, antialias=False, show=False)
    fig = _plot(df, '-', **kw)
    traces = _data_traces(fig)
    forecasts = _forecast_traces(fig)
    assert len(traces) == 8, '6 subject leaves + 2 per-condition means'
    assert len(forecasts) == 8, 'one forecast per FINAL trace, means included'
    assert sorted((t.meta or {})['hyp_dataset']
                  for t in forecasts) == list(range(8))
    # each forecast wears its own source trace's colour -- the row axis's
    # half of the same rule the column tests state
    src = {(t.meta or {})['hyp_trace_index']: _rgb(t.line.color)
           for t in traces}
    assert {(t.meta or {})['hyp_dataset']: _rgb(t.line.color)
            for t in forecasts} == src
    # ...and matplotlib draws the identical colours from the identical frame
    mpl = {ds: _mpl_rgb(ln.get_color())
           for ds, ln in _mpl_forecasts(_mpl(df, '-', **kw)).items()}
    assert mpl == src


@pytest.mark.parametrize('antialias', [False, True])
def test_one_dimensional_forecast_continues_the_observed_series(antialias):
    """`ndims=1` puts the ROW INDEX on x, so a forecast has to be drawn over
    the rows that FOLLOW the observed run.

    matplotlib passed no x at all (`ax.plot(fc[:, 0], ...)`), so it defaulted
    to `0..len(fc)-1` and painted every forecast back over the START of the
    plot, while plotly built its x from the observed run's offset
    (`_aa_x(step, arr.shape[0] - 1, ...)`). The y VALUE at the seam was right
    on both, which is why nothing caught it: only the x was wrong, and every
    other forecast test in this module reads 3-D coordinates.
    """
    df = market_frame()
    kw = dict(ndims=1, predict='Kalman', t=3, antialias=antialias,
              show=False)
    mpl_fig = _mpl(df, '-', **kw)
    obs = [ln for ln in mpl_fig.axes[0].lines
           if getattr(ln, '_hyp_forecast_role', None) is None]
    fcs = _mpl_forecasts(mpl_fig)
    assert len(obs) == len(fcs) == 3
    for ds, ln in fcs.items():
        assert float(ln.get_xdata()[0]) == pytest.approx(
            float(obs[ds].get_xdata()[-1])), (
            f'forecast {ds} does not start where its observed series ends')

    fig = _plot(df, '-', **kw)
    p_obs = _data_traces(fig)
    p_fc = {(t.meta or {})['hyp_dataset']: t for t in _forecast_traces(fig)}
    assert len(p_obs) == len(p_fc) == 3
    for ds, t in p_fc.items():
        assert float(t.x[0]) == pytest.approx(float(p_obs[ds].x[-1]))
    # the parity statement: the same drawn span on both backends
    assert [float(fcs[d].get_xdata()[-1]) for d in range(3)] == pytest.approx(
        [float(p_fc[d].x[-1]) for d in range(3)])


def test_return_model_bundle_matches_the_matplotlib_bundle():
    hyp.set_interactive_backend('plotly')
    try:
        out = hyp.plot(market_frame(), '-', predict='Kalman', t=1,
                       return_model=True, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert len(out['xform_data']) == 2
    assert len(out['trace_data']) == 3
    assert len(out['predict']['forecasts']) == 3
    assert out['trace_metadata']['is_mean'] == [False, False, True]


def test_hierarchy_with_animated_prediction():
    """Prerequisite: forecast-animation Tasks 1-2 (the precomputed
    schedule) and its Task 6 (plotly parity)."""
    fig = _plot(market_frame(), '-', predict='Kalman', t=1, animate=True,
                duration=2, frame_rate=4, show=False)
    assert getattr(fig, 'frames', None), 'expected plotly animation frames'
    assert len(_data_traces(fig)) == 3
    # the animated forecast traces are preallocated EMPTY and filled per
    # frame, so this counts them by tag rather than by point count
    live = [t for t in _forecast_traces(fig)
            if (t.meta or {})['hyp_forecast_role'] == 'live']
    assert len(live) == 3, 'one live forecast trace per FINAL trace'
    assert any(len(fr.data[0].x) for fr in fig.frames), (
        'no frame reveals any data'
    )


def test_animated_forecast_hue_colour_is_the_SAME_on_both_backends():
    """Characterization, and the parity claim for the case F14 does not
    reach.

    F14 ("a forecast overlay under a continuous hue takes the final observed
    hue colour of its source trace") is implemented for the STATIC overlay on
    both backends -- `test_plotly_and_matplotlib_agree_on_the_forecast_hue_
    colour` above. An ANIMATED forecast is governed by a different, already
    resolved rule: *Decision R3*, "the colour a live/retained forecast wears
    is the HEAD RUN's, which changes from frame to frame"
    (`plot._update_forecasts._run_colour`, `plotly_backend`'s
    `forecast_frame_colors`). Under a continuous hue the head run's own line
    artist carries the per-dataset PALETTE colour, so that is what both
    backends draw -- and the two rules genuinely conflict, since R3's colour
    is per-frame while F14's is a fixed tail. Which one wins for animated
    hue is an owner decision, recorded in the plan's *Decisions (resolved)*
    #7; until it is made, this test pins the behaviour on BOTH backends so
    the divergence between the static and animated paths is visible and
    cannot change silently on one backend only.
    """
    df = market_frame()
    hues = [np.linspace(0.0, 1.0, len(df)), np.linspace(9.0, 10.0, len(df))]
    kw = dict(hue=hues, palette='viridis', predict='Kalman', t=1,
              antialias=False, animate=True, duration=2, frame_rate=4,
              show=False)
    mpl = {ds: _mpl_rgb(ln.get_color()) for ds, ln
           in _mpl_forecasts(_mpl(df, '-', **kw).figure, role='live').items()}
    ply = {(t.meta or {})['hyp_dataset']: _rgb(t.line.color)
           for t in _forecast_traces(_plot(df, '-', **kw))
           if (t.meta or {})['hyp_forecast_role'] == 'live'}
    assert sorted(mpl) == sorted(ply) == [0, 1, 2]
    assert mpl == ply, f'matplotlib {mpl} vs plotly {ply}'

    # ...and it is NOT the static rule: stated positively so that the day
    # F14 is extended to animations, this test fails and points at the
    # decision rather than silently agreeing with either outcome.
    static_kw = dict(kw, animate=False)
    del static_kw['duration'], static_kw['frame_rate']
    static = {ds: _mpl_rgb(ln.get_color())
              for ds, ln in _mpl_forecasts(_mpl(df, '-', **static_kw)).items()}
    assert mpl != static, (
        'the animated forecast now matches the static hue-tail rule (F14); '
        'if that was intentional, resolve Decisions #7 and update this test')


def test_dual_axis_frame_is_rejected_on_plotly():
    idx = pd.MultiIndex.from_product([['a', 'b'], range(20)])
    cols = pd.MultiIndex.from_tuples([('M', 'T'), ('M', 'E')])
    df = pd.DataFrame(np.zeros((40, 2)), index=idx, columns=cols)
    with pytest.raises(ValueError, match='both a row and a column MultiIndex'):
        _plot(df, '-', show=False)
