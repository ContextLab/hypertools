# -*- coding: utf-8 -*-
"""Plotly markers sit on the TRUE observations of an antialiased line.

`plot()`'s ``antialias=`` docstring promises that smoothing changes only how
a LINE is drawn and that markers "always render at the true sample points".
The 1.1 release review found the plotly backend drawing a ``'o-'`` trace as
ONE ``lines+markers`` trace over the ~900-vertex smoothed curve, so it put a
marker on every interpolated vertex (945 markers for 60 observations) and
the line rendered as a thick tube of overlapping dots, in 1-D, 2-D and 3-D;
``forecast_fmt='ro:'`` did the same to a dotted forecast; and a continuous
``hue=`` with ``'o-'`` in 1-D/2-D drew no markers at all.

Every assertion reads a real figure: the trace properties plotly receives,
and (for the visual claims) kaleido-rendered pixels.
"""

import io

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools.plot.plotly_backend import _marker_size_px

pytest.importorskip('plotly')


def _walk(n=60, d=3, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((n, d)), axis=0)


def _data_traces(fig, index=0):
    return [t for t in fig.data
            if (t.meta or {}).get('hyp_trace_index') == index]


def _marker_traces(fig, index=0):
    return [t for t in _data_traces(fig, index)
            if t.mode and 'markers' in t.mode]


def _coords(trace):
    cols = [np.asarray(trace.x, dtype=float), np.asarray(trace.y, dtype=float)]
    if trace.type == 'scatter3d':
        cols.append(np.asarray(trace.z, dtype=float))
    return np.column_stack(cols)


def _marked_vertices(trace):
    """The vertices a trace draws a visible marker at."""
    xyz = _coords(trace)
    size = trace.marker.size
    if size is None or np.isscalar(size):
        return xyz
    size = np.asarray(size, dtype=float)
    return xyz[:len(size)][size[:len(xyz)] > 0]


def _png(fig, width=600, height=450):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(
        fig.to_image(format='png', width=width, height=height)))
        .convert('RGB')).astype(int)


# --------------------------------------------------------------------------
# finding 1: 'o-' data traces


@pytest.mark.parametrize('ndims', [1, 2, 3])
def test_o_dash_marks_every_observation_and_nothing_else(ndims):
    x = _walk()[:, :ndims] if ndims > 1 else _walk()[:, 0]
    smooth = hyp.plot(x, backend='plotly', show=False, fmt='o-')
    raw = hyp.plot(x, backend='plotly', show=False, fmt='o-',
                   antialias=False)
    tr, = _marker_traces(smooth)
    raw_tr, = _marker_traces(raw)
    # the line itself IS smoothed (many more vertices than rows) ...
    assert len(tr.x) > 5 * 60
    # ... but exactly the 60 observations carry a marker: evenly spaced
    # vertices of the smooth curve (every sample is one of its vertices) ...
    sizes = np.asarray(tr.marker.size, dtype=float)
    marked_idx = np.flatnonzero(sizes)
    assert len(marked_idx) == 60
    assert marked_idx[0] == 0 and marked_idx[-1] == len(tr.x) - 1
    assert len(set(np.diff(marked_idx))) == 1
    # ... that ARE the observations: the un-smoothed figure's vertices up
    # to the per-axis centring/scaling (which the extra vertices shift)
    marked = _marked_vertices(tr)
    raw_xyz = _coords(raw_tr)
    for d in range(1 if ndims == 1 else 0, marked.shape[1]):
        a, b = np.polyfit(raw_xyz[:, d], marked[:, d], 1)
        np.testing.assert_allclose(a * raw_xyz[:, d] + b, marked[:, d],
                                   atol=1e-9)
    # each at the fmt's own marker size
    assert set(sizes[sizes > 0]) == {_marker_size_px(6.0, 'o', ndims)}


@pytest.mark.parametrize('kw', [dict(marker='o'), dict(markers='o')])
@pytest.mark.parametrize('ndims', [2, 3])
def test_marker_kwarg_on_a_line_marks_only_the_observations(kw, ndims):
    """`marker=`/its `markers=` alias on a line style is a marker+line
    trace like 'o-': the same observation-only markers."""
    tr, = _marker_traces(hyp.plot(_walk()[:, :ndims], backend='plotly',
                                  show=False, **kw))
    assert len(tr.x) > 5 * 60
    assert len(_marked_vertices(tr)) == 60


@pytest.mark.parametrize('ndims', [1, 2])
def test_observation_markers_keep_scalar_marker_look_in_2d(ndims):
    """A per-point size array is a plotly 'bubble' trace, whose defaults
    are 70%-opaque markers with a white outline; an observation marker must
    look exactly like an ordinary one (the matplotlib marker is opaque and
    outlined in its own colour)."""
    x = _walk()[:, :ndims] if ndims > 1 else _walk()[:, 0]
    tr, = _marker_traces(hyp.plot(x, backend='plotly', show=False,
                                  fmt='o-'))
    assert tr.marker.opacity == 1
    assert tr.marker.line.width == 0


def test_o_dash_line_is_not_a_tube_of_markers_rendered():
    """Pixels: the smoothed 'o-' figure inks about as much as the raw 'o-'
    figure (same 60 markers, same line), not a solid band of ~900 dots."""
    x = _walk(seed=3)[:, :2]

    def ink(**kw):
        img = _png(hyp.plot(x, backend='plotly', show=False, fmt='o-',
                            color='steelblue', **kw))
        # steelblue-ish pixels (the frame square is black, bg white)
        return int(((img[..., 2] - img[..., 0]) > 60).sum())

    smooth, raw = ink(), ink(antialias=False)
    assert smooth < 1.25 * raw, (smooth, raw)


def test_observation_marker_renders_like_an_ordinary_marker():
    """Pixels: an observation marker on the smoothed line is drawn exactly
    as the un-smoothed figure draws its (scalar-size) markers -- fully
    opaque, in the trace colour, without the white outline plotly's bubble
    defaults add (which drew it at 70% opacity over white)."""
    rng = np.random.default_rng(5)
    x = np.column_stack([np.linspace(-1, 1, 7), rng.uniform(-.5, .5, 7)])

    def steelblue_pixels(**kw):
        img = _png(hyp.plot(x, backend='plotly', show=False, fmt='o-',
                            color='steelblue', markersize=14, reduce=None,
                            **kw))
        return int((np.abs(img - [70, 130, 180]).sum(axis=2) <= 6).sum())

    smooth, raw = steelblue_pixels(), steelblue_pixels(antialias=False)
    # 7 discs of ~19 px diameter: far more than the thin line alone
    assert raw > 7 * 150
    assert abs(smooth - raw) <= 0.1 * raw, (smooth, raw)


def test_animation_frames_keep_markers_on_observations():
    x = _walk(n=30)
    fig = hyp.plot(x, backend='plotly', show=False, fmt='o-', animate=True,
                   duration=1, frame_rate=6)
    base, = _marker_traces(fig)
    sizes = np.asarray(base.marker.size, dtype=float)
    # the full curve marks the 30 observations (each at the frame-grid
    # vertex nearest it: an animation draws a resampled frame grid)
    assert int((sizes > 0).sum()) == 30
    observed = {tuple(np.round(v, 12)) for v in _marked_vertices(base)}
    idx = fig.data.index(base)
    checked = 0
    for frame in fig.frames:
        for k, tr in zip(frame.traces, frame.data):
            if k != idx or tr.x is None or len(tr.x) == 0:
                continue
            # every frame sends the sizes of its own window ...
            fs = np.asarray(tr.marker.size, dtype=float)
            assert len(fs) == len(tr.x)
            # ... so what it marks is observations, never an in-between
            # vertex of the moving window
            marked = {tuple(np.round(v, 12)) for v in _marked_vertices(tr)}
            assert marked and marked <= observed
            checked += len(tr.x) > 10 * len(marked)
    # (and the smoothed windows do carry many unmarked vertices)
    assert checked > 0


def test_trail_traces_mark_observations_only():
    x = _walk(n=30)
    fig = hyp.plot(x, backend='plotly', show=False, fmt='o-', animate=True,
                   chemtrails=True, duration=1, frame_rate=6)
    trails = [t for t in fig.data
              if t.mode == 'lines+markers' and t.showlegend is False
              and (t.meta or {}).get('hyp_trace_index') is None
              and not (t.meta or {}).get('hyp_forecast_role')
              and t.type == 'scatter3d' and t.x is not None
              and len(t.x) == 0]
    assert trails, [t.mode for t in fig.data]
    sizes = np.asarray(trails[0].marker.size, dtype=float)
    assert sizes.ndim == 1 and int((sizes > 0).sum()) == 30


def test_observation_vertices_follow_the_observations_not_one_grid():
    """The observation -> vertex mapping is read from the data, not from
    one resampling grid's arithmetic: a non-uniform grid that contains the
    observations (as an animation grid that keeps every sample would) maps
    each to its exact vertex, and a uniform grid that does not maps each to
    the nearest vertex."""
    from hypertools.plot.plotly_backend import _observation_vertices
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((6, 3))
    # non-uniform: 0, 3, 4, 10, 11, 19 hold the samples, noise elsewhere
    where = [0, 3, 4, 10, 11, 19]
    dense = rng.standard_normal((20, 3)) + 10
    dense[where] = raw
    np.testing.assert_array_equal(
        _observation_vertices(dense, raw, 20, 1), where)
    # uniform grid without the samples: nearest by the shared parameter
    grid = rng.standard_normal((11, 3)) + 10
    np.testing.assert_array_equal(
        _observation_vertices(grid, raw, 11, 1), [0, 2, 4, 6, 8, 10])
    # the rows ARE the observations: every aa_step-th vertex
    np.testing.assert_array_equal(
        _observation_vertices(grid, None, 6, 3), [0, 3, 6, 9, 12, 15])


# --------------------------------------------------------------------------
# finding 2: continuous hue + 'o-' in 1-D/2-D draws its markers


@pytest.mark.parametrize('ndims', [1, 2])
def test_continuous_hue_o_dash_draws_one_marker_per_observation(ndims):
    x = _walk(n=30)[:, :ndims] if ndims > 1 else _walk(n=30)[:, 0]
    hue = np.linspace(0, 1, 30)
    fig = hyp.plot(x, backend='plotly', show=False, fmt='o-', hue=hue,
                   palette='viridis')
    markers = _marker_traces(fig)
    assert len(markers) == 1
    m = markers[0]
    assert len(_marked_vertices(m)) == 30
    # coloured per observation, in the hue's own colours
    colors = list(m.marker.color)
    assert len(colors) == 30 and len(set(colors)) > 20
    # and still one trajectory to a reader counting data traces by tag
    assert {(t.meta or {}).get('hyp_trace_index')
            for t in _data_traces(fig)} == {0}


def test_continuous_hue_o_dash_2d_renders_markers():
    x = _walk(n=30, seed=4)[:, :2]
    hue = np.linspace(0, 1, 30)
    with_markers = _png(hyp.plot(x, backend='plotly', show=False, fmt='o-',
                                 hue=hue, palette='viridis'))
    line_only = _png(hyp.plot(x, backend='plotly', show=False, fmt='-',
                              hue=hue, palette='viridis'))
    ink = [int((np.abs(img - 255).sum(axis=2) > 60).sum())
           for img in (with_markers, line_only)]
    # 30 visible marker discs add real ink on top of the thin line
    assert ink[0] > ink[1] + 30 * 20, ink


# --------------------------------------------------------------------------
# finding 5: forecast_fmt markers


def _forecast_traces(fig, role='static'):
    return [t for t in fig.data
            if (t.meta or {}).get('hyp_forecast_role') == role]


@pytest.mark.parametrize('ndims', [2, 3])
def test_forecast_fmt_markers_only_on_forecast_steps(ndims):
    data = _walk(n=40)[:, :ndims]
    t = 5
    fig = hyp.plot(data, backend='plotly', show=False, t=t, predict='Kalman',
                   forecast_fmt='ro:')
    fc, = _forecast_traces(fig)
    assert fc.mode == 'lines+markers'
    assert len(fc.x) > 10 * (t + 1)          # the dotted line is smoothed
    # the seam (last observation) plus the t forecast steps are marked
    assert len(_marked_vertices(fc)) == t + 1
    if ndims < 3:
        assert fc.marker.opacity == 1 and fc.marker.line.width == 0


def test_animated_forecast_fmt_markers_only_on_forecast_steps():
    data = _walk(n=30)
    t = 4
    fig = hyp.plot(data, backend='plotly', show=False, t=t, predict='Kalman',
                   forecast_fmt='ro:', animate=True, duration=1,
                   frame_rate=6)
    live, = _forecast_traces(fig, 'live')
    idx = fig.data.index(live)
    checked = 0
    for frame in fig.frames:
        for k, tr in zip(frame.traces, frame.data):
            if k != idx or tr.x is None or len(tr.x) < 2:
                continue
            sizes = np.asarray(tr.marker.size, dtype=float)
            assert len(sizes) == len(tr.x)
            assert int((sizes > 0).sum()) == t + 1
            checked += 1
    assert checked > 0
