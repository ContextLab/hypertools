# -*- coding: utf-8 -*-
"""A continuous `hue=` travels with the data in a plotly ANIMATION.

1.1 release review: (3-D) animation frames rewrote only a trace's x/y/z,
so the per-vertex colours stayed aligned to the FULL curve and a sliding
window was painted with the colours of the trajectory's first rows; (2-D and
1-D) the multicoloured line was one plotly trace per segment, but every
frame sent a single trace mapped to index 0 -- segment 0 was overwritten
with the whole window in one colour while the other segments stayed on
screen, so the full trajectory never went away. The parity reference is
matplotlib's animated hue (`plot._apply_multicolor_animation`): per-dataset
head (and trail) collections re-sliced to exactly the revealed window, with
their own per-segment colours.

Real figures; each frame is rendered as `_frame_snapshots` renders it for
export (base traces updated with the frame's payload), and the visual
claims are checked on kaleido pixels.
"""

import io

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools.plot.plotly_backend import _frame_snapshots

go = pytest.importorskip('plotly.graph_objects')

N = 40
HUE = np.linspace(0, 1, N)


def _walk(d, seed=0):
    return np.cumsum(np.random.default_rng(seed).standard_normal((N, d)), 0)


def _plot(d, animate='window', **kw):
    # a 0.4 s window over a 2 s animation: the window really slides
    kw.setdefault('tail_duration', 0.4)
    return hyp.plot(_walk(d), backend='plotly', show=False, hue=HUE,
                    palette='viridis', animate=animate, duration=2,
                    frame_rate=5, **kw)


def _rgb(color):
    inner = color[color.index('(') + 1:-1].split(',')
    return tuple(round(float(v)) for v in inner[:3])


def _drawn_segments(snapshot):
    """(start, end, rgb) of every visible data-line segment of a 2-D/1-D
    snapshot, whatever traces carry them."""
    segs = []
    for t in snapshot.data:
        if (t.meta or {}).get('hyp_trace_index') is None \
                or t.mode != 'lines' or t.x is None:
            continue
        x = np.asarray(t.x, dtype=float)
        y = np.asarray(t.y, dtype=float)
        color = t.line.color
        for j in range(len(x) - 1):
            if np.isfinite(x[j:j + 2]).all() and np.isfinite(y[j:j + 2]).all():
                segs.append(((x[j], y[j]), (x[j + 1], y[j + 1]),
                             _rgb(color)))
    return segs


def _static_segment_colors(fig):
    """The rgb of every segment of the full static 2-D curve, by its
    endpoints, from the animated figure's own full-curve base traces."""
    snap = go.Figure(fig)
    snap.frames = ()
    return {(round(a[0], 9), round(a[1], 9), round(b[0], 9),
             round(b[1], 9)): c for a, b, c in _drawn_segments(snap)}


# --------------------------------------------------------------------------
# 3-D: the colour arrays are sliced with the window


@pytest.mark.parametrize('style', ['window', True, 'serial'])
def test_3d_frames_send_the_colours_of_their_window(style):
    fig = _plot(3, animate=style)
    base = [t for t in fig.data
            if (t.meta or {}).get('hyp_trace_index') == 0][0]
    idx = fig.data.index(base)
    full = list(base.line.color)
    xyz = np.column_stack([base.x, base.y, base.z])
    checked = 0
    for frame in fig.frames:
        for k, tr in zip(frame.traces, frame.data):
            if k != idx or tr.x is None or len(tr.x) < 2:
                continue
            colors = list(tr.line.color)
            assert len(colors) == len(tr.x)
            # the window's first vertex is some vertex of the full curve;
            # the frame paints it (and the rest) in THAT vertex's colours
            first = np.array([tr.x[0], tr.y[0], tr.z[0]])
            j = int(np.argmin(np.abs(xyz - first).sum(axis=1)))
            assert colors == full[j:j + len(colors)]
            checked += j > 0
    if style != 'serial':       # (one serial dataset grows from row 0)
        assert checked > 0      # some window really started past row 0


def test_3d_window_frame_renders_the_windows_colours():
    """Pixels: a late window of a viridis ramp is yellow-green, not the
    purple the ramp STARTS with."""
    fig = _plot(3, animate='window', linewidth=6)
    snaps = list(_frame_snapshots(fig))
    from PIL import Image
    img = np.asarray(Image.open(io.BytesIO(snaps[-1].to_image(
        format='png', width=600, height=450))).convert('RGB')).astype(int)
    sat = (img.max(axis=2) - img.min(axis=2)) > 60
    px = img[sat]
    assert len(px) > 100
    # viridis' late colours are green/yellow (G high); its start is purple
    # (B > G) -- a last-frame window painted from row 0 would be purple
    assert np.mean(px[:, 1] > px[:, 2]) > 0.8


# --------------------------------------------------------------------------
# 2-D / 1-D: the multicoloured line animates


@pytest.mark.parametrize('style', ['window', True, 'serial'])
def test_2d_frames_draw_only_the_window_in_its_own_colours(style):
    fig = _plot(2, animate=style)
    static = _static_segment_colors(fig)
    data_idx = {k for k, t in enumerate(fig.data)
                if (t.meta or {}).get('hyp_trace_index') is not None}
    sizes = []
    for frame, snap in zip(fig.frames, _frame_snapshots(fig)):
        # every data trace is rewritten by every frame: nothing of the
        # full trajectory is left standing
        assert data_idx <= set(frame.traces)
        segs = _drawn_segments(snap)
        sizes.append(len(segs))
        for a, b, c in segs:
            key = (round(a[0], 9), round(a[1], 9), round(b[0], 9),
                   round(b[1], 9))
            assert key in static, 'a drawn segment is not on the curve'
            # the segment wears (within the colour binning) its OWN colour
            assert max(abs(u - v) for u, v in zip(c, static[key])) <= 12
    # the reveal really moves: windows differ in size/position over time
    assert len(set(sizes)) > 1
    assert min(sizes) < len(static)


def test_2d_window_frames_have_many_colours_not_one():
    # a window spanning the whole ramp (tail_duration = duration)
    fig = _plot(2, animate=True, tail_duration=2)
    snap = list(_frame_snapshots(fig))[-1]
    colours = {c for _, _, c in _drawn_segments(snap)}
    assert len(colours) > 5


@pytest.mark.parametrize('flag', ['chemtrails', 'precog', 'bullettime'])
def test_2d_trails_are_multicoloured_and_translucent(flag):
    fig = _plot(2, animate=True, tail_duration=0.4, **{flag: True})
    trail_traces = [t for t in fig.data
                    if (t.meta or {}).get('hyp_trail_index') == 0]
    assert trail_traces
    snaps = list(_frame_snapshots(fig))
    mid = snaps[len(snaps) // 2]
    trail = [t for t in mid.data if (t.meta or {}).get('hyp_trail_index') == 0]
    drawn = [t for t in trail if t.x is not None
             and np.isfinite(np.asarray(t.x, dtype=float)).sum() > 1]
    assert len({t.line.color for t in drawn}) > 2
    assert all('0.3)' in t.line.color.replace(' ', '') for t in drawn)


def test_1d_hue_animation_is_refused_like_matplotlib():
    """1-D (non-series) animations raise on both backends (batch 1)."""
    with pytest.raises(ValueError, match='Animations are only supported'):
        hyp.plot(_walk(1)[:, 0], backend='plotly', show=False, hue=HUE,
                 animate=True, duration=1, frame_rate=4)


def test_2d_mid_frame_renders_only_the_window():
    """Pixels: a mid-animation window frame draws the moving window, not
    the whole trajectory (which inked every segment on every frame)."""
    from PIL import Image
    fig = _plot(2, animate='window', linewidth=3)
    snaps = list(_frame_snapshots(fig))
    full = go.Figure(fig)
    full.frames = ()
    full.layout.updatemenus = ()

    def ink(f):
        img = np.asarray(Image.open(io.BytesIO(f.to_image(
            format='png', width=500, height=400))).convert('RGB')).astype(int)
        return int(((img.max(axis=2) - img.min(axis=2)) > 60).sum())

    assert ink(snaps[len(snaps) // 2]) < 0.6 * ink(full)


def test_2d_frames_draw_the_palettes_own_colours_exactly():
    """A continuous hue maps through the 100-colour ramp, and the bins
    never exceed that: every drawn colour IS a ramp colour."""
    from hypertools.plot.colors import continuous_colormap
    ramp = {tuple(round(float(v) * 255) for v in c[:3])
            for c in continuous_colormap('viridis').colors}
    fig = _plot(2, animate=True, tail_duration=2)
    for snap in list(_frame_snapshots(fig))[::3]:
        for _, _, c in _drawn_segments(snap):
            assert c in ramp, c


def test_colour_bins_are_exact_up_to_the_cap_and_close_beyond_it():
    from hypertools.plot.plotly_backend import (HUE_ANIM_MAX_BINS,
                                                _hue_line_bins, _parse_rgba,
                                                _rgb_string)
    rng = np.random.default_rng(0)
    few = [_rgb_string(c) for c in rng.random((30, 3))]
    colors = [few[j] for j in rng.integers(0, 30, 500)]
    bins = _hue_line_bins(colors)
    assert [bins['colors'][k] for k in bins['seg_bin']] == colors[:-1]
    # a smooth blend with more distinct colours than the cap: each segment
    # takes a close bin colour (a matrix hue's mixtures)
    t = np.linspace(0, 1, 900)
    blend = np.column_stack([t, 1 - t, 0.5 + 0.5 * np.sin(6 * t)])
    colors = [_rgb_string(c) for c in blend]
    bins = _hue_line_bins(colors)
    assert len(bins['colors']) <= HUE_ANIM_MAX_BINS
    want = np.array([_parse_rgba(c)[:3] for c in colors[:-1]])
    got = np.array([_parse_rgba(bins['colors'][k])[:3]
                    for k in bins['seg_bin']])
    assert np.abs(want - got).max() <= 8


def test_3d_trails_send_their_windows_colours_at_trail_opacity():
    fig = _plot(3, animate=True, chemtrails=True)
    trail = [t for t in fig.data
             if (t.meta or {}).get('hyp_trail_index') == 0]
    assert len(trail) == 1
    idx = fig.data.index(trail[0])
    assert trail[0].opacity == pytest.approx(0.3)
    base = [t for t in fig.data
            if (t.meta or {}).get('hyp_trace_index') == 0][0]
    head_rgb = {_rgb(c) for c in base.line.color}
    checked = 0
    for frame in fig.frames:
        for k, tr in zip(frame.traces, frame.data):
            if k != idx or tr.x is None or len(tr.x) < 2:
                continue
            colours = [_rgb(c) for c in tr.line.color]
            assert len(colours) == len(tr.x)
            # the trail wears the head's own colours ...
            assert set(colours) <= head_rgb
            # ... at the trail's opacity
            assert tr.opacity == pytest.approx(0.3)
            checked += 1
    assert checked > 0


def test_2d_marker_line_animates_its_observation_markers():
    fig = _plot(2, animate='window', fmt='o-')
    markers = [t for t in fig.data
               if (t.meta or {}).get('hyp_trace_index') == 0
               and t.mode == 'markers']
    assert len(markers) == 1
    idx = fig.data.index(markers[0])
    counts = []
    for frame in fig.frames:
        tr = dict(zip(frame.traces, frame.data))[idx]
        assert len(tr.marker.color) == len(tr.x)
        counts.append(len(tr.x))
    # the window shows a moving handful of observations, never all 40
    assert 0 < max(counts) < N


def test_dataset_fade_reaches_every_colour_bin_of_a_dataset():
    """`FrameContext.artists` holds ONE artist per dataset: a multicoloured
    2-D dataset's colour-bin traces arrive as one `PlotlyTraceGroup`, so
    `dataset_fade=` fades all of them."""
    from hypertools.plot.plotly_backend import PlotlyTraceGroup
    seen = []

    def grab(ctx):
        seen.append(ctx.artists)

    fig = hyp.plot([_walk(2), _walk(2, seed=1) + 4], backend='plotly',
                   show=False, hue=[HUE, HUE], palette='viridis',
                   animate=True, order='serial', duration=2, frame_rate=5,
                   dataset_fade=(0.2, 0.5), on_frame=grab)
    assert seen and all(len(a) == 2 for a in seen)
    assert all(isinstance(a, PlotlyTraceGroup) for arts in seen
               for a in arts)
    # the last frame: dataset 0 (revealed first) is faded in EVERY bin
    last = fig.frames[-1]
    ds0 = {k for k, t in enumerate(fig.data)
           if (t.meta or {}).get('hyp_trace_index') == 0}
    opac = [tr.opacity for k, tr in zip(last.traces, last.data) if k in ds0]
    assert opac and all(o is not None and o < 1 for o in opac)
    assert len(set(opac)) == 1
