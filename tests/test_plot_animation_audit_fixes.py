# -*- coding: utf-8 -*-
"""Regression tests for the release-1.0 audit's animation findings
(fix batch B3-animation-save; units F04-plot-animate-window and
F05-plot-animate-special).

Every test drives real hypertools plots (no mocks): MPLBACKEND=Agg,
show=False, seeded data.
"""

import os
import warnings

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _spiral(n=40):
    t = np.linspace(0, 4 * np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t), np.linspace(0, 1, n)])


def _walk(n=60, d=5, seed=0):
    return np.cumsum(np.random.RandomState(seed).randn(n, d), axis=0)


def _trail_artists(ax):
    return [ln for ln in ax.lines
            if ln.get_alpha() is not None and ln.get_alpha() < 1]


# ---------------------------------------------------------------------------
# F05-001: chemtrails must never show the future (negative-slice bug) nor
# blink empty mid-animation
# ---------------------------------------------------------------------------

def test_chemtrails_never_shows_future_3d():
    r = hyp.plot([_spiral()], animate=True, chemtrails=True, duration=4,
                 tail_duration=1, frame_rate=12, show=False)
    ax = r.figure.axes[0]
    counts = {}
    for num in (0, 5, 10, 11, 12, 47):
        r.animation._draw_frame(num)
        trail = _trail_artists(ax)[0]
        counts[num] = len(trail.get_data_3d()[0])
    # nothing has left the 12-frame head window before frame 11 -- the
    # historical negative slice drew 37-47 of 48 FUTURE points here
    assert counts[0] == 0
    assert counts[5] == 0
    assert counts[10] == 0
    assert counts[11] == 0
    assert counts[12] == 1
    # and the trail is present (no blink to empty) at the final frame
    assert counts[47] == 36


def test_chemtrails_never_shows_future_2d():
    r = hyp.plot([_spiral()], ndims=2, animate=True, chemtrails=True,
                 duration=4, tail_duration=1, frame_rate=12, show=False)
    ax = r.figure.axes[0]
    counts = {}
    for num in (0, 10, 12, 47):
        r.animation._draw_frame(num)
        trail = _trail_artists(ax)[0]
        counts[num] = len(trail.get_xdata())
    assert counts[0] == 0
    assert counts[10] == 0
    assert counts[12] == 1
    assert counts[47] == 36


# ---------------------------------------------------------------------------
# F05-008: precog trail must share the head's last vertex (no gap)
# ---------------------------------------------------------------------------

def test_precog_trail_shares_head_vertex():
    r = hyp.plot([_spiral()], animate=True, precog=True, duration=4,
                 tail_duration=1, frame_rate=12, show=False)
    r.animation._draw_frame(20)
    ax = r.figure.axes[0]
    head = ax.lines[0].get_data_3d()
    trail = _trail_artists(ax)[0].get_data_3d()
    head_last = [float(v[-1]) for v in head]
    trail_first = [float(v[0]) for v in trail]
    np.testing.assert_allclose(head_last, trail_first)


# ---------------------------------------------------------------------------
# F04-001 / F05-002: continuous (per-observation) hue + animate must ANIMATE
# on matplotlib (it silently rendered the full trajectory statically)
# ---------------------------------------------------------------------------

def test_continuous_hue_window_animates_3d():
    np.random.seed(2)
    c = np.cumsum(np.random.randn(200, 6), axis=0)
    ha = hyp.plot(c, animate='window', duration=4, frame_rate=10, focused=1,
                  hue=np.arange(200.0), show=False)
    ax = ha.figure.axes[0]
    # the head collection is the FIRST collection added (cube wireframe
    # collections have exactly 4 segments each)
    ha.animation._draw_frame(10)
    seg_counts = [len(getattr(co, '_segments3d', []))
                  for co in ax.collections]
    # no static full-trajectory collection (39 segments) may remain
    assert max(seg_counts) <= 10
    head = [co for co in ax.collections
            if len(getattr(co, '_segments3d', [])) == 10][0]
    segs10 = np.array(head._segments3d)
    ha.animation._draw_frame(30)
    segs30 = np.array(head._segments3d)
    # the 1-second window (10 segments) must SLIDE: same size, new geometry
    assert segs10.shape == segs30.shape == (10, 2, 3)
    assert not np.allclose(segs10, segs30)


def test_continuous_hue_parallel_animates_2d_gif(tmp_path):
    from PIL import Image
    p = str(tmp_path / 'hue2d.gif')
    hyp.plot(_spiral(), ndims=2, animate=True, hue=np.linspace(0, 1, 40),
             duration=2, frame_rate=12, show=False, save_path=p)
    # pixel-identical frames get merged by Pillow: the buggy build produced
    # a 1-frame "animation" here
    assert Image.open(p).n_frames == 24


def test_continuous_hue_chemtrails_trail_windows():
    n = 40
    r = hyp.plot(_spiral(n), animate=True, chemtrails=True, duration=2,
                 tail_duration=0.5, frame_rate=12,
                 hue=np.linspace(0, 1, n), show=False)
    ax = r.figure.axes[0]
    r.animation._draw_frame(20)
    seg_counts = [len(getattr(co, '_segments3d', []))
                  for co in ax.collections]
    # nothing may hold the full 23-segment trajectory; head window is
    # 6 frames (0.5 s * 12 fps) -> 6 segments, trail = 20 - 6 = 14 pts
    assert max(seg_counts) < 23


# ---------------------------------------------------------------------------
# F04-003 / F05-012: multi-dataset animations must cover EVERY dataset in
# full (no first-dataset-driven truncation; 1-point datasets cannot crash)
# ---------------------------------------------------------------------------

def test_unequal_datasets_all_fully_animated():
    short = _walk(100, 8, seed=3)
    long = _walk(300, 8, seed=2)
    for order in ([short, long], [long, short]):
        r = hyp.plot(order, animate=True, duration=4, frame_rate=10,
                     show=False)
        interp = r.animation._args[0]
        assert [d.shape[0] for d in interp] == [40, 40]
        assert r.animation._save_count == 40
        # at the final frame every head line reaches its dataset's end
        r.animation._draw_frame(39)
        ax = r.figure.axes[0]
        for line, data in zip(ax.lines[:2], interp):
            xs, ys, zs = line.get_data_3d()
            np.testing.assert_allclose(
                [xs[-1], ys[-1], zs[-1]], data[-1])
        plt.close('all')


def test_single_point_dataset_does_not_crash_or_truncate():
    spiral = _spiral()
    pt = spiral[:1]
    # [trajectory, point] used to crash inside scipy pchip
    r1 = hyp.plot([spiral, pt], animate=True, duration=2, frame_rate=12,
                  show=False)
    assert r1.animation._save_count == 24
    # [point, trajectory] used to yield a silent 1-frame animation
    r2 = hyp.plot([pt, spiral], animate=True, duration=2, frame_rate=12,
                  show=False)
    assert r2.animation._save_count == 24
    # the single point stays visible at the end (freeze, never vanishes)
    r2.animation._draw_frame(23)
    ax = r2.figure.axes[0]
    assert len(ax.lines[0].get_data_3d()[0]) == 1


# ---------------------------------------------------------------------------
# F04-004: exactly frame_rate * duration frames, as documented
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n, frame_rate, duration', [
    (300, 30, 30),  # historical 901
    (10, 10, 4),    # historical 41
    (50, 10, 4),    # historical 41
])
def test_exact_frame_count(n, frame_rate, duration):
    r = hyp.plot(_walk(n, 10, seed=1), animate=True, duration=duration,
                 frame_rate=frame_rate, show=False)
    assert r.animation._save_count == frame_rate * duration


# ---------------------------------------------------------------------------
# F04-005 / F05-010: marker-only animations honor duration/frame_rate
# ---------------------------------------------------------------------------

def test_marker_only_honors_duration():
    r = hyp.plot([_walk(40, 3)], fmt='.', animate=True, duration=2,
                 frame_rate=12, show=False)
    assert r.animation._save_count == 24


# ---------------------------------------------------------------------------
# F04-006 / F05-004: invalid animate scalars raise instead of silent static
# ---------------------------------------------------------------------------

def test_animate_2_raises_value_error():
    with pytest.raises(ValueError, match='animate=2'):
        hyp.plot(_walk(30, 3), animate=2, duration=1, frame_rate=6,
                 show=False)
    with pytest.raises(ValueError, match='not a recognized animate value'):
        hyp.plot(_walk(30, 3), animate=2, duration=1, frame_rate=6,
                 save_path='/tmp/never_written.gif', show=False)


def test_animate_boolish_scalars_still_animate():
    from hypertools.plot.hyper_animation import HyperAnimation
    for val in (1, np.True_):
        r = hyp.plot(_walk(30, 3), animate=val, duration=1, frame_rate=6,
                     show=False)
        assert isinstance(r, HyperAnimation)
        plt.close('all')
    assert isinstance(
        hyp.plot(_walk(30, 3), animate=np.False_, show=False), plt.Figure)


# ---------------------------------------------------------------------------
# F04-007: duration/frame_rate=None -> clear ValueError, not TypeError
# ---------------------------------------------------------------------------

def test_duration_none_raises_clear_error():
    with pytest.raises(ValueError, match='duration must be a positive'):
        hyp.plot(_walk(30, 3), animate=True, duration=None, show=False)
    with pytest.raises(ValueError, match='frame_rate must be a positive'):
        hyp.plot(_walk(30, 3), animate=True, frame_rate=None, show=False)


# ---------------------------------------------------------------------------
# F05-009: negative tail_duration rejected
# ---------------------------------------------------------------------------

def test_negative_tail_duration_raises():
    with pytest.raises(ValueError, match='tail_duration must be a '
                                         'non-negative'):
        hyp.plot([_spiral()], animate=True, chemtrails=True,
                 tail_duration=-1, duration=2, frame_rate=12, show=False)


# ---------------------------------------------------------------------------
# F05-005: trail flags validate types (ndarray works; str/dict raise)
# ---------------------------------------------------------------------------

def test_trail_flag_ndarray_matches_list():
    data = [_walk(30, 3, seed=0), _walk(30, 3, seed=1)]
    r_arr = hyp.plot(data, animate=True,
                     chemtrails=np.array([True, False]), duration=1,
                     frame_rate=6, show=False)
    r_list = hyp.plot(data, animate=True, chemtrails=[True, False],
                      duration=1, frame_rate=6, show=False)
    assert (len(r_arr.figure.axes[0].lines)
            == len(r_list.figure.axes[0].lines) == 3)  # 2 heads + 1 trail


@pytest.mark.parametrize('bad', ['yes', {'0': True}])
def test_trail_flag_non_bool_scalar_raises(bad):
    with pytest.raises(TypeError, match='chemtrails must be a bool'):
        hyp.plot([_walk(30, 3)], animate=True, chemtrails=bad, duration=1,
                 frame_rate=6, show=False)


# ---------------------------------------------------------------------------
# F05-007: trail flags on a STATIC plot warn (previously silent no-op)
# ---------------------------------------------------------------------------

def test_static_trail_flags_warn():
    with pytest.warns(UserWarning, match='only affect ANIMATED plots'):
        hyp.plot([_walk(30, 3)], chemtrails=True, show=False)


# ---------------------------------------------------------------------------
# F05-011: all-NaN rows -> hypertools-level error, not raw scipy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('animate', [False, 'bullettime'])
def test_full_row_nan_gets_hypertools_error(animate):
    d = _spiral()
    d[10:13, :] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # PPCA "cannot fill" warning
        with pytest.raises(ValueError, match='ALL features missing'):
            hyp.plot([d], animate=animate, duration=2, frame_rate=12,
                     show=False)


# ---------------------------------------------------------------------------
# F05-003: azim= honored as the starting camera angle for spin/parallel/
# window (previously always 0; serial/morph already honored it)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('style', ['spin', True, 'window'])
def test_azim_honored_as_start_angle(style):
    r = hyp.plot(_walk(30, 3), animate=style, azim=45, duration=1,
                 frame_rate=6, show=False)
    r.animation._draw_frame(0)
    assert r.figure.axes[0].azim == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# F04-012: 1-D + animate raises ValueError (not a bare assert)
# ---------------------------------------------------------------------------

def test_1d_animate_raises_value_error():
    with pytest.raises(ValueError,
                       match='only supported for 2-D or 3-D'):
        hyp.plot(_walk(60, 5), ndims=1, animate=True, show=False)


# ---------------------------------------------------------------------------
# F04-011 / F05-013: unknown save extension -> clear ValueError naming the
# supported formats, never a leaked ffmpeg command line
# ---------------------------------------------------------------------------

def test_unknown_animation_extension_raises_clear_error(tmp_path):
    r = hyp.plot(_walk(30, 3), animate='spin', duration=1, frame_rate=6,
                 show=False)
    with pytest.raises(ValueError) as excinfo:
        r.save(str(tmp_path / 'anim.xyz'))
    msg = str(excinfo.value)
    assert '.gif' in msg and '.mp4' in msg and '.svg' in msg
    assert 'rawvideo' not in msg and 'pipe:' not in msg
    # a path with NO extension must raise too (it used to fall through to
    # ffmpeg because rsplit('.') returned the whole basename)
    with pytest.raises(ValueError, match='missing extension'):
        r.save(str(tmp_path / 'anim'))
    # save_path= goes through the same writer dispatch
    with pytest.raises(ValueError, match='unsupported animation save'):
        hyp.plot(_walk(30, 3), animate='spin', duration=1, frame_rate=6,
                 show=False, save_path=str(tmp_path / 'x.html'))


# ---------------------------------------------------------------------------
# F04-010: exported gif timing within ~2% of the requested duration
# ---------------------------------------------------------------------------

def test_gif_wall_clock_matches_requested_duration(tmp_path):
    from PIL import Image, ImageSequence
    p = str(tmp_path / 'd1fr30.gif')
    hyp.plot(_walk(300, 10), animate='window', duration=1, frame_rate=30,
             focused=0.2, show=False, save_path=p)
    im = Image.open(p)
    durs = [f.info.get('duration') for f in ImageSequence.Iterator(im)]
    assert len(durs) == 30
    total_s = sum(durs) / 1000.0
    # the historical single 30 ms delay played 0.900 s (10% fast)
    assert total_s == pytest.approx(1.0, rel=0.02)


# ---------------------------------------------------------------------------
# F04-002: legend + trail flags must not run the canvas-width fit away to
# the 3x cap (plot squashed into the left third)
# ---------------------------------------------------------------------------

def test_legend_with_trail_flags_keeps_canvas_width():
    data = [_walk(150, 8, seed=2), _walk(150, 8, seed=3)]
    widths = {}
    for flags in ({}, {'chemtrails': True}, {'precog': True},
                  {'bullettime': True}):
        r = hyp.plot(data, animate=True, duration=2, frame_rate=5,
                     legend=['a', 'b'], show=False, **flags)
        widths[tuple(flags)] = r.figure.get_size_inches()[0]
        plt.close('all')
    base = widths[()]
    for key, w in widths.items():
        # every trail variant must match the no-trail fit (not the 3x cap)
        assert w == pytest.approx(base, abs=0.5), (key, w, base)
