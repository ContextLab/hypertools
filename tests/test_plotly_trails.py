# -*- coding: utf-8 -*-
"""Plotly backend parity for window-animation extras: chemtrails (past
trail), precog (future trail), bullettime (both), tail_duration (window
length), and zoom (camera distance) -- previously matplotlib-only."""

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp

pytest.importorskip('plotly')


def _walks(n=120, d=6):
    w = np.cumsum(np.random.default_rng(0).standard_normal((n, d)), axis=0)
    return [w, w + 4]


def _mid_frame(geo):
    return geo.fig.frames[len(geo.fig.frames) // 2]


def test_plotly_chemtrails_past_trail():
    geo = hyp.plot(_walks(), animate=True, duration=4, tail_duration=1,
                   chemtrails=True, backend='plotly', show=False)
    # 2 data + 2 trail + cube
    assert len(geo.fig.data) == 5
    f = _mid_frame(geo)
    assert len(f.data) == 4
    head, trail = f.data[0], f.data[2]
    # past trail: starts at the beginning, ends where the window starts
    assert trail.x[0] == geo.fig.data[0].x[0]
    assert len(trail.x) > len(head.x)


def test_plotly_precog_future_trail():
    geo = hyp.plot(_walks(), animate=True, duration=4, tail_duration=1,
                   precog=True, backend='plotly', show=False)
    f = _mid_frame(geo)
    trail = f.data[2]
    # future trail: ends at the trajectory's end
    assert trail.x[-1] == geo.fig.data[0].x[-1]


def test_plotly_bullettime_full_trail():
    geo = hyp.plot(_walks(), animate=True, duration=4, tail_duration=1,
                   bullettime=True, backend='plotly', show=False)
    f = _mid_frame(geo)
    n_total = len(geo.fig.data[0].x)
    assert len(f.data[2].x) == n_total


def test_plotly_tail_duration_sets_window():
    long_t = hyp.plot(_walks(), animate=True, duration=4, tail_duration=2,
                      backend='plotly', show=False)
    short_t = hyp.plot(_walks(), animate=True, duration=4, tail_duration=0.5,
                       backend='plotly', show=False)
    mid_long = len(_mid_frame(long_t).data[0].x)
    mid_short = len(_mid_frame(short_t).data[0].x)
    assert mid_long > 2 * mid_short


def test_plotly_zoom_moves_camera_closer():
    near = hyp.plot(_walks(), zoom=3, backend='plotly', show=False)
    far = hyp.plot(_walks(), zoom=1, backend='plotly', show=False)

    def r(geo):
        eye = geo.fig.layout.scene.camera.eye
        return float(np.sqrt(eye.x ** 2 + eye.y ** 2 + eye.z ** 2))

    assert r(near) < r(far)


def test_plotly_static_has_no_trails():
    geo = hyp.plot(_walks(), backend='plotly', show=False)
    assert len(geo.fig.data) == 3  # 2 data + cube, no trail traces


def test_backends_have_identical_animation_pacing():
    """Round-6.5 standard: 30 fps, duration 30 s, 1 rotation per 30 s --
    and the two backends generate exactly the same number of frames at the
    same per-frame duration."""
    w = np.cumsum(np.random.default_rng(3).standard_normal((60, 5)), axis=0)

    gp = hyp.plot(w, animate='spin', backend='plotly', show=False)
    n_plotly = len(gp.fig.frames)
    ms_plotly = gp.fig.layout.updatemenus[0].buttons[0].args[1][
        'frame']['duration']

    gm = hyp.plot(w, animate='spin', show=False)
    n_mpl = gm.line_ani._save_count
    ms_mpl = gm.line_ani._interval

    assert n_plotly == n_mpl == 900          # 30 fps * 30 s
    assert abs(ms_plotly - ms_mpl) <= 1.0    # ~33 ms per frame
    import matplotlib.pyplot as plt
    plt.close('all')
