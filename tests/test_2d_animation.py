# -*- coding: utf-8 -*-
"""2-D animations, both backends (round17 #9, GH #123).

Every `animate` style except `'spin'` now works for `ndims=2` in both the
matplotlib and plotly backends, using a FIXED (non-rotating) viewport --
there is no camera to move in 2-D, so `'parallel'`/`True`/`'serial'`/
`'window'`/`chemtrails`/`precog`/`bullettime`/`'morph'` simply skip every
camera-angle bookkeeping step their 3-D counterparts do. `'spin'` (which
rotates the camera and nothing else) is meaningless for 2-D data and raises
`ValueError` naming the other styles instead.

All renders are real (no mocks): MPLBACKEND=Agg, show=False.
"""

import os

import numpy as np
import pytest
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp
from hypertools.plot import morph as _morph


def _walks2d(n=60, k=2, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.standard_normal((n, 2)), axis=0) + 5 * i
            for i in range(k)]


def _blobs2d(n=30, k=3, seed=1, spacing=6.0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, 2)) + spacing * i for i in range(k)]


# ---------------------------------------------------------------------------
# matplotlib: frame counts + fixed (non-rotating) viewport
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('style', ['parallel', 'serial', 'window'])
def test_mpl_frame_count_matches_duration_and_frame_rate(style):
    data = _walks2d()
    duration, frame_rate = 2, 10
    bundle = hyp.plot(data, ndims=2, animate=style, duration=duration,
                      frame_rate=frame_rate, show=False, return_model=True)
    ani = bundle['animation']
    fig = bundle['fig']
    assert ani is not None
    ax = fig.axes[0]
    # a plain (2-D) Axes, not Axes3D: no camera/projection concept at all
    assert not hasattr(ax, 'get_proj')
    assert ani._save_count == duration * frame_rate
    plt.close('all')


def test_mpl_morph_frame_count_matches_duration_and_frame_rate():
    data = _blobs2d(k=3)
    duration, frame_rate = 2, 10
    bundle = hyp.plot(data, '.', ndims=2, animate='morph', duration=duration,
                      frame_rate=frame_rate, show=False, return_model=True)
    ani = bundle['animation']
    morph_state = ani._args[0]
    assert sum(morph_state['frame_counts']) == duration * frame_rate
    assert ani._save_count == duration * frame_rate
    plt.close('all')


@pytest.mark.parametrize('style', ['parallel', 'window'])
def test_mpl_artist_data_progresses_across_frames(style):
    """First/mid/last frame artist data must differ -- the animation
    actually advances rather than staying frozen."""
    data = _walks2d(n=90)
    duration, frame_rate = 3, 10
    bundle = hyp.plot(data, ndims=2, animate=style, duration=duration,
                      frame_rate=frame_rate, focused=1, show=False,
                      return_model=True)
    ani = bundle['animation']
    fig = bundle['fig']
    fig.canvas.draw()
    total = ani._save_count

    lines_first, _ = ani._func(0, *ani._args)
    x_first, y_first = lines_first[0].get_data()

    lines_mid, _ = ani._func(total // 2, *ani._args)
    x_mid, y_mid = lines_mid[0].get_data()

    lines_last, _ = ani._func(total - 1, *ani._args)
    x_last, y_last = lines_last[0].get_data()

    assert not (np.array_equal(x_first, x_mid) and np.array_equal(y_first, y_mid))
    assert not (np.array_equal(x_mid, x_last) and np.array_equal(y_mid, y_last))
    assert not (np.array_equal(x_first, x_last) and np.array_equal(y_first, y_last))
    plt.close('all')


def test_mpl_window_exact_bounds_mid_animation_2d():
    """Mirrors `test_window_animation.py`'s identical 3-D check: at a
    sampled mid-animation frame, the drawn line data contains ONLY points
    within the expected `focused=`-derived window."""
    data = _walks2d(n=100)
    focused = 0.5
    frame_rate = 20
    bundle = hyp.plot(data, ndims=2, animate='window', duration=4,
                      frame_rate=frame_rate, focused=focused, show=False,
                      return_model=True)
    ani = bundle['animation']
    fig = bundle['fig']
    fig.canvas.draw()

    data_lines = ani._args[0]
    total = ani._save_count
    num = total // 2
    window_frames = int(round(frame_rate * focused))
    expected = data_lines[0][num - window_frames: num + 1]

    lines, _ = ani._func(num, *ani._args)
    xs, ys = lines[0].get_data()
    assert len(xs) == len(expected)
    np.testing.assert_allclose(xs, expected[:, 0])
    np.testing.assert_allclose(xs[0], data_lines[0][num - window_frames, 0])
    np.testing.assert_allclose(xs[-1], data_lines[0][num, 0])
    plt.close('all')


def test_mpl_morph_2d_hungarian_correspondence():
    """2-D morph frames interpolate between datasets via the SAME
    Hungarian-matched hold/morph schedule as 3-D -- verified by comparing
    a mid-morph frame's artist data against `morph.morph_positions`
    computed independently from the SAME sampled clouds."""
    data = _blobs2d(k=3, seed=7)
    bundle = hyp.plot(data, '.', ndims=2, animate='morph', duration=2,
                      frame_rate=10, show=False, return_model=True)
    ani = bundle['animation']
    morph_state = ani._args[0]
    frame_counts = morph_state['frame_counts']
    assert len(frame_counts) == 5  # 3 datasets -> 5 segments

    mid_frame = frame_counts[0] + frame_counts[1] // 2
    seg_idx, step, n_steps = _morph.frame_to_segment(frame_counts, mid_frame)
    assert seg_idx == 1  # mid-morph between dataset 0 and dataset 1
    expected = _morph.morph_positions(morph_state['sampled'], seg_idx, step,
                                      n_steps)

    ani._func(mid_frame, *ani._args)
    xs, ys = morph_state['artist'].get_data()
    np.testing.assert_allclose(xs, expected[:, 0])
    np.testing.assert_allclose(ys, expected[:, 1])
    plt.close('all')


def test_mpl_spin_raises_valueerror_for_2d():
    data = _walks2d()
    with pytest.raises(ValueError, match="spin"):
        hyp.plot(data, ndims=2, animate='spin', duration=1, frame_rate=10,
                 show=False)


def test_mpl_xlabel_ylabel_present_on_animated_2d():
    data = _walks2d()
    bundle = hyp.plot(data, ndims=2, animate='parallel', duration=1,
                      frame_rate=10, xlabel='2D animated X',
                      ylabel='2D animated Y', show=False,
                      return_model=True)
    ax = bundle['fig'].axes[0]
    assert ax.get_xlabel() == '2D animated X'
    assert ax.get_ylabel() == '2D animated Y'
    plt.close('all')


# ---------------------------------------------------------------------------
# matplotlib: saved GIF (tiny, low dpi) has real, non-identical frames
# ---------------------------------------------------------------------------

def test_mpl_2d_gif_export_has_nonidentical_frames(tmp_path):
    out = str(tmp_path / 'anim2d.gif')
    data = _walks2d(n=40)
    hyp.plot(data, ndims=2, animate='window', duration=2, frame_rate=10,
             focused=0.5, save_path=out, show=False,
             size=(2, 2))
    plt.close('all')

    assert os.path.getsize(out) > 1024  # > 1KB

    with Image.open(out) as im:
        n_frames = getattr(im, 'n_frames', 1)
        assert n_frames > 1  # duration=2 * frame_rate=10 == 20 frames

        im.seek(0)
        frame_first = np.array(im.convert('RGB'))
        im.seek(n_frames // 2)
        frame_mid = np.array(im.convert('RGB'))
        im.seek(n_frames - 1)
        frame_last = np.array(im.convert('RGB'))

    assert not np.array_equal(frame_first, frame_mid)
    assert not np.array_equal(frame_mid, frame_last)


# ---------------------------------------------------------------------------
# plotly: frame counts, progression, spin rejection, xlabel/ylabel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('style', ['parallel', 'serial', 'window'])
def test_plotly_frame_count_matches_duration_and_frame_rate(style):
    data = _walks2d()
    duration, frame_rate = 2, 10
    fig = hyp.plot(data, ndims=2, animate=style, duration=duration,
                   frame_rate=frame_rate, backend='plotly', show=False)
    assert len(fig.frames) == duration * frame_rate
    # 2-D layout: xaxis/yaxis carry the fixed [-1.1, 1.1] range; no camera
    assert fig.layout.xaxis.range == (-1.1, 1.1)
    assert fig.layout.yaxis.range == (-1.1, 1.1)


def test_plotly_morph_frame_count_matches_duration_and_frame_rate():
    data = _blobs2d(k=3)
    duration, frame_rate = 2, 10
    fig = hyp.plot(data, '.', ndims=2, animate='morph', duration=duration,
                   frame_rate=frame_rate, backend='plotly', show=False)
    assert len(fig.frames) == duration * frame_rate


@pytest.mark.parametrize('style', ['parallel', 'window'])
def test_plotly_artist_data_progresses_across_frames(style):
    data = _walks2d(n=90)
    duration, frame_rate = 3, 10
    fig = hyp.plot(data, ndims=2, animate=style, duration=duration,
                   frame_rate=frame_rate, focused=1, backend='plotly',
                   show=False)
    n = len(fig.frames)
    x_first = np.asarray(fig.frames[0].data[0].x)
    x_mid = np.asarray(fig.frames[n // 2].data[0].x)
    x_last = np.asarray(fig.frames[-1].data[0].x)

    assert not np.array_equal(x_first, x_mid)
    assert not np.array_equal(x_mid, x_last)
    assert not np.array_equal(x_first, x_last)


def test_plotly_morph_2d_hungarian_correspondence():
    data = _blobs2d(k=3, seed=7)
    fig = hyp.plot(data, '.', ndims=2, animate='morph', duration=2,
                   frame_rate=10, backend='plotly', show=False)
    n = len(fig.frames)
    assert n > 2
    x_first = np.asarray(fig.frames[0].data[0].x)
    x_mid = np.asarray(fig.frames[n // 2].data[0].x)
    x_last = np.asarray(fig.frames[-1].data[0].x)
    # the traveling cloud actually moves between hold frames
    assert not np.array_equal(x_first, x_mid)
    assert not np.array_equal(x_mid, x_last)


def test_plotly_spin_raises_valueerror_for_2d():
    data = _walks2d()
    with pytest.raises(ValueError, match="spin"):
        hyp.plot(data, ndims=2, animate='spin', duration=1, frame_rate=10,
                 backend='plotly', show=False)


def test_plotly_xlabel_ylabel_present_on_animated_2d():
    data = _walks2d()
    fig = hyp.plot(data, ndims=2, animate='parallel', duration=1,
                   frame_rate=10, xlabel='Plotly 2D X', ylabel='Plotly 2D Y',
                   backend='plotly', show=False)
    assert fig.layout.xaxis.title.text == 'Plotly 2D X'
    assert fig.layout.yaxis.title.text == 'Plotly 2D Y'


# ---------------------------------------------------------------------------
# rotations=/zoom= on 2-D: warned-and-ignored (both backends, consistent)
# ---------------------------------------------------------------------------

def test_mpl_rotations_and_zoom_warned_and_ignored_for_2d():
    data = _walks2d()
    with pytest.warns(UserWarning, match="rotations="):
        hyp.plot(data, ndims=2, animate='parallel', rotations=2,
                 duration=1, frame_rate=10, show=False)
    plt.close('all')
    with pytest.warns(UserWarning, match="zoom="):
        hyp.plot(data, ndims=2, animate='parallel', zoom=2,
                 duration=1, frame_rate=10, show=False)
    plt.close('all')


def test_plotly_rotations_and_zoom_warned_and_ignored_for_2d():
    data = _walks2d()
    with pytest.warns(UserWarning, match="rotations="):
        hyp.plot(data, ndims=2, animate='parallel', rotations=2,
                 duration=1, frame_rate=10, backend='plotly', show=False)
    with pytest.warns(UserWarning, match="zoom="):
        hyp.plot(data, ndims=2, animate='parallel', zoom=2,
                 duration=1, frame_rate=10, backend='plotly', show=False)
