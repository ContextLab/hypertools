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


def _three_walks(n=120, d=6):
    rng = np.random.default_rng(0)
    return [np.cumsum(rng.standard_normal((n, d)), axis=0) + i * 4.0
           for i in range(3)]


def _mid_frame(fig):
    return fig.frames[len(fig.frames) // 2]


def test_plotly_chemtrails_past_trail():
    fig = hyp.plot(_walks(), animate=True, duration=4, tail_duration=1,
                   chemtrails=True, backend='plotly', show=False)
    # 2 data + 2 trail + cube
    assert len(fig.data) == 5
    f = _mid_frame(fig)
    assert len(f.data) == 4
    head, trail = f.data[0], f.data[2]
    # past trail: starts at the beginning, ends where the window starts --
    # sharing that one vertex, so the faded trail and the opaque head join
    # with no visible gap. This used to assert `len(trail) > len(head)`,
    # which held only by a coincidental margin of one point: the head window
    # was a point short of matplotlib's at the time, and the "trail is
    # longer" inequality was what that missing point bought. The shared
    # vertex is the actual contract (`trails.anim_window_bounds`:
    # trail_stop == start + 1), and it cannot be satisfied by an off-by-one.
    assert trail.x[0] == fig.data[0].x[0]
    assert trail.x[-1] == head.x[0]
    assert trail.y[-1] == head.y[0]
    assert trail.z[-1] == head.z[0]


def test_plotly_precog_future_trail():
    fig = hyp.plot(_walks(), animate=True, duration=4, tail_duration=1,
                   precog=True, backend='plotly', show=False)
    f = _mid_frame(fig)
    trail = f.data[2]
    # future trail: ends at the trajectory's end
    assert trail.x[-1] == fig.data[0].x[-1]


def test_plotly_bullettime_full_trail():
    fig = hyp.plot(_walks(), animate=True, duration=4, tail_duration=1,
                   bullettime=True, backend='plotly', show=False)
    f = _mid_frame(fig)
    n_total = len(fig.data[0].x)
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

    def r(fig):
        eye = fig.layout.scene.camera.eye
        return float(np.sqrt(eye.x ** 2 + eye.y ** 2 + eye.z ** 2))

    assert r(near) < r(far)


def test_plotly_animation_zooms_out_vs_static():
    """Animated plots pull the camera slightly farther back than the static
    view so the wireframe box keeps a margin at every rotation angle (never
    clipped); static plots are visually unchanged."""
    from hypertools.plot.plotly_backend import _zoom_r, _anim_zoom_r

    # animation radius is strictly larger (camera farther => box smaller)
    for z in (0.5, 1, 2, 3):
        assert _anim_zoom_r(z) > _zoom_r(z)

    def r(eye):
        return float(np.sqrt(eye.x ** 2 + eye.y ** 2 + eye.z ** 2))

    # static plot keeps the un-zoomed-out radius
    stat = hyp.plot(_walks(), backend='plotly', show=False)
    assert r(stat.layout.scene.camera.eye) == pytest.approx(_zoom_r(1))

    # spin animation: initial camera AND every frame use the zoomed-out
    # radius (initial == frame 0 so playback does not jump)
    spin = hyp.plot(_walks(), animate='spin', duration=2, backend='plotly',
                    show=False)
    assert r(spin.layout.scene.camera.eye) == pytest.approx(_anim_zoom_r(1))
    mid = spin.frames[len(spin.frames) // 2]
    # frames store the camera as scene.camera (the scene_camera= setter in
    # _add_animation expands to scene.camera)
    assert r(mid.layout.scene.camera.eye) == pytest.approx(_anim_zoom_r(1))


def test_plotly_trail_alpha_honors_per_dataset_alpha():
    """Important finding 3 (whole-branch review): plotly trail traces
    hardcoded `_to_plotly_color(color, 0.3)`, dropping alpha= entirely,
    while matplotlib folds the 0.3 trail-fade factor into whatever alpha
    the dataset carries (`0.3 * kw.pop('alpha', 1.0)`). Repro:
    alpha=[1.0, 0.5, 0.2], chemtrails=True -> matplotlib trails are
    0.3/0.15/0.06 (heads stay 1.0/0.5/0.2); plotly used to give 0.3/0.3/0.3
    for every trail regardless of alpha=."""
    walks = _three_walks()
    alphas_in = [1.0, 0.5, 0.2]
    expected = [0.3 * a for a in alphas_in]

    fig, ani = hyp.plot(walks, animate=True, duration=2, tail_duration=1,
                        chemtrails=True, alpha=alphas_in, show=False)
    trail = ani._args[2]
    assert all(t is not None for t in trail), 'expected a trail per dataset'
    mpl_trail_alphas = [t.get_alpha() for t in trail]
    assert mpl_trail_alphas == pytest.approx(expected)

    pfig = hyp.plot(walks, animate=True, duration=2, tail_duration=1,
                    chemtrails=True, alpha=alphas_in, backend='plotly',
                    show=False)
    n = len(walks)
    trail_traces = pfig.data[n:2 * n]
    assert len(trail_traces) == n
    ply_trail_alphas = [
        float(t.line.color.rsplit(',', 1)[1].rstrip(') '))
        for t in trail_traces]
    assert ply_trail_alphas == pytest.approx(expected), (
        "plotly trail traces must honor per-dataset alpha= (0.3 * alpha), "
        "matching matplotlib, not a hardcoded 0.3")
    assert ply_trail_alphas == pytest.approx(mpl_trail_alphas)


def test_plotly_static_has_no_trails():
    fig = hyp.plot(_walks(), backend='plotly', show=False)
    assert len(fig.data) == 3  # 2 data + cube, no trail traces


def test_backends_have_identical_animation_pacing():
    """Round-6.5 standard: 30 fps, duration 30 s, 1 rotation per 30 s --
    and the two backends generate exactly the same number of frames at the
    same per-frame duration."""
    w = np.cumsum(np.random.default_rng(3).standard_normal((60, 5)), axis=0)

    gp = hyp.plot(w, animate='spin', backend='plotly', show=False)
    n_plotly = len(gp.frames)
    ms_plotly = gp.layout.updatemenus[0].buttons[0].args[1][
        'frame']['duration']

    gm_fig, gm_ani = hyp.plot(w, animate='spin', show=False)
    n_mpl = gm_ani._save_count
    ms_mpl = gm_ani._interval

    assert n_plotly == n_mpl == 900          # 30 fps * 30 s
    assert abs(ms_plotly - ms_mpl) <= 1.0    # ~33 ms per frame
    import matplotlib.pyplot as plt
    plt.close('all')
