# -*- coding: utf-8 -*-
"""animate='window' + focused= + duration semantics (round17 #8, GH #275).

`animate='window'`: a sliding FULLY-OPAQUE window of the trajectory --
nothing outside the window is drawn at all (unlike `bullettime`, which also
paints a low-opacity trail outside its own opaque window). Jeremy's own
definition: "like bullettime, but without the precog and chemtrail parts."

`focused=`: the length (seconds, same unit as `tail_duration`) of that
opaque "in-focus" window for `animate='window'`/`chemtrails`/`precog`/
`bullettime`. `None` (default) resolves to `tail_duration`'s own value, so
omitting `focused=` never changes any pre-existing behavior. Silently
ignored for `animate='spin'`/`'parallel'` (with no trail flag set on any
dataset)/`'morph'`.

`duration=`: verified here to control wall-clock animation length (frame
count = duration * frame_rate) for every style, including the new 'window'
and 'morph'.

All renders are real (no mocks): MPLBACKEND=Agg, show=False.
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp
from hypertools.plot.matplotlib_backend import _draw
from hypertools.plot.plotly_backend import plotly_draw


def _walks(n=200, d=3, k=2, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.standard_normal((n, d)), axis=0) + 5 * i
            for i in range(k)]


# ---------------------------------------------------------------------------
# matplotlib: animate='window' draws ONLY the current window, nothing else
# ---------------------------------------------------------------------------

def test_mpl_window_draws_only_current_window():
    data = _walks()
    bundle = hyp.plot(data, animate='window', duration=4, frame_rate=30,
                      focused=1, show=False, return_model=True)
    ani = bundle['animation']
    fig = bundle['fig']
    assert ani is not None

    fig.canvas.draw()
    total = ani._save_count
    full_len = ani._args[0][0].shape[0]
    frame_rate = 30
    focused_frames = int(round(frame_rate * 1))  # focused=1s @ 30fps -> 30

    for num in (total // 4, total // 2, 3 * total // 4):
        lines, trail_lines = ani._func(num, *ani._args)
        # no trail artist at all for 'window' (nothing faded is drawn)
        assert all(t is None for t in trail_lines)
        for line in lines:
            xs, ys, zs = line.get_data_3d()
            # the drawn window is AT MOST focused_frames + 1 points long
            assert len(xs) <= focused_frames + 1
            # the full trajectory is NEVER fully drawn mid-animation
            assert len(xs) < full_len

    import matplotlib.pyplot as plt
    plt.close('all')


def test_mpl_window_exact_bounds_mid_animation():
    """At a sampled mid-animation frame, the drawn line data contains ONLY
    points within the expected window (computed from focused= and the frame
    index) -- points before/after the window are absent."""
    data = _walks()
    focused = 0.5
    frame_rate = 20
    bundle = hyp.plot(data, animate='window', duration=4,
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
    xs, ys, zs = lines[0].get_data_3d()
    assert len(xs) == len(expected)
    np.testing.assert_allclose(xs, expected[:, 0])
    np.testing.assert_allclose(xs[0], data_lines[0][num - window_frames, 0])
    np.testing.assert_allclose(xs[-1], data_lines[0][num, 0])

    plt.close('all')


def test_mpl_window_never_shows_full_trajectory():
    data = _walks(n=90)
    bundle = hyp.plot(data, animate='window', duration=3, frame_rate=30,
                      focused=1, show=False, return_model=True)
    ani = bundle['animation']
    fig = bundle['fig']
    fig.canvas.draw()
    full_len = ani._args[0][0].shape[0]
    total = ani._save_count
    for num in range(0, total, max(1, total // 15)):
        lines, _ = ani._func(num, *ani._args)
        for line in lines:
            xs, ys, zs = line.get_data_3d()
            assert len(xs) < full_len
    plt.close('all')


def test_mpl_window_ignores_trail_flags_with_warning():
    data = _walks(n=60)
    with pytest.warns(UserWarning, match=r"animate='window'.*chemtrails"):
        bundle = hyp.plot(data, animate='window', duration=1,
                          tail_duration=1, frame_rate=10,
                          chemtrails=[True, False], show=False,
                          return_model=True)
    fig = bundle['fig']
    ax = fig.axes[0]
    # exactly one Line3D per dataset -- no trail artist ever created
    assert len(ax.lines) == 2
    plt.close('all')


def test_mpl_window_works_with_rotations():
    data = _walks(n=60)
    bundle = hyp.plot(data, animate='window', duration=1, frame_rate=10,
                      rotations=2, focused=0.5, show=False,
                      return_model=True)
    ani = bundle['animation']
    fig = bundle['fig']
    fig.canvas.draw()
    ax = fig.axes[0]
    ani._func(2, *ani._args)
    azim2 = ax.azim
    ani._func(ani._save_count - 2, *ani._args)
    azim_end = ax.azim
    # the camera actually rotated over the course of the animation
    assert azim2 != azim_end
    plt.close('all')


# ---------------------------------------------------------------------------
# plotly: animate='window' draws ONLY the current window, nothing else
# ---------------------------------------------------------------------------

def _mid_frame(fig):
    return fig.frames[len(fig.frames) // 2]


def test_plotly_window_draws_only_current_window():
    pytest.importorskip('plotly')
    data = _walks()
    fig = hyp.plot(data, animate='window', duration=4, frame_rate=30,
                   focused=1, backend='plotly', show=False)
    full_len = len(fig.data[0].x)
    assert len(fig.frames) == 120
    for k in (30, 60, 90):
        f = fig.frames[k]
        # no trail traces at all for 'window'
        assert len(f.data) == len(fig.data) - 1  # data traces only, no cube
        for trace in f.data:
            assert len(trace.x) < full_len


def test_plotly_window_exact_bounds_mid_animation():
    pytest.importorskip('plotly')
    data = _walks()
    focused = 0.5
    frame_rate = 20
    fig = hyp.plot(data, animate='window', duration=4, frame_rate=frame_rate,
                   focused=focused, backend='plotly', show=False)
    max_len = max(len(d) for d in [np.asarray(t) for t in
                                    [fig.data[0].x, fig.data[1].x]])
    n_frames = len(fig.frames)
    mid_k = n_frames // 2
    window = max(2, int(round(max_len * focused / 4.0)))
    end = max(2, int(np.ceil((mid_k + 1) * max_len / n_frames)))
    start = max(0, end - window)
    f = fig.frames[mid_k]
    assert len(f.data[0].x) == min(end, max_len) - start


def test_plotly_window_ignores_trail_flags():
    pytest.importorskip('plotly')
    data = _walks(n=60)
    # chemtrails with animate='window' deliberately provokes the
    # trail-styles-ignored notice
    with pytest.warns(UserWarning, match='does not support trail styles'):
        fig = hyp.plot(data, animate='window', duration=1, tail_duration=1,
                       chemtrails=True, backend='plotly', show=False)
    # 2 data + cube, no trail traces ever created for 'window'
    assert len(fig.data) == 3


# ---------------------------------------------------------------------------
# focused=: monotonic effect on the in-focus window size
# ---------------------------------------------------------------------------

def test_focused_larger_gives_more_points_mpl_window():
    data = _walks()
    small = hyp.plot(data, animate='window', duration=4, frame_rate=30,
                     focused=0.5, show=False, return_model=True)['animation']
    large = hyp.plot(data, animate='window', duration=4, frame_rate=30,
                     focused=2.0, show=False, return_model=True)['animation']
    num = small._save_count // 2
    l_small, _ = small._func(num, *small._args)
    l_large, _ = large._func(num, *large._args)
    n_small = len(l_small[0].get_data_3d()[0])
    n_large = len(l_large[0].get_data_3d()[0])
    assert n_large > n_small
    plt.close('all')


def test_focused_larger_gives_more_points_mpl_bullettime():
    data = _walks()
    small = hyp.plot(data, animate=True, duration=4, frame_rate=30,
                     tail_duration=1, focused=0.5, bullettime=True,
                     show=False, return_model=True)['animation']
    large = hyp.plot(data, animate=True, duration=4, frame_rate=30,
                     tail_duration=1, focused=2.0, bullettime=True,
                     show=False, return_model=True)['animation']
    num = small._save_count // 2
    l_small, t_small = small._func(num, *small._args)
    l_large, t_large = large._func(num, *large._args)
    # the OPAQUE head window (not the full bullettime trail) grows with
    # focused
    n_small = len(l_small[0].get_data_3d()[0])
    n_large = len(l_large[0].get_data_3d()[0])
    assert n_large > n_small
    plt.close('all')


def test_focused_larger_gives_more_points_plotly_chemtrails():
    pytest.importorskip('plotly')
    data = _walks()
    small = hyp.plot(data, animate=True, duration=4, frame_rate=30,
                     tail_duration=1, focused=0.5, chemtrails=True,
                     backend='plotly', show=False)
    large = hyp.plot(data, animate=True, duration=4, frame_rate=30,
                     tail_duration=1, focused=2.0, chemtrails=True,
                     backend='plotly', show=False)
    n_small = len(_mid_frame(small).data[0].x)
    n_large = len(_mid_frame(large).data[0].x)
    assert n_large > n_small


def test_focused_ignored_for_spin():
    """focused ignored for spin: identical frames with focused=2 vs
    focused=8."""
    data = _walks(n=60)
    a1 = hyp.plot(data, animate='spin', duration=1, frame_rate=10,
                 focused=2, show=False, return_model=True)['animation']
    a2 = hyp.plot(data, animate='spin', duration=1, frame_rate=10,
                 focused=8, show=False, return_model=True)['animation']
    num = a1._save_count // 2
    out1 = a1._func(num, *a1._args)
    out2 = a2._func(num, *a2._args)
    lines1 = out1[0] if isinstance(out1, tuple) else out1
    lines2 = out2[0] if isinstance(out2, tuple) else out2
    for l1, l2 in zip(lines1, lines2):
        c1 = l1.get_data_3d()
        c2 = l2.get_data_3d()
        for a, b in zip(c1, c2):
            np.testing.assert_allclose(a, b)
    plt.close('all')


def test_focused_ignored_for_plain_parallel_no_trail_flags():
    """focused has no effect on plain animate=True/'parallel' when no
    dataset has a chemtrails/precog/bullettime flag set -- tail_duration
    alone governs, as documented."""
    data = _walks()
    baseline = hyp.plot(data, animate=True, duration=4, frame_rate=30,
                        tail_duration=1, show=False,
                        return_model=True)['animation']
    with_focused = hyp.plot(data, animate=True, duration=4, frame_rate=30,
                            tail_duration=1, focused=8, show=False,
                            return_model=True)['animation']
    num = baseline._save_count // 2
    l1, _ = baseline._func(num, *baseline._args)
    l2, _ = with_focused._func(num, *with_focused._args)
    n1 = len(l1[0].get_data_3d()[0])
    n2 = len(l2[0].get_data_3d()[0])
    assert n1 == n2
    plt.close('all')


# ---------------------------------------------------------------------------
# duration=: frame count assertions per style
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('mode,extra', [
    ('window', {'focused': 1}),
    ('spin', {}),
    (True, {'bullettime': True, 'tail_duration': 1}),
    ('morph', {}),
])
def test_mpl_duration_controls_frame_count(mode, extra):
    data = _walks(n=60)
    frame_rate = 15
    a5 = hyp.plot(data, animate=mode, duration=5, frame_rate=frame_rate,
                 show=False, return_model=True, **extra)['animation']
    a10 = hyp.plot(data, animate=mode, duration=10, frame_rate=frame_rate,
                  show=False, return_model=True, **extra)['animation']
    assert a5._save_count == 5 * frame_rate
    assert a10._save_count == 10 * frame_rate
    # interval timing: frames * interval ~= duration * 1000 ms
    assert a5._save_count * a5._interval == pytest.approx(5000, rel=0.02)
    assert a10._save_count * a10._interval == pytest.approx(10000, rel=0.02)
    plt.close('all')


@pytest.mark.parametrize('mode,extra', [
    ('window', {'focused': 1}),
    ('spin', {}),
    (True, {'bullettime': True, 'tail_duration': 1}),
    ('morph', {}),
])
def test_plotly_duration_controls_frame_count(mode, extra):
    pytest.importorskip('plotly')
    data = _walks(n=60)
    frame_rate = 15
    f5 = hyp.plot(data, animate=mode, duration=5, frame_rate=frame_rate,
                 backend='plotly', show=False, **extra)
    f10 = hyp.plot(data, animate=mode, duration=10, frame_rate=frame_rate,
                  backend='plotly', show=False, **extra)
    assert len(f5.frames) == 5 * frame_rate
    assert len(f10.frames) == 10 * frame_rate
    ms5 = f5.layout.updatemenus[0].buttons[0].args[1]['frame']['duration']
    ms10 = f10.layout.updatemenus[0].buttons[0].args[1]['frame']['duration']
    assert len(f5.frames) * ms5 == pytest.approx(5000, rel=0.05)
    assert len(f10.frames) * ms10 == pytest.approx(10000, rel=0.05)


def test_backends_agree_on_window_frame_count_and_pacing():
    data = _walks(n=60)
    pytest.importorskip('plotly')
    gp = hyp.plot(data, animate='window', duration=3, frame_rate=20,
                 focused=1, backend='plotly', show=False)
    gm_fig, gm_ani = hyp.plot(data, animate='window', duration=3,
                              frame_rate=20, focused=1, show=False)
    n_plotly = len(gp.frames)
    n_mpl = gm_ani._save_count
    assert n_plotly == n_mpl == 60
    plt.close('all')


# ---------------------------------------------------------------------------
# animate= dict form (GH #154 resolution) works with style='window'
# ---------------------------------------------------------------------------

def test_animate_dict_form_window_style():
    data = _walks(n=60)
    bundle = hyp.plot(
        data, animate={'style': 'window', 'duration': 10, 'focused': 4},
        frame_rate=10, show=False, return_model=True,
    )
    ani = bundle['animation']
    assert ani is not None
    assert ani._save_count == 100  # duration=10 * frame_rate=10
    plt.close('all')


def test_animate_dict_form_window_style_plotly():
    pytest.importorskip('plotly')
    data = _walks(n=60)
    fig = hyp.plot(
        data, animate={'style': 'window', 'duration': 10, 'focused': 4},
        frame_rate=10, backend='plotly', show=False,
    )
    assert len(fig.frames) == 100


def test_focused_invalid_value_raises():
    data = _walks(n=20)
    with pytest.raises(ValueError, match='focused'):
        hyp.plot(data, animate='window', focused=-1, show=False)
    with pytest.raises(ValueError, match='focused'):
        hyp.plot(data, animate='window', focused=True, show=False)
