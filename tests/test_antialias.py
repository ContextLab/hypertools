# -*- coding: utf-8 -*-
"""Tests for `hyp.plot(antialias=...)` -- automatic line smoothing.

`antialias=True` (the default) upsamples every drawn LINE along a monotone
PCHIP interpolant so there are no sharp angles between successive
observations. It must:

* only affect styles that draw a LINE (never marker-only styles),
* keep every original sample as a vertex of the drawn line (data-faithful),
* leave the ANIMATION's frame count and reveal schedule untouched,
* draw, in each frame, exactly the portion of the trajectory that frame
  would have shown, and
* be fully disable-able with ``antialias=False``, which must reproduce the
  raw straight-segment rendering exactly.

Real `hyp.plot` calls throughout -- no mocks.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hypertools as hyp

from hypertools._shared.helpers import antialias_line


def _helix(n=40, d=3):
    """A trajectory with genuinely sharp turns between samples."""
    t = np.linspace(0, 6 * np.pi, n)
    cols = [np.cos(t), np.sin(t), t / 10.0]
    return np.column_stack(cols[:d])


def _line_verts(ax, is_3d=True):
    """Vertex counts of every drawn data line (cube/frame artists excluded)."""
    out = []
    for ln in ax.get_lines():
        n = len(ln.get_data_3d()[0]) if is_3d else len(ln.get_xdata())
        if n > 2:
            out.append(n)
    return out


def _max_turn_angle(pts):
    """Largest turning angle (degrees) between consecutive drawn segments."""
    seg = np.diff(pts, axis=0)
    norm = np.linalg.norm(seg, axis=1)
    keep = norm > 1e-12
    seg, norm = seg[keep], norm[keep]
    if len(seg) < 2:
        return 0.0
    unit = seg / norm[:, None]
    cos = np.clip((unit[:-1] * unit[1:]).sum(axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).max())


# --- the shared helper's contract -----------------------------------------

@pytest.mark.parametrize('n', [2, 3, 17, 100, 780])
def test_antialias_line_keeps_every_sample_and_maps_windows(n):
    """`dense[::step]` is the input exactly, and any original window maps
    onto the dense curve exactly -- the property the animation relies on."""
    arr = np.cumsum(np.random.default_rng(0).standard_normal((n, 3)), axis=0)
    dense, step = antialias_line(arr)
    assert dense.shape[0] >= arr.shape[0]
    np.testing.assert_allclose(dense[::step], arr)
    for a, b in [(0, n), (0, 2), (1, n), (n // 3, n - 1)]:
        if b <= a:
            continue
        window = dense[a * step:(b - 1) * step + 1]
        np.testing.assert_allclose(window[::step], arr[a:b])
        np.testing.assert_allclose(window[0], arr[a])
        np.testing.assert_allclose(window[-1], arr[b - 1])


def test_antialias_line_never_decimates_dense_input():
    """Already-dense trajectories are returned untouched (step == 1)."""
    arr = np.cumsum(np.random.default_rng(1).standard_normal((2000, 3)), axis=0)
    dense, step = antialias_line(arr)
    assert step == 1
    np.testing.assert_array_equal(dense, arr)


# --- static plots ----------------------------------------------------------

def test_static_line_is_smoothed_and_toggleable():
    data = _helix()
    on = hyp.plot(data, fmt='-', show=False)
    off = hyp.plot(data, fmt='-', antialias=False, show=False)
    n_on = max(_line_verts(on.axes[0]))
    n_off = max(_line_verts(off.axes[0]))
    plt.close('all')
    assert n_on > n_off          # antialiasing adds vertices
    assert n_off == len(data)    # disabled == raw samples


def test_static_smoothing_removes_sharp_angles():
    """The actual user-visible property: no sharp angles between samples."""
    data = _helix()
    on = hyp.plot(data, fmt='-', show=False)
    off = hyp.plot(data, fmt='-', antialias=False, show=False)

    def worst(fig):
        ln = max(fig.axes[0].get_lines(),
                 key=lambda l: len(l.get_data_3d()[0]))
        return _max_turn_angle(np.column_stack(ln.get_data_3d()))

    a_on, a_off = worst(on), worst(off)
    plt.close('all')
    assert a_off > 20            # raw data really does turn sharply
    assert a_on < a_off / 3      # smoothing markedly reduces the turn


def test_marker_only_style_is_never_antialiased():
    data = _helix()
    on = hyp.plot(data, fmt='o', show=False)
    off = hyp.plot(data, fmt='o', antialias=False, show=False)
    n_on = len(on.axes[0].get_lines()[0].get_data_3d()[0])
    n_off = len(off.axes[0].get_lines()[0].get_data_3d()[0])
    plt.close('all')
    assert n_on == n_off == len(data)


def test_marker_line_combo_smooths_line_but_keeps_markers_on_samples():
    """'o-' gets a smoothed line PLUS markers at the true sample points."""
    data = _helix()
    fig = hyp.plot(data, fmt='o-', show=False)
    lens = sorted(len(l.get_data_3d()[0]) for l in fig.axes[0].get_lines()
                  if len(l.get_data_3d()[0]) > 2)
    plt.close('all')
    assert lens[0] == len(data)   # markers stay on the raw samples
    assert lens[-1] > len(data)   # the line is densified


# --- animations ------------------------------------------------------------

@pytest.mark.parametrize('style', [True, 'serial', 'spin', 'window'])
def test_animated_line_is_smoothed_without_changing_frame_count(style):
    data = _helix()
    common = dict(fmt='-', animate=style, duration=2, frame_rate=10,
                  show=False, return_model=True)
    on = hyp.plot(data, **common)
    off = hyp.plot(data, antialias=False, **common)
    ani_on, ani_off = on['animation'], off['animation']
    ani_on._func(12, *ani_on._args)
    ani_off._func(12, *ani_off._args)
    n_on = max(_line_verts(on['fig'].axes[0]))
    n_off = max(_line_verts(off['fig'].axes[0]))
    # the animation's pacing is untouched -- only the drawn curve is denser
    assert ani_on._save_count == ani_off._save_count
    plt.close('all')
    assert n_on > n_off


def test_animated_2d_line_is_smoothed():
    data = _helix(d=2)
    common = dict(fmt='-', animate=True, duration=2, frame_rate=10,
                  ndims=2, show=False, return_model=True)
    on = hyp.plot(data, **common)
    off = hyp.plot(data, antialias=False, **common)
    on['animation']._func(12, *on['animation']._args)
    off['animation']._func(12, *off['animation']._args)
    n_on = max(_line_verts(on['fig'].axes[0], is_3d=False))
    n_off = max(_line_verts(off['fig'].axes[0], is_3d=False))
    plt.close('all')
    assert n_on > n_off


def test_animated_frame_draws_same_span_it_would_have_drawn_raw():
    """Antialiasing changes the DENSITY of the drawn window, never its
    extent: the smooth curve starts and ends exactly where the raw window
    would have."""
    data = _helix()
    common = dict(fmt='-', animate=True, duration=2, frame_rate=10,
                  show=False, return_model=True)
    on = hyp.plot(data, **common)
    off = hyp.plot(data, antialias=False, **common)
    a_on, a_off = on['animation'], off['animation']
    for num in (3, 9, 15):
        a_on._func(num, *a_on._args)
        a_off._func(num, *a_off._args)
        p_on = np.column_stack(a_on._args[1][0].get_data_3d())
        p_off = np.column_stack(a_off._args[1][0].get_data_3d())
        assert len(p_on) >= len(p_off)
        np.testing.assert_allclose(p_on[0], p_off[0], atol=1e-9)
        np.testing.assert_allclose(p_on[-1], p_off[-1], atol=1e-9)
    plt.close('all')


def test_animated_marker_only_is_never_antialiased():
    data = _helix()
    common = dict(fmt='o', animate=True, duration=2, frame_rate=10,
                  show=False, return_model=True)
    on = hyp.plot(data, **common)
    off = hyp.plot(data, antialias=False, **common)
    on['animation']._func(12, *on['animation']._args)
    off['animation']._func(12, *off['animation']._args)
    n_on = len(on['animation']._args[1][0].get_data_3d()[0])
    n_off = len(off['animation']._args[1][0].get_data_3d()[0])
    plt.close('all')
    assert n_on == n_off


def test_animated_point_labels_still_track_their_datapoint():
    """Regression guard: antialiasing must not shift per-point labels, which
    index the ORIGINAL rows (it is applied at draw time only)."""
    traj = _helix(n=30)
    labels = [None] * 30
    labels[4], labels[22] = 'AAA', 'BBB'
    fig, ani = hyp.plot(traj, '-o', labels=labels, animate='window',
                        focused=1.0, duration=3, frame_rate=20, show=False)
    anns = {a.get_text(): a for a in fig.axes[0].get_children()
            if getattr(a, '_hyp_point_idx', None) is not None}
    assert set(anns) == {'AAA', 'BBB'}
    seen = {'AAA': 0, 'BBB': 0}
    for fr in range(ani._save_count):
        ani._func(fr, *ani._args)
        for k, a in anns.items():
            seen[k] += bool(a.get_visible())
    plt.close('all')
    # each label is visible during part of the animation, not all of it
    assert 0 < seen['AAA'] < ani._save_count
    assert 0 < seen['BBB'] < ani._save_count


# --- forecasts -------------------------------------------------------------

def test_predict_forecast_overlay_is_antialiased_and_toggleable():
    data = _helix(n=30)
    t = 5
    on = hyp.plot(data, predict='Kalman', t=t, show=False)
    off = hyp.plot(data, predict='Kalman', t=t, antialias=False, show=False)

    def fc_len(fig):
        # forecast artists identify THEMSELVES (`_hyp_forecast_role`);
        # linestyle is not a discriminator -- since 1.1.0 a forecast inherits
        # its observed trace's linestyle, so it is solid here.
        fc = [l for l in fig.axes[0].get_lines()
              if getattr(l, '_hyp_forecast_role', None) == 'static']
        assert fc, 'no forecast overlay drawn'
        return len(fc[0].get_xdata())

    n_on, n_off = fc_len(on), fc_len(off)
    plt.close('all')
    assert n_off == t + 1        # t forecast rows + the prepended seam vertex
    assert n_on > n_off          # smoothed for drawing
