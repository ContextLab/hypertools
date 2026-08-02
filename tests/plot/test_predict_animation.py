# tests/plot/test_predict_animation.py
"""`predict=` with time-progressing animations (matplotlib backend)."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp


def _series(n=3, rows=60, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _forecasts(ax, role=None):
    """Forecast artists identify THEMSELVES (Contract 5). Linestyle is not a
    discriminator: user data drawn with fmt='--' is dashed too."""
    out = [ln for ln in ax.lines
           if getattr(ln, '_hyp_forecast_role', None) is not None]
    if role is not None:
        out = [ln for ln in out if ln._hyp_forecast_role == role]
    return out


def test_predict_with_animate_true_no_longer_raises():
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        duration=2, frame_rate=4, show=False)
    assert ani is not None


@pytest.mark.parametrize('mode', [True, 'parallel', 'serial', 'window'])
def test_time_progressing_animation_draws_no_static_full_history_overlay(mode):
    """plot.py:4907 had no `animate` guard, so a time-progressing animation
    would draw BOTH a frozen full-history overlay AND the per-frame one.
    Measured before the fix on animate='spin': 3 dashed 901-vertex overlays,
    landing FIRST in ax.lines."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=mode,
                        duration=2, frame_rate=4, show=False)
    assert _forecasts(_ax(fig), role='static') == []


def test_spin_still_draws_the_static_overlay():
    """Regression: 'spin' only rotates the camera, so its fixed overlay is
    correct and must be untouched -- including alpha/label/clip."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate='spin',
                        duration=2, frame_rate=4, show=False)
    ax = _ax(fig)
    static = _forecasts(ax, role='static')
    assert len(static) == 3
    for fc in static:
        assert fc.get_linestyle() == '--'
        assert fc.get_alpha() == pytest.approx(0.6)
        assert fc.get_label() == '_nolegend_'
        assert fc.get_clip_on() is False
    ani._func(1, *ani._args)
    first = [np.array(ln.get_data_3d()) for ln in static]
    ani._func(6, *ani._args)
    for a, ln in zip(first, static):
        assert np.allclose(a, np.array(ln.get_data_3d())), \
            'spin forecast overlay must stay fixed'


def test_static_plot_still_draws_the_static_overlay():
    fig = hyp.plot(_series(), '-', predict='Kalman', t=3, show=False)
    assert len(_forecasts(_ax(fig), role='static')) == 3


def test_scalar_morph_still_refuses_predict():
    """A morph interpolates between point clouds; there is no time axis."""
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(120, 3)) + off for off in (0.0, 4.0)]
    with pytest.raises(NotImplementedError, match='morph'):
        hyp.plot(clouds, '.', predict='Kalman', t=3, animate='morph',
                 morph_samples=120, duration=1, frame_rate=2, show=False)


def test_list_form_morph_still_refuses_predict():
    """`_resolve_animate_mode` runs at plot.py:4158, ~1400 lines AFTER the
    refusal at plot.py:2740 -- so at the check `animate` is still a raw list
    and `animate == 'morph'` is False. Naive narrowing would silently ACCEPT
    a per-dataset morph list into the forecast path."""
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(120, 3)) + off for off in (0.0, 4.0)]
    with pytest.raises(NotImplementedError, match='morph'):
        hyp.plot(clouds, '.', predict='Kalman', t=3,
                 animate=['morph', 'morph'], morph_samples=120,
                 duration=1, frame_rate=2, show=False)
