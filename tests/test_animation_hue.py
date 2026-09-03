"""Animating with per-observation hue must not crash (QC 2026-07).

Before this fix, `hyp.plot(X, animate='spin', hue=<continuous>)` raised
"ValueError: `x` must contain at least 2 elements": an animated continuous hue
was excluded from the exact-per-point-color path and fell into the categorical
regroup, which split it into single-point "groups" that crashed the per-frame
interpolation. Every non-morph animation now uses the same per-point-color path
as static plots. morph interpolates between point CLOUDS, so per-observation
hue there is dropped with a warning rather than crashing.

Real data, no mocks; headless (Agg).
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import hypertools as hyp
from hypertools import HyperAnimation


def _traj(n=30, d=3):
    return np.random.default_rng(0).normal(size=(n, d)).cumsum(axis=0)


@pytest.mark.parametrize('style', ['spin', 'window', 'parallel', 'serial'])
def test_animate_with_continuous_hue_does_not_crash(style):
    out = hyp.plot(_traj(), '-', ndims=3, animate=style, duration=2,
                   hue=np.linspace(0, 1, 30), show=False)
    assert isinstance(out, HyperAnimation)


@pytest.mark.parametrize('style', ['spin', 'window', 'parallel'])
def test_animate_with_matrix_hue_does_not_crash(style):
    hue = np.random.default_rng(1).random((30, 4))
    out = hyp.plot(_traj(), '.', ndims=3, animate=style, duration=2, hue=hue,
                   show=False)
    assert isinstance(out, HyperAnimation)


def test_animate_hue_uses_per_point_colors():
    # the per-point colors actually vary across observations (not one flat color)
    from hypertools.plot import plot as plot_mod
    captured = {}
    orig = plot_mod._multicolor_line_colors

    def _spy(hue_src, orig_lengths, xform, palette, is_rgb=False):
        res = orig(hue_src, orig_lengths, xform, palette, is_rgb=is_rgb)
        captured['colors'] = res
        return res

    plot_mod._multicolor_line_colors = _spy
    try:
        hyp.plot(_traj(), '-', ndims=3, animate='spin', duration=2,
                 hue=np.linspace(0, 1, 30), show=False)
    finally:
        plot_mod._multicolor_line_colors = orig
    colors = np.vstack(captured['colors'])
    assert len(np.unique(np.round(colors, 3), axis=0)) > 5


@pytest.mark.parametrize('hue', [np.linspace(0, 1, 60), ['a', 'b']],
                         ids=['continuous', 'categorical'])
def test_morph_with_hue_warns_and_does_not_crash(hue):
    x = [_traj(), _traj() + 3]
    with pytest.warns(UserWarning, match="hue is not supported with animate='morph'"):
        out = hyp.plot(x, '.', ndims=3, animate='morph', duration=2, hue=hue,
                       show=False)
    assert isinstance(out, HyperAnimation)


def test_morph_without_hue_still_works():
    out = hyp.plot([_traj(), _traj() + 3], ndims=3, animate='morph',
                   duration=2, show=False)
    assert isinstance(out, HyperAnimation)


def test_static_continuous_hue_unaffected():
    import matplotlib.figure
    out = hyp.plot(_traj(), '-', ndims=3, hue=np.linspace(0, 1, 30), show=False)
    assert isinstance(out, matplotlib.figure.Figure)
