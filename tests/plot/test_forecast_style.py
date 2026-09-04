# -*- coding: utf-8 -*-
"""A forecast inherits the style of the observed trace it continues.

Maintainer contract (1.1.0): a forecast should read as the SAME series
projected forward, so it takes its observed trace's colour, linestyle AND
linewidth, and differs only in transparency -- ``forecast_alpha =
observed_alpha * 0.5``, with an unset alpha counting as matplotlib's opaque
1.0. This deliberately REPLACES the pre-1.1.0 rule (always
``linestyle='--'`` at a hard-coded ``alpha=0.6``).

Both backends express the one policy
(`hypertools.plot.forecast.FORECAST_ALPHA_SCALE`), so every matplotlib
assertion here has a plotly twin.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.forecast import (FORECAST_ALPHA_SCALE, forecast_alpha,
                                      trail_alpha)


def _walk(seed, n=30, d=3, offset=0.0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((n, d)), axis=0) + offset


def _mpl_forecasts(ax, role='static'):
    """Forecast artists identify THEMSELVES -- see docs/animation.rst.

    Linestyle cannot be the discriminator any more (that is the whole point
    of this contract change: a forecast of a solid line is now solid).
    """
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) == role]


def _observed(ax):
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) is None]


def _ply_forecasts(fig, role='static'):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') == role]


def _ply_observed(fig):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') is None
            and tr.mode is not None and 'lines' in str(tr.mode)]


def _ply_alpha(trace):
    """The alpha baked into a plotly trace's ``rgba(...)`` line colour."""
    return float(trace.line.color.rsplit(',', 1)[1].rstrip(') '))


# --- the policy itself -----------------------------------------------------

def test_forecast_alpha_policy():
    assert FORECAST_ALPHA_SCALE == 0.5
    # matplotlib's "no alpha set" IS opaque, so it counts as 1.0
    assert forecast_alpha(None) == pytest.approx(0.5)
    assert forecast_alpha(1.0) == pytest.approx(0.5)
    assert forecast_alpha(0.4) == pytest.approx(0.2)
    assert forecast_alpha(0.0) == pytest.approx(0.0)


# --- alpha: default, scalar, per-dataset list ------------------------------

def test_default_opaque_observed_gives_half_alpha_forecast():
    fig = hyp.plot(_walk(1), predict='Kalman', t=6, show=False)
    ax = fig.axes[0]
    (src,), (fc,) = _observed(ax), _mpl_forecasts(ax)
    plt.close(fig)
    assert src.get_alpha() is None          # opaque
    assert fc.get_alpha() == pytest.approx(0.5)


def test_explicit_scalar_alpha_is_halved():
    fig = hyp.plot(_walk(2), predict='Kalman', t=6, alpha=0.8, show=False)
    ax = fig.axes[0]
    (src,), (fc,) = _observed(ax), _mpl_forecasts(ax)
    plt.close(fig)
    assert src.get_alpha() == pytest.approx(0.8)
    assert fc.get_alpha() == pytest.approx(0.4)


def test_per_dataset_alpha_list_is_halved_dataset_by_dataset():
    """The RELATIONSHIP must hold, not just one number: alpha=[1.0, 0.4]
    gives forecasts [0.5, 0.2], so a faint dataset keeps a faint forecast."""
    data = [_walk(3), _walk(4, offset=6.0)]
    fig = hyp.plot(data, predict='Kalman', t=6, alpha=[1.0, 0.4], show=False)
    ax = fig.axes[0]
    src, fcs = _observed(ax), _mpl_forecasts(ax)
    plt.close(fig)
    assert [s.get_alpha() for s in src] == pytest.approx([1.0, 0.4])
    assert [f.get_alpha() for f in fcs] == pytest.approx([0.5, 0.2])


def test_plotly_alpha_parity_with_matplotlib():
    pytest.importorskip('plotly')
    data = [_walk(5, d=2), _walk(6, d=2, offset=6.0)]
    kw = dict(predict='Kalman', t=6, ndims=2, alpha=[1.0, 0.4], show=False)

    fig = hyp.plot(data, **kw)
    mpl_alphas = [f.get_alpha() for f in _mpl_forecasts(fig.axes[0])]
    plt.close(fig)

    pfig = hyp.plot(data, backend='plotly', **kw)
    ply_alphas = [_ply_alpha(t) for t in _ply_forecasts(pfig)]
    declared = [t.meta['hyp_forecast_alpha'] for t in _ply_forecasts(pfig)]

    assert ply_alphas == pytest.approx([0.5, 0.2])
    assert ply_alphas == pytest.approx(mpl_alphas)
    # the declared alpha and the one baked into the rgba are the SAME float
    assert declared == pytest.approx(ply_alphas)


# --- linestyle inheritance -------------------------------------------------

@pytest.mark.parametrize('linestyle', ['-', ':', '-.', '--'])
def test_forecast_inherits_observed_linestyle(linestyle):
    """A SOLID observed line gives a solid forecast; a dotted one gives a
    dotted forecast. Pre-1.1.0 every one of these came out '--'."""
    fig = hyp.plot(_walk(7), predict='Kalman', t=6, linestyle=linestyle,
                   show=False)
    ax = fig.axes[0]
    (src,), (fc,) = _observed(ax), _mpl_forecasts(ax)
    plt.close(fig)
    assert src.get_linestyle() == linestyle
    assert fc.get_linestyle() == src.get_linestyle()


@pytest.mark.parametrize('linestyle,dash', [('-', 'solid'), (':', 'dot'),
                                            ('-.', 'dashdot'),
                                            ('--', 'dash')])
def test_plotly_forecast_inherits_observed_dash(linestyle, dash):
    pytest.importorskip('plotly')
    fig = hyp.plot(_walk(8, d=2), predict='Kalman', t=6, ndims=2,
                   linestyle=linestyle, backend='plotly', show=False)
    (src,), (fc,) = _ply_observed(fig), _ply_forecasts(fig)
    assert src.line.dash == dash
    assert fc.line.dash == src.line.dash


def test_per_dataset_linestyle_list_is_inherited_dataset_by_dataset():
    data = [_walk(9), _walk(10, offset=6.0)]
    fig = hyp.plot(data, predict='Kalman', t=6, linestyle=['-', ':'],
                   show=False)
    ax = fig.axes[0]
    src, fcs = _observed(ax), _mpl_forecasts(ax)
    plt.close(fig)
    assert [s.get_linestyle() for s in src] == ['-', ':']
    assert [f.get_linestyle() for f in fcs] == ['-', ':']


# --- linewidth inheritance -------------------------------------------------

def test_forecast_inherits_observed_linewidth():
    data = [_walk(11), _walk(12, offset=6.0)]
    fig = hyp.plot(data, predict='Kalman', t=6, linewidth=[0.5, 4.0],
                   show=False)
    ax = fig.axes[0]
    src, fcs = _observed(ax), _mpl_forecasts(ax)
    plt.close(fig)
    assert [s.get_linewidth() for s in src] == pytest.approx([0.5, 4.0])
    assert [f.get_linewidth() for f in fcs] == pytest.approx([0.5, 4.0])


def test_plotly_forecast_inherits_observed_linewidth():
    pytest.importorskip('plotly')
    data = [_walk(13, d=2), _walk(14, d=2, offset=6.0)]
    fig = hyp.plot(data, predict='Kalman', t=6, ndims=2,
                   linewidth=[0.5, 4.0], backend='plotly', show=False)
    src, fcs = _ply_observed(fig), _ply_forecasts(fig)
    assert len(fcs) == len(src) == 2      # not a vacuous comparison
    assert [f.line.width for f in fcs] == pytest.approx(
        [s.line.width for s in src])


# --- colour ----------------------------------------------------------------

def test_forecast_inherits_observed_colour_both_backends():
    pytest.importorskip('plotly')
    data = [_walk(15, d=2), _walk(16, d=2, offset=6.0)]
    kw = dict(predict='Kalman', t=6, ndims=2, color=['red', 'blue'],
              show=False)

    fig = hyp.plot(data, **kw)
    ax = fig.axes[0]
    src, fcs = _observed(ax), _mpl_forecasts(ax)
    assert [f.get_color() for f in fcs] == [s.get_color() for s in src]
    plt.close(fig)

    pfig = hyp.plot(data, backend='plotly', **kw)
    # same rgb, different alpha -- compare the rgb triple only
    def _rgb(trace):
        return trace.line.color.rsplit(',', 1)[0]
    assert len(_ply_forecasts(pfig)) == len(_ply_observed(pfig)) == 2
    assert [_rgb(f) for f in _ply_forecasts(pfig)] == [
        _rgb(s) for s in _ply_observed(pfig)]


# --- trails: a trail is never more opaque than its own live forecast -------

@pytest.mark.parametrize('observed_alpha', [None, 1.0, 0.5, 0.2, 0.05, 0.01])
@pytest.mark.parametrize('n_retained', [1, 4, 16])
def test_trail_alpha_never_exceeds_its_live_forecast(observed_alpha,
                                                     n_retained):
    """The floor is RELATIVE to the live alpha. With the old absolute 0.08
    floor, alpha=0.05 (live 0.025) produced trails at 0.08 -- more than
    3x MORE opaque than the live forecast they were supposed to fade from."""
    live = forecast_alpha(observed_alpha)
    assert trail_alpha(0, n_retained, live_alpha=live) == pytest.approx(live)
    previous = live
    for age in range(1, n_retained + 1):
        a = trail_alpha(age, n_retained, live_alpha=live)
        assert a <= live, 'a trail must never out-shine its live forecast'
        assert a <= previous, 'the fan must not brighten with age'
        assert a > 0, 'emptiness, not alpha 0, is how "nothing here" is said'
        previous = a


def test_trail_alpha_default_matches_an_opaque_dataset():
    """The default live alpha IS the opaque-dataset forecast alpha, so a
    paused animation of default-styled data looks like the static plot."""
    assert trail_alpha(0, 8) == pytest.approx(FORECAST_ALPHA_SCALE)
    assert trail_alpha(3, 8) == pytest.approx(
        trail_alpha(3, 8, live_alpha=forecast_alpha(None)))


def test_animated_trail_artists_fade_from_their_own_live_forecast():
    data = [_walk(17), _walk(18, offset=6.0)]
    fig, ani = hyp.plot(data, '-', predict='Kalman', t=3, animate=True,
                        forecast_trail=3, duration=1, frame_rate=4,
                        alpha=[1.0, 0.2], show=False)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    live = _mpl_forecasts(ax, role='live')
    trails = _mpl_forecasts(ax, role='trail')
    assert [ln.get_alpha() for ln in live] == pytest.approx([0.5, 0.1])
    assert len(trails) == 2 * 3
    for ln in trails:
        # role tags survive, ages survive, and each trail is dimmer than the
        # live forecast of ITS OWN dataset
        assert ln._hyp_forecast_age in (1, 2, 3)
        own_live = live[0] if ln.get_color() == live[0].get_color() \
            else live[1]
        assert ln.get_alpha() < own_live.get_alpha()
    plt.close(fig)


def test_plotly_animated_trail_alpha_parity():
    pytest.importorskip('plotly')
    data = [_walk(19, d=2), _walk(20, d=2, offset=6.0)]
    kw = dict(predict='Kalman', t=3, animate=True, forecast_trail=3,
              duration=1, frame_rate=4, ndims=2, alpha=[1.0, 0.2],
              show=False)

    fig, ani = hyp.plot(data, '-', **kw)
    ax = fig.axes[0]
    mpl_live = [ln.get_alpha() for ln in _mpl_forecasts(ax, role='live')]
    mpl_trail = [ln.get_alpha() for ln in _mpl_forecasts(ax, role='trail')]
    plt.close(fig)

    pfig = hyp.plot(data, '-', backend='plotly', **kw)
    ply_live = [_ply_alpha(t) for t in _ply_forecasts(pfig, role='live')]
    ply_trail = [_ply_alpha(t) for t in _ply_forecasts(pfig, role='trail')]

    assert ply_live == pytest.approx(mpl_live)
    assert sorted(ply_trail) == pytest.approx(sorted(mpl_trail))
    # and every trail stays under its own dataset's live forecast
    for dataset, live_a in enumerate(ply_live):
        own = [t for t in _ply_forecasts(pfig, role='trail')
               if t.meta['hyp_dataset'] == dataset]
        assert own, 'expected a fan per dataset'
        assert all(_ply_alpha(t) < live_a for t in own)


# --- role tags survive the restyle ----------------------------------------

def test_role_tags_and_ages_survive():
    """The tags are the documented way to find forecast artists, and are the
    ONLY way now that linestyle is inherited (docs/animation.rst)."""
    pytest.importorskip('plotly')
    data = [_walk(21, d=2)]
    fig, ani = hyp.plot(data, '-', predict='Kalman', t=3, animate=True,
                        forecast_trail=2, duration=1, frame_rate=4, ndims=2,
                        show=False)
    ax = fig.axes[0]
    assert len(_mpl_forecasts(ax, role='live')) == 1
    assert sorted(ln._hyp_forecast_age
                  for ln in _mpl_forecasts(ax, role='trail')) == [1, 2]
    assert _mpl_forecasts(ax, role='static') == []
    plt.close(fig)

    pfig = hyp.plot(data, '-', predict='Kalman', t=3, animate=True,
                    forecast_trail=2, duration=1, frame_rate=4, ndims=2,
                    backend='plotly', show=False)
    assert len(_ply_forecasts(pfig, role='live')) == 1
    assert sorted(t.meta['hyp_forecast_age']
                  for t in _ply_forecasts(pfig, role='trail')) == [1, 2]
