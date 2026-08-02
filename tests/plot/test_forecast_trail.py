# tests/plot/test_forecast_trail.py
"""The retained forecast fan -- the forecast analogue of chemtrails=."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp


def _series(n=1, rows=60, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _forecasts(ax, role=None):
    out = [ln for ln in ax.lines
           if getattr(ln, '_hyp_forecast_role', None) is not None]
    if role is not None:
        out = [ln for ln in out if ln._hyp_forecast_role == role]
    return out


def _drawn(ax, role=None):
    """Artists that are actually on screen. A preallocated-but-unwritten slot
    is hidden with EMPTY data -- alpha is not the emptiness signal (the v1
    plan's trail_alpha never returned 0, so an alpha>0 count could not grow)."""
    return [ln for ln in _forecasts(ax, role)
            if ln.get_visible() and np.array(ln.get_data_3d()).size]


def _drive(ani, upto):
    for f in range(upto + 1):
        ani._func(f, *ani._args)


def test_trail_accumulates_past_forecasts():
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        forecast_trail=True, duration=4, frame_rate=4,
                        show=False)
    ax = _ax(fig)
    _drive(ani, 3)
    early = len(_drawn(ax))
    _drive(ani, 14)
    late = len(_drawn(ax))
    assert late > early, f'trail should accumulate; got {early} -> {late}'


def test_without_trail_only_the_live_forecast_is_drawn():
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        duration=4, frame_rate=4, show=False)
    ax = _ax(fig)
    _drive(ani, 12)
    assert len(_drawn(ax)) == 1
    assert _drawn(ax)[0]._hyp_forecast_role == 'live'


def test_trail_is_capped_by_an_integer():
    """Driven SEQUENTIALLY: a single _func(20) call could satisfy any cap."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        forecast_trail=4, duration=6, frame_rate=4,
                        show=False)
    ax = _ax(fig)
    _drive(ani, 23)
    # BOTH bounds. An upper bound alone passes when nothing is drawn at all:
    # with no trail implemented `_drawn` returns just the live forecast, and
    # `1 <= 5` / `0 <= 4` are both true -- so the test could not fail for the
    # reason it exists. By frame 23 a cap of 4 must be SATURATED.
    assert len(_drawn(ax, role='trail')) == 4, (
        f'expected the cap to be reached by frame 23; got '
        f"{len(_drawn(ax, role='trail'))} trail artists")
    assert len(_drawn(ax)) == 5, 'cap of 4 past forecasts plus the live one'


def test_an_uncapped_trail_retains_more_than_a_capped_one():
    """Proves the cap is what limits the fan, not the frame count."""
    kw = dict(predict='Kalman', t=3, animate=True, duration=6, frame_rate=4,
              show=False)
    big, ani_big = hyp.plot(_series(), '-', forecast_trail=16, **kw)
    small, ani_small = hyp.plot(_series(), '-', forecast_trail=2, **kw)
    _drive(ani_big, 23)
    _drive(ani_small, 23)
    assert len(_drawn(_ax(big))) > len(_drawn(_ax(small)))


def test_live_forecast_is_strictly_more_opaque_than_every_trail():
    """T1: the v1 assertion `max(a) == a[0] or max(a) >= sorted(a)[-1]` was a
    tautology (`sorted(a)[-1]` IS `max(a)`). Roles make this checkable."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        forecast_trail=True, duration=4, frame_rate=4,
                        show=False)
    ax = _ax(fig)
    _drive(ani, 14)
    live = _drawn(ax, role='live')[0]
    trails = _drawn(ax, role='trail')
    assert trails, 'expected a fan by frame 14'
    assert all(live.get_alpha() > tr.get_alpha() for tr in trails)


def test_trail_alpha_decreases_with_age():
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        forecast_trail=True, duration=4, frame_rate=4,
                        show=False)
    ax = _ax(fig)
    _drive(ani, 14)
    by_age = sorted(_drawn(ax, role='trail'), key=lambda ln: ln._hyp_forecast_age)
    alphas = [ln.get_alpha() for ln in by_age]
    # guard first: `[] == sorted([])` is True, so the ordering assertion
    # below passes on an empty fan, and `min([])` would then fail with
    # "min() arg is an empty sequence" -- a confusing error for a plain
    # "no trail was drawn"
    assert len(alphas) >= 2, (
        f'expected at least two trail artists by frame 14 to compare '
        f'alphas; got {len(alphas)}')
    assert alphas == sorted(alphas, reverse=True), alphas
    assert min(alphas) < max(alphas), 'the trail must actually fade'


def test_the_fan_is_a_pure_function_of_the_frame_index():
    """G5: FuncAnimation replays from frame 0 for save()/to_jshtml(), and the
    tests above drive frames out of order. A ring buffer would diverge."""
    fig, ani = hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                        forecast_trail=True, duration=4, frame_rate=4,
                        show=False)
    ax = _ax(fig)
    _drive(ani, 12)
    sequential = [np.array(ln.get_data_3d()) for ln in _drawn(ax)]
    for f in (0, 15, 3, 12):
        ani._func(f, *ani._args)
    jumped = [np.array(ln.get_data_3d()) for ln in _drawn(ax)]
    # non-empty guard: `zip([], [])` iterates zero times, so without this the
    # whole comparison below is a no-op and the test passes when NOTHING is
    # drawn -- an assertion that cannot fail is worth exactly nothing
    assert len(sequential) > 1, (
        f'expected a live forecast plus a fan at frame 12; got '
        f'{len(sequential)} drawn artists')
    assert len(sequential) == len(jumped)
    for a, b in zip(sequential, jumped):
        assert np.allclose(a, b)


def test_forecast_trail_requires_predict():
    with pytest.raises(ValueError, match='forecast_trail= requires predict='):
        hyp.plot(_series(), '-', animate=True, forecast_trail=True,
                 duration=2, frame_rate=4, show=False)


@pytest.mark.parametrize('bad', [-1, 'yes', 2.5])
def test_invalid_forecast_trail_raises(bad):
    with pytest.raises((ValueError, TypeError), match='forecast_trail'):
        hyp.plot(_series(), '-', predict='Kalman', t=3, animate=True,
                 forecast_trail=bad, duration=2, frame_rate=4, show=False)
