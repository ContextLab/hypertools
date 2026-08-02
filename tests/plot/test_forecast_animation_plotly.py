# tests/plot/test_forecast_animation_plotly.py
"""matplotlib/plotly parity for animated predict=."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp

pytest.importorskip('plotly')


def _series(n=2, rows=60, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(axis=0) for _ in range(n)]


def _fc_traces(fig):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') is not None]


def _fc_role(fig, role):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') == role]


def _frame_snapshot(fig, k):
    from hypertools.plot.plotly_backend import _frame_snapshots
    for i, snap in enumerate(_frame_snapshots(fig)):
        if i == k:
            return snap
    raise AssertionError(f'no frame {k}')


def _mpl_live(fig, frame, ani):
    ani._func(frame, *ani._args)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    live = [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) == 'live']
    return [np.array(ln.get_data_3d()).T for ln in live]


# Every time-progressing style this plan newly accepts, because
# `_add_animation` builds frames in FOUR separate branches and a forecast
# wired into only one of them is frozen in the others. `order='serial'` is
# the fourth case: animation-core Task 5 made `order=` orthogonal to
# `animate=`, so `animate=True, order='serial'` reaches the SERIAL branch
# while spelling a parallel style.
STYLES = [
    pytest.param(dict(animate=True), id='parallel'),
    pytest.param(dict(animate='serial'), id='serial'),
    pytest.param(dict(animate='window'), id='window'),
    pytest.param(dict(animate=True, order='serial'), id='order-serial'),
]


@pytest.mark.parametrize('style', STYLES)
def test_plotly_animated_plot_has_a_live_forecast_trace_per_dataset(style):
    fig = hyp.plot(_series(), '-', predict='Kalman', t=3,
                   duration=2, frame_rate=4, backend='plotly', show=False,
                   **style)
    assert len(_fc_role(fig, 'live')) == 2


@pytest.mark.parametrize('style', STYLES)
def test_plotly_forecast_traces_are_updated_per_frame_not_frozen(style):
    """plotly's frame updates address only the data + trail trace ranges
    (plotly_backend.py:3005-3006), so an un-wired forecast trace stays
    frozen at its setup value -- and it must be wired in EVERY branch, not
    just the parallel one."""
    fig = hyp.plot(_series(n=1), '-', predict='Kalman', t=3,
                   duration=4, frame_rate=4, backend='plotly', show=False,
                   **style)
    early = _fc_role(_frame_snapshot(fig, 4), 'live')[0]
    late = _fc_role(_frame_snapshot(fig, 12), 'live')[0]
    assert not np.allclose(np.asarray(early.x, dtype=float),
                           np.asarray(late.x, dtype=float))


@pytest.mark.parametrize('style', STYLES)
def test_plotly_and_matplotlib_draw_the_same_final_frame_forecast(style):
    """Contract 8. At the final frame both backends have revealed the whole
    history, so both draw the full-history forecast in the same display box."""
    kw = dict(predict='Kalman', t=3, duration=4, frame_rate=4,
              antialias=False, show=False, **style)
    data = _series(n=1)
    pl = hyp.plot(data, '-', backend='plotly', **kw)
    mpl_fig, ani = hyp.plot(data, '-', backend='matplotlib', **kw)

    tr = _fc_role(_frame_snapshot(pl, 15), 'live')[0]
    plotly_pts = np.column_stack([np.asarray(tr.x, dtype=float),
                                  np.asarray(tr.y, dtype=float),
                                  np.asarray(tr.z, dtype=float)])
    mpl_pts = _mpl_live(mpl_fig, 15, ani)[0]
    assert plotly_pts.shape == mpl_pts.shape
    assert np.allclose(plotly_pts, mpl_pts, atol=1e-6)


def test_plotly_serial_reveals_one_datasets_forecast_at_a_time():
    """Under a serial reveal only the CURRENTLY-revealing dataset has a
    growing forecast; datasets not yet reached have none, and finished ones
    are frozen at their full-history forecast (the 'freeze' decision). This
    is the behaviour the parallel branch cannot exercise at all."""
    fig = hyp.plot(_series(n=3), '-', predict='Kalman', t=3,
                   animate='serial', duration=6, frame_rate=4,
                   backend='plotly', show=False)
    early = _fc_role(_frame_snapshot(fig, 2), 'live')
    drawn_early = [tr for tr in early
                   if np.asarray(tr.x, dtype=float).size]
    assert len(drawn_early) == 1, 'only the first dataset is being revealed'
    late = _fc_role(_frame_snapshot(fig, 23), 'live')
    drawn_late = [tr for tr in late if np.asarray(tr.x, dtype=float).size]
    assert len(drawn_late) == 3, 'every dataset is revealed by the last frame'


def test_plotly_forecast_stays_inside_the_scene_range():
    """Same Contract 4 guarantee as matplotlib: the box was built to hold it."""
    fig = hyp.plot(_series(), '-', predict='Kalman', t=5, animate=True,
                   duration=4, frame_rate=4, backend='plotly', show=False)
    checked = 0
    for k in range(16):
        for tr in _fc_role(_frame_snapshot(fig, k), 'live'):
            pts = np.concatenate([np.asarray(getattr(tr, a), dtype=float)
                                  for a in ('x', 'y', 'z')])
            if pts.size == 0:
                continue
            checked += 1
            assert pts.min() >= -1.0 - 1e-6 and pts.max() <= 1.0 + 1e-6
    # without this the whole test is a no-op when no forecast trace is ever
    # drawn (or every one is empty): the `continue` swallows the empty case
    # and the loop body simply never runs. An in-range check that inspects
    # nothing reports the same green as one that inspects everything.
    assert checked >= 16, (
        f'expected a drawn live forecast in most of the 16 frames across 2 '
        f'datasets; only {checked} non-empty traces were inspected')


def test_plotly_forecast_trail_traces_carry_decreasing_opacity():
    """Mirrors the chemtrails mechanism: separate traces at a fixed alpha,
    data rewritten per frame (plotly_backend.py:1000, :3241-3250)."""
    fig = hyp.plot(_series(n=1), '-', predict='Kalman', t=3, animate=True,
                   forecast_trail=4, duration=4, frame_rate=4,
                   backend='plotly', show=False)
    trails = _fc_role(fig, 'trail')
    assert len(trails) == 4
    by_age = sorted(trails, key=lambda tr: tr.meta['hyp_forecast_age'])
    alphas = [tr.meta['hyp_forecast_alpha'] for tr in by_age]
    assert alphas == sorted(alphas, reverse=True), alphas
    assert min(alphas) < max(alphas), 'the trail must actually fade'
    live_alpha = _fc_role(fig, 'live')[0].meta['hyp_forecast_alpha']
    assert all(live_alpha > a for a in alphas)
    # the declared alpha is the one actually baked into the rgba colour
    for tr, a in zip(by_age, alphas):
        assert f'{a}' in tr.line.color or str(round(a, 3)) in tr.line.color


def test_plotly_trail_is_populated_by_the_late_frames():
    fig = hyp.plot(_series(n=1), '-', predict='Kalman', t=3, animate=True,
                   forecast_trail=4, duration=4, frame_rate=4,
                   backend='plotly', show=False)
    late = _fc_role(_frame_snapshot(fig, 15), 'trail')
    drawn = [tr for tr in late if np.asarray(tr.x, dtype=float).size]
    assert len(drawn) == 4


def test_plotly_and_matplotlib_agree_on_the_forecast_trace_count():
    kw = dict(predict='Kalman', t=3, animate=True, forecast_trail=4,
              duration=2, frame_rate=4, show=False)
    pl = hyp.plot(_series(), '-', backend='plotly', **kw)
    mpl_fig, ani = hyp.plot(_series(), '-', backend='matplotlib', **kw)
    ax = [a for a in mpl_fig.axes if hasattr(a, 'zaxis')][0]
    mpl_n = len([ln for ln in ax.lines
                 if getattr(ln, '_hyp_forecast_role', None) is not None])
    # 2 datasets x (1 live + 4 trail). Pin the absolute count too: equality
    # alone is satisfied by both backends drawing NOTHING, which is exactly
    # the state this task exists to fix.
    assert mpl_n == 10, f'matplotlib drew {mpl_n} forecast artists, expected 10'
    assert len(_fc_traces(pl)) == mpl_n


def test_plotly_morph_still_refuses_predict():
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(120, 3)) + off for off in (0.0, 4.0)]
    with pytest.raises(NotImplementedError, match='morph'):
        hyp.plot(clouds, '.', predict='Kalman', t=3, animate='morph',
                 morph_samples=120, duration=1, frame_rate=2,
                 backend='plotly', show=False)
