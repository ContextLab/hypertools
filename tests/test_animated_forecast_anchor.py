"""An animated forecast starts at the CURRENT frame's drawn endpoint.

Maintainer report, 2026-09-11: "for animations with predictions, the
predictions should show *from the endpoint in the current frame* not just
from the last observation."

An animation is paced on a refined frame grid (`plot._interp_anim_line`), so
the drawn head usually sits BETWEEN two raw observations. The forecast was
anchored on the last raw observation at or before it, so the forecast hung
back from the line's tip and only caught up on the frames where the head
happened to land on a raw row. Measured before the fix, on a 2-D spiral
drawn into a ``[-1, 1]`` box:

===================  =========  ===============  ==============
rows / frames        max gap    frames with gap  longest stall
===================  =========  ===============  ==============
20 rows / 40 frames  0.21       24 / 37          2 frames
12 rows / 90 frames  0.51       71 / 81          8 frames
8 rows / 160 frames  0.90       130 / 137        23 frames
===================  =========  ===============  ==============

A "stall" is consecutive frames over which the head advances while the
forecast's start stays put -- the artifact the report describes.

The contract these tests pin: on EVERY frame, on BOTH backends, the first
vertex of the live forecast is exactly the last vertex the observed
trajectory draws. The forecast's own predicted points are unchanged -- only
the vertex it hangs from moves -- so `t=` still means the same raw steps.
"""

import os

os.environ.setdefault('PLOTLY_RENDERER', 'json')

import matplotlib                                             # noqa: E402

matplotlib.use('Agg')

import numpy as np                                            # noqa: E402
import pytest                                                 # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402

import hypertools as hyp                                      # noqa: E402


def spiral(n):
    """A trajectory whose successive raw steps are large enough that an
    anchor one observation behind the head is visible, not a rounding
    difference."""
    th = np.linspace(0, 2.2 * np.pi, n)
    return np.c_[np.cos(th) * (1 + th / 8), np.sin(th) * (1 + th / 8)]


def matplotlib_frames(n_rows, duration, frame_rate, animate=True, **kwargs):
    """(head, forecast start) per frame, read off the drawn artists."""
    anim = hyp.plot(spiral(n_rows), '-', predict='Kalman', t=3,
                    animate=animate, duration=duration, frame_rate=frame_rate,
                    show=False, backend='matplotlib',
                    slow_warning_seconds=None, **kwargs)
    try:
        ax = anim.figure.axes[0]
        for frame in range(anim.n_frames):
            anim.draw_frame(frame)
            observed = [ln for ln in ax.get_lines()
                        if getattr(ln, '_hyp_forecast_role', None) is None
                        and len(ln.get_xdata())]
            live = [ln for ln in ax.get_lines()
                    if getattr(ln, '_hyp_forecast_role', None) == 'live'
                    and len(ln.get_xdata())]
            if not observed or not live:
                continue
            head = np.array([observed[0].get_xdata()[-1],
                             observed[0].get_ydata()[-1]])
            start = np.array([live[0].get_xdata()[0],
                              live[0].get_ydata()[0]])
            yield frame, head, start
    finally:
        plt.close(anim.figure)


def plotly_frames(n_rows, duration, frame_rate, animate=True, **kwargs):
    """The same measurement on the plotly backend, off the frame payloads."""
    fig = hyp.plot(spiral(n_rows), '-', predict='Kalman', t=3,
                   animate=animate, duration=duration, frame_rate=frame_rate,
                   show=False, backend='plotly', slow_warning_seconds=None,
                   **kwargs)
    meta = [tr.meta if isinstance(tr.meta, dict) else {} for tr in fig.data]
    data_slot = next(i for i, m in enumerate(meta)
                     if 'hyp_forecast_role' not in m)
    live_slot = next(i for i, m in enumerate(meta)
                     if m.get('hyp_forecast_role') == 'live')
    for frame, payload in enumerate(fig.frames):
        slots = (list(payload.traces) if payload.traces is not None
                 else list(range(len(payload.data))))
        by_slot = dict(zip(slots, payload.data))
        observed, live = by_slot.get(data_slot), by_slot.get(live_slot)
        if observed is None or live is None:
            continue
        if observed.x is None or live.x is None:
            continue
        if not len(observed.x) or not len(live.x):
            continue
        head = np.array([np.asarray(observed.x, float)[-1],
                         np.asarray(observed.y, float)[-1]])
        start = np.array([np.asarray(live.x, float)[0],
                          np.asarray(live.y, float)[0]])
        yield frame, head, start


READERS = {'matplotlib': matplotlib_frames, 'plotly': plotly_frames}


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
@pytest.mark.parametrize('n_rows,duration,frame_rate', [
    (20, 4, 10),        # the report's shape: head between rows every 2nd frame
    (12, 6, 15),        # coarser data, finer frame grid
    (8, 8, 20),         # the worst measured case: a 23-frame stall
])
def test_forecast_starts_at_the_frames_drawn_endpoint(backend, n_rows,
                                                      duration, frame_rate):
    """The live forecast hangs off the vertex the line actually ends on."""
    checked = 0
    for frame, head, start in READERS[backend](n_rows, duration, frame_rate):
        np.testing.assert_allclose(
            start, head, atol=1e-9,
            err_msg=(f'{backend} frame {frame}: the forecast starts at '
                     f'{start}, but the observed trajectory ends at {head} '
                     f'-- the forecast must continue from the CURRENT '
                     f"frame's endpoint, not from the last raw observation "
                     f'at or before it.'))
        checked += 1
    assert checked > 3, (
        f'{backend}: only {checked} frames drew a forecast, too few to show '
        f'the anchor tracking the head')


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
@pytest.mark.parametrize('animate', ['parallel', 'serial', 'window'])
def test_every_time_progressing_style_hangs_off_its_own_head(backend, animate):
    """The head table is built from each style's OWN reveal --
    `serial_reveal_counts` for `'serial'`, `trails.anim_window_bounds` for
    `'parallel'`/`'window'` -- so each style is checked rather than trusting
    one of them to stand for the others."""
    checked = 0
    for frame, head, start in READERS[backend](12, 6, 15, animate=animate):
        np.testing.assert_allclose(
            start, head, atol=1e-9,
            err_msg=(f'{backend} animate={animate!r} frame {frame}: forecast '
                     f'starts at {start}, trajectory ends at {head}'))
        checked += 1
    assert checked > 3, f'{backend} animate={animate!r}: only {checked} frames'


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_the_forecast_start_advances_on_every_frame(backend):
    """The artifact was a STALL: the head moved while the forecast's start
    stayed put for up to 23 consecutive frames. Once the reveal is under way,
    a moving head must move the forecast with it."""
    seen = [(frame, head, start)
            for frame, head, start in READERS[backend](8, 8, 20)]
    stalls = 0
    for (_, prev_head, prev_start), (frame, head, start) in zip(seen, seen[1:]):
        head_moved = not np.allclose(head, prev_head, atol=1e-12)
        start_moved = not np.allclose(start, prev_start, atol=1e-12)
        if head_moved and not start_moved:
            stalls += 1
    assert stalls == 0, (
        f'{backend}: the forecast start stood still on {stalls} frame(s) '
        f'whose drawn head had moved')


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_a_model_comparison_anchors_every_models_forecast(backend):
    """`predict=[...]` draws one forecast per model per dataset behind one
    flat model-major index, so the head lookups go through
    `MultiModelSchedule`, which forwards each slot to the sub-schedule that
    owns it. Every model's forecast must start at the same drawn head."""
    models = ['Kalman', 'ARIMA']
    checked = 0
    if backend == 'matplotlib':
        anim = hyp.plot(spiral(12), '-', predict=models, t=3, animate=True,
                        duration=6, frame_rate=15, show=False,
                        backend='matplotlib', slow_warning_seconds=None)
        try:
            ax = anim.figure.axes[0]
            for frame in range(anim.n_frames):
                anim.draw_frame(frame)
                observed = [ln for ln in ax.get_lines()
                            if getattr(ln, '_hyp_forecast_role', None) is None
                            and len(ln.get_xdata())]
                live = [ln for ln in ax.get_lines()
                        if getattr(ln, '_hyp_forecast_role', None) == 'live'
                        and len(ln.get_xdata())]
                if not observed or not live:
                    continue
                head = np.array([observed[0].get_xdata()[-1],
                                 observed[0].get_ydata()[-1]])
                for line in live:
                    start = np.array([line.get_xdata()[0],
                                      line.get_ydata()[0]])
                    np.testing.assert_allclose(
                        start, head, atol=1e-9,
                        err_msg=(f'frame {frame}: one model\'s forecast '
                                 f'starts at {start}, head is {head}'))
                    checked += 1
        finally:
            plt.close(anim.figure)
    else:
        fig = hyp.plot(spiral(12), '-', predict=models, t=3, animate=True,
                       duration=6, frame_rate=15, show=False,
                       backend='plotly', slow_warning_seconds=None)
        meta = [tr.meta if isinstance(tr.meta, dict) else {} for tr in fig.data]
        data_slot = next(i for i, m in enumerate(meta)
                         if 'hyp_forecast_role' not in m)
        live_slots = [i for i, m in enumerate(meta)
                      if m.get('hyp_forecast_role') == 'live']
        assert len(live_slots) == len(models), live_slots
        for frame, payload in enumerate(fig.frames):
            slots = (list(payload.traces) if payload.traces is not None
                     else list(range(len(payload.data))))
            by_slot = dict(zip(slots, payload.data))
            observed = by_slot.get(data_slot)
            if observed is None or observed.x is None or not len(observed.x):
                continue
            head = np.array([np.asarray(observed.x, float)[-1],
                             np.asarray(observed.y, float)[-1]])
            for slot in live_slots:
                live = by_slot.get(slot)
                if live is None or live.x is None or not len(live.x):
                    continue
                start = np.array([np.asarray(live.x, float)[0],
                                  np.asarray(live.y, float)[0]])
                np.testing.assert_allclose(
                    start, head, atol=1e-9,
                    err_msg=(f'frame {frame} slot {slot}: forecast starts at '
                             f'{start}, head is {head}'))
                checked += 1
    assert checked > 6, f'{backend}: only {checked} forecast/frame pairs drawn'


def test_a_retained_trail_forecast_starts_at_the_head_it_was_fit_from():
    """`forecast_trail=` keeps earlier forecasts on screen so a viewer can
    see how the prediction changed as history accumulated. Each retained
    forecast therefore has to start where the trajectory ENDED when it was
    fit -- otherwise the fan shows predictions hanging off points the line
    never visibly reached."""
    from hypertools.plot.forecast import DEFAULT_FORECAST_TRAIL, trail_frames

    anim = hyp.plot(spiral(8), '-', predict='Kalman', t=3, animate=True,
                    forecast_trail=True, duration=8, frame_rate=20,
                    show=False, backend='matplotlib',
                    slow_warning_seconds=None)
    try:
        ax = anim.figure.axes[0]

        def drawn_head():
            observed = [ln for ln in ax.get_lines()
                        if getattr(ln, '_hyp_forecast_role', None) is None
                        and len(ln.get_xdata())]
            if not observed:
                return None
            return np.array([observed[0].get_xdata()[-1],
                             observed[0].get_ydata()[-1]])

        heads = {}
        for frame in range(anim.n_frames):
            anim.draw_frame(frame)
            head = drawn_head()
            if head is not None:
                heads[frame] = head

        checked = 0
        for frame in range(anim.n_frames // 2, anim.n_frames):
            anim.draw_frame(frame)
            retained = [ln for ln in ax.get_lines()
                        if getattr(ln, '_hyp_forecast_role', None) == 'trail'
                        and len(ln.get_xdata())]
            # membership, not order: the artists carry no age we can read
            # back, and the contract is that each one hangs off SOME earlier
            # frame's head -- which a raw-observation anchor would fail.
            expected = [heads[p] for p in trail_frames(
                frame, DEFAULT_FORECAST_TRAIL) if p in heads]
            for line in retained:
                start = np.array([line.get_xdata()[0], line.get_ydata()[0]])
                assert any(np.allclose(start, head, atol=1e-9)
                           for head in expected), (
                    f'frame {frame}: a retained forecast starts at {start}, '
                    f'which is not the drawn head of any frame it could have '
                    f'been fit at')
                checked += 1
        assert checked > 0, 'no retained forecasts were drawn to check'
    finally:
        plt.close(anim.figure)


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_a_regrouped_hue_animation_anchors_on_the_drawn_head(backend):
    """`hue=` cuts a dataset into one drawn trace per contiguous category
    run, so the reveal is expressed in RUNS rather than in the dataset's own
    rows and the head belongs to whichever run is currently drawing it.
    Measured before this path was covered: the forecast trailed the drawn
    head by up to 0.51, on 75 of 81 frames."""
    hue = ['early'] * 6 + ['late'] * 6
    checked = 0
    if backend == 'matplotlib':
        anim = hyp.plot(spiral(12), '-', hue=hue, predict='Kalman', t=3,
                        animate=True, duration=6, frame_rate=15, show=False,
                        backend='matplotlib', slow_warning_seconds=None)
        try:
            ax = anim.figure.axes[0]
            for frame in range(anim.n_frames):
                anim.draw_frame(frame)
                drawn = [ln for ln in ax.get_lines()
                         if getattr(ln, '_hyp_forecast_role', None) is None
                         and len(ln.get_xdata())]
                live = [ln for ln in ax.get_lines()
                        if getattr(ln, '_hyp_forecast_role', None) == 'live'
                        and len(ln.get_xdata())]
                if not drawn or not live:
                    continue
                # the run furthest along owns the head
                head = np.array([drawn[-1].get_xdata()[-1],
                                 drawn[-1].get_ydata()[-1]])
                start = np.array([live[0].get_xdata()[0],
                                  live[0].get_ydata()[0]])
                np.testing.assert_allclose(
                    start, head, atol=1e-9,
                    err_msg=(f'frame {frame}: the regrouped forecast starts '
                             f'at {start}, the drawn head is {head}'))
                checked += 1
        finally:
            plt.close(anim.figure)
    else:
        fig = hyp.plot(spiral(12), '-', hue=hue, predict='Kalman', t=3,
                       animate=True, duration=6, frame_rate=15, show=False,
                       backend='plotly', slow_warning_seconds=None)
        meta = [tr.meta if isinstance(tr.meta, dict) else {} for tr in fig.data]
        data_slots = [i for i, m in enumerate(meta)
                      if 'hyp_forecast_role' not in m]
        live_slots = [i for i, m in enumerate(meta)
                      if m.get('hyp_forecast_role') == 'live']
        for frame, payload in enumerate(fig.frames):
            slots = (list(payload.traces) if payload.traces is not None
                     else list(range(len(payload.data))))
            by_slot = dict(zip(slots, payload.data))
            drawn = [by_slot[s] for s in data_slots
                     if s in by_slot and by_slot[s].x is not None
                     and len(by_slot[s].x)]
            live = [by_slot[s] for s in live_slots
                    if s in by_slot and by_slot[s].x is not None
                    and len(by_slot[s].x)]
            if not drawn or not live:
                continue
            head = np.array([np.asarray(drawn[-1].x, float)[-1],
                             np.asarray(drawn[-1].y, float)[-1]])
            start = np.array([np.asarray(live[0].x, float)[0],
                              np.asarray(live[0].y, float)[0]])
            np.testing.assert_allclose(
                start, head, atol=1e-9,
                err_msg=(f'frame {frame}: the regrouped forecast starts at '
                         f'{start}, the drawn head is {head}'))
            checked += 1
    assert checked > 3, f'{backend}: only {checked} frames drew a forecast'


def test_the_predicted_points_themselves_are_unchanged():
    """Re-anchoring moves only the vertex the forecast hangs from. Every
    point the model actually predicted keeps its place, so `t=` still counts
    the same raw steps from the last OBSERVATION."""
    from hypertools.plot.forecast import ForecastSchedule

    history = np.cumsum(np.ones((10, 2)), axis=0)
    grid = np.linspace(0, 9, 46)
    dense = np.column_stack([np.interp(grid, np.arange(10), history[:, col])
                             for col in range(2)])
    schedule = ForecastSchedule.for_parallel(
        [history], [len(dense)], model='Kalman', t=3, n_frames=40,
        slow_warning_seconds=None, grids=[dense])
    for frame in range(40):
        drawn = schedule.polyline(0, frame)
        if drawn is None:
            continue
        rows = schedule.revealed_rows(0, frame)
        path = schedule.path(0, frame)
        predicted = history[rows[-1]] + path[1:]
        np.testing.assert_allclose(
            drawn[1:], predicted, atol=1e-12,
            err_msg=f'frame {frame}: a predicted point moved')
