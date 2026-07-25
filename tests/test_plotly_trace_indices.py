# -*- coding: utf-8 -*-
"""Regression test for a latent trace-mis-indexing bug in the plotly
backend's frame-update logic (`_add_animation`).

Trace construction order in `plotly_draw` is: data traces, then (if
`forecasts=` is given) one dashed forecast trace per dataset, then (if
`animate` + chemtrails/precog/bullettime) one trail trace per dataset,
then the cube trace. `_add_animation` used to assume trail traces are
*immediately after* the data traces (`range(n_data_traces +
n_trail_traces)`), which only holds when no forecast traces are present.
When both forecasts and trails coexist, the forecast traces sit between
data and trails, shifting the trails to the right -- so frame updates
would overwrite the (static) forecast traces with trail geometry every
frame, while the real trail traces are never updated and stay empty.

`hyp.plot()`'s public `predict=` parameter cannot reach this combination
(`predict=` is allowed only with the camera-only `animate='spin'`, which has
no trail traces; the time-progressing modes that DO draw trails --
`True`/`'parallel'`/`'window'`/etc. -- still raise `NotImplementedError` when
combined with `predict=`, by design), but the lower-level `plotly_draw`
function -- exported from
`hypertools.plot.interactive` and directly unit-tested -- accepts
`forecasts=` and `animate=`/`chemtrails=` independently, so the
combination IS reachable through that entry point.
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from hypertools.plot.plotly_backend import plotly_draw

pytest.importorskip('plotly')


def _walks_and_forecasts(n=30, d=3, t=5):
    rng = np.random.default_rng(0)
    d1 = np.cumsum(rng.standard_normal((n, d)), axis=0)
    d2 = np.cumsum(rng.standard_normal((n, d)), axis=0) + 3
    fc1 = np.vstack([d1[-1], d1[-1] + rng.standard_normal((t, d))])
    fc2 = np.vstack([d2[-1], d2[-1] + rng.standard_normal((t, d))])
    return [d1, d2], [fc1, fc2]


def test_trace_layout_data_forecast_trail_cube():
    """Locks the exact trace order so future insertions between these
    groups fail loudly instead of silently mis-indexing frame updates."""
    data, forecasts = _walks_and_forecasts()
    fig = plotly_draw(data, animate=True, duration=2, tail_duration=1,
                      chemtrails=True, forecasts=forecasts, show=False)

    # 2 data + 2 forecast + 2 trail + 1 cube
    assert len(fig.data) == 7

    dash = [getattr(tr.line, 'dash', None) for tr in fig.data]
    x_lens = [len(tr.x) if tr.x is not None else 0 for tr in fig.data]

    # data traces: solid, full-length
    assert dash[0] == dash[1] == 'solid'
    assert x_lens[0] == x_lens[1] == 30

    # forecast traces: dashed, t+1 points, sandwiched between data and trail
    assert dash[2] == dash[3] == 'dash'
    assert x_lens[2] == x_lens[3] == 6

    # trail traces: solid, start EMPTY (populated only via frame updates)
    assert dash[4] == dash[5] == 'solid'
    assert x_lens[4] == x_lens[5] == 0


def test_frame_updates_target_trails_not_forecasts():
    """The actual bug: frame trace-index lists must point at the real
    trail traces (4, 5), never at the forecast traces (2, 3)."""
    data, forecasts = _walks_and_forecasts()
    fig = plotly_draw(data, animate=True, duration=2, tail_duration=1,
                      chemtrails=True, forecasts=forecasts, show=False)

    assert len(fig.frames) > 0
    for frame in fig.frames:
        assert 2 not in frame.traces, (
            "frame update targets a forecast trace -- trail/forecast "
            "trace indices are mixed up")
        assert 3 not in frame.traces
        assert 4 in frame.traces and 5 in frame.traces

    # sanity: trails actually grow across frames once addressed correctly
    mid = fig.frames[len(fig.frames) // 2]
    trail_data = [d for idx, d in zip(mid.traces, mid.data) if idx in (4, 5)]
    assert all(len(d.x) > 0 for d in trail_data)


def test_forecast_traces_unaffected_by_animation_frames():
    """The forecast dashed traces are static overlays: no frame should ever
    carry an update for them, and they must never appear zeroed-out."""
    data, forecasts = _walks_and_forecasts()
    fig = plotly_draw(data, animate=True, duration=2, tail_duration=1,
                      chemtrails=True, forecasts=forecasts, show=False)

    for frame in fig.frames:
        assert 2 not in frame.traces
        assert 3 not in frame.traces

    # figure-level forecast traces retain their original (t+1)-point data
    assert len(fig.data[2].x) == 6
    assert len(fig.data[3].x) == 6


def test_precog_and_bullettime_also_use_explicit_trail_indices():
    data, forecasts = _walks_and_forecasts()
    for kw in ({'precog': True}, {'bullettime': True}):
        fig = plotly_draw(data, animate=True, duration=2, tail_duration=1,
                          forecasts=forecasts, show=False, **kw)
        for frame in fig.frames:
            assert 2 not in frame.traces and 3 not in frame.traces
            assert 4 in frame.traces and 5 in frame.traces


def test_no_forecasts_trail_indices_unchanged():
    """Without forecasts, trail traces sit right after the data traces --
    same layout as before this fix, confirming no regression."""
    data, _ = _walks_and_forecasts()
    fig = plotly_draw(data, animate=True, duration=2, tail_duration=1,
                      chemtrails=True, show=False)

    # 2 data + 2 trail + 1 cube (no forecast traces)
    assert len(fig.data) == 5
    for frame in fig.frames:
        assert set(frame.traces) == {0, 1, 2, 3}
