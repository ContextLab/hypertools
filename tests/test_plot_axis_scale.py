"""`axis_scale=` -- raw data coordinates instead of the unit frame box.

GH #285: every 2-D plot used to be mean-centred, rescaled into ``[-1, 1]``
and pinned to ``xlim/ylim=(-1.1, 1.1)``, so ``hyp.plot(..., reduce=None,
ndims=2)`` could not draw a time series in its own units. ``axis_scale=
'data'`` keeps the pipeline's own coordinates on both backends, static and
animated.

Everything here reads the coordinates back off the artists/traces the call
actually produced -- no mocks, no monkeypatching.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import pytest                                             # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402

import hypertools as hyp                                  # noqa: E402


@pytest.fixture
def series():
    """A 100-row (t, y) series with x = 0..99 -- the shape the tutorials
    hand-build as ``np.column_stack([day, price])``."""
    t = np.arange(100.0)
    return t, np.sin(t / 7.0) * 3.0 + 10.0


def _lines(fig):
    return [line for line in fig.axes[0].lines
            if getattr(line, '_hyp_forecast_role', None) is None]


def test_data_scale_draws_the_raw_x_and_y(series):
    t, y = series
    fig = hyp.plot(np.column_stack([t, y]), reduce=None, ndims=2,
                   axis_scale='data', antialias=False, show=False)
    line = _lines(fig)[0]
    assert np.array_equal(np.asarray(line.get_xdata()), t)
    assert np.allclose(np.asarray(line.get_ydata()), y)
    plt.close(fig)


def test_unit_scale_is_still_the_default_and_still_rescales(series):
    t, y = series
    fig = hyp.plot(np.column_stack([t, y]), reduce=None, ndims=2,
                   antialias=False, show=False)
    line = _lines(fig)[0]
    x = np.asarray(line.get_xdata())
    assert not np.array_equal(x, t)
    assert x.min() >= -1.0 - 1e-9 and x.max() <= 1.0 + 1e-9
    assert fig.axes[0].get_xlim() == (-1.1, 1.1)
    assert fig.axes[0].get_ylim() == (-1.1, 1.1)
    plt.close(fig)


def test_data_scale_draws_no_frame_square_and_keeps_the_axes(series):
    t, y = series
    unit = hyp.plot(np.column_stack([t, y]), reduce=None, ndims=2,
                    show=False)
    data = hyp.plot(np.column_stack([t, y]), reduce=None, ndims=2,
                    axis_scale='data', show=False)
    # the frame square is a Rectangle patch; 'data' draws none
    assert len(unit.axes[0].patches) == 1
    assert len(data.axes[0].patches) == 0
    # ...and the axes stay on, so the ticks read real values
    assert data.axes[0].get_xaxis().get_visible()
    assert data.axes[0].xaxis.get_ticklabels()
    assert not data.axes[0].spines['top'].get_visible()
    plt.close(unit)
    plt.close(data)


def test_data_scale_limits_contain_the_data_with_a_margin(series):
    t, y = series
    fig = hyp.plot(np.column_stack([t, y]), reduce=None, ndims=2,
                   axis_scale='data', show=False)
    lo, hi = fig.axes[0].get_xlim()
    assert lo < t.min() and hi > t.max()
    assert lo == pytest.approx(-4.95) and hi == pytest.approx(103.95)
    plt.close(fig)


def test_explicit_xlim_ylim_win(series):
    t, y = series
    fig = hyp.plot(np.column_stack([t, y]), reduce=None, ndims=2,
                   axis_scale='data', xlim=(10, 20), ylim=(0, 30),
                   show=False)
    assert fig.axes[0].get_xlim() == (10, 20)
    assert fig.axes[0].get_ylim() == (0, 30)
    plt.close(fig)


def test_xlim_with_unit_scale_is_refused_rather_than_ignored(series):
    t, y = series
    with pytest.raises(ValueError, match='axis_scale=.unit.'):
        hyp.plot(np.column_stack([t, y]), reduce=None, ndims=2,
                 xlim=(0, 1), show=False)


def test_a_bad_axis_scale_names_both_options(series):
    t, y = series
    with pytest.raises(ValueError, match="'unit'.*'data'"):
        hyp.plot(np.column_stack([t, y]), reduce=None, ndims=2,
                 axis_scale='raw', show=False)


def test_three_d_data_refuses_the_data_scale():
    x = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.raises(ValueError, match='1-D and 2-D plots only'):
        hyp.plot(x, reduce=None, ndims=3, axis_scale='data', show=False)


def test_data_scale_honours_a_caller_supplied_ax(series):
    t, y = series
    fig, ax = plt.subplots()
    hyp.plot(np.column_stack([t, y]), reduce=None, ndims=2,
             axis_scale='data', ax=ax, antialias=False, show=False)
    line = [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) is None][0]
    assert np.array_equal(np.asarray(line.get_xdata()), t)
    assert len(ax.patches) == 0
    plt.close(fig)


def test_dataframe_columns_name_the_axes_under_the_data_scale():
    df = pd.DataFrame({'year': np.arange(1900.0, 1950.0),
                       'degrees': np.linspace(10.0, 12.0, 50)})
    fig = hyp.plot(df, reduce=None, ndims=2, axis_scale='data', show=False)
    assert fig.axes[0].get_xlabel() == 'year'
    assert fig.axes[0].get_ylabel() == 'degrees'
    plt.close(fig)


def test_animated_data_scale_keeps_one_fixed_viewport(series):
    t, y = series
    anim = hyp.plot(np.column_stack([t[:30], y[:30]]), reduce=None, ndims=2,
                    axis_scale='data', animate=True, duration=1,
                    frame_rate=4, show=False)
    fig, ani = anim.figure, anim.animation
    ani._init_draw()
    seen = set()
    for frame in range(4):
        ani._func(frame, *ani._args)
        seen.add((fig.axes[0].get_xlim(), fig.axes[0].get_ylim()))
    assert len(seen) == 1
    (xlim, _), = seen
    assert xlim[0] < 0 and xlim[1] > 29
    plt.close(fig)


def test_animated_data_scale_frames_draw_raw_coordinates(series):
    t, y = series
    anim = hyp.plot(np.column_stack([t[:30], y[:30]]), reduce=None, ndims=2,
                    axis_scale='data', animate=True, duration=1,
                    frame_rate=4, antialias=False, show=False)
    fig, ani = anim.figure, anim.animation
    ani._init_draw()
    ani._func(3, *ani._args)
    drawn = np.asarray(fig.axes[0].lines[0].get_xdata())
    # an animation paces on its own frame grid, so the drawn vertices are
    # resampled -- but they are resampled in the DATA's units (0..29), not
    # rescaled into [-1, 1], which is what this pins
    assert len(drawn)
    assert drawn.min() >= 0.0 and drawn.max() <= 29.0
    assert np.all(np.diff(drawn) > 0)
    assert drawn[0] == pytest.approx(0.0)
    plt.close(fig)


def test_plotly_data_scale_draws_the_raw_x_and_no_square(series):
    pytest.importorskip('plotly')
    import plotly.io as pio
    pio.renderers.default = 'json'
    t, y = series
    fig = hyp.plot(np.column_stack([t, y]), reduce=None, ndims=2,
                   axis_scale='data', backend='plotly', antialias=False,
                   show=False)
    assert np.array_equal(np.asarray(fig.data[0].x, dtype=float), t)
    assert not (fig.layout.shapes or ())
    assert fig.layout.xaxis.visible is True
    lo, hi = fig.layout.xaxis.range
    assert lo < 0 and hi > 99


def test_plotly_unit_scale_is_unchanged(series):
    pytest.importorskip('plotly')
    import plotly.io as pio
    pio.renderers.default = 'json'
    t, y = series
    fig = hyp.plot(np.column_stack([t, y]), reduce=None, ndims=2,
                   backend='plotly', antialias=False, show=False)
    assert tuple(fig.layout.xaxis.range) == (-1.1, 1.1)
    assert len(fig.layout.shapes) == 1


def test_forecast_overlay_stays_in_data_coordinates():
    rng = np.random.default_rng(7)
    t = np.arange(40.0)
    y = np.cumsum(rng.standard_normal(40)) + 50.0
    fig = hyp.plot(np.column_stack([t, y]), reduce=None, ndims=2,
                   axis_scale='data', predict='Kalman', t=6,
                   antialias=False, show=False)
    forecast = [ln for ln in fig.axes[0].lines
                if getattr(ln, '_hyp_forecast_role', None) == 'static'][0]
    fx = np.asarray(forecast.get_xdata())
    # the seam row is the last observation, and the forecast continues past it
    assert fx[0] == pytest.approx(t[-1])
    assert fx[-1] > t[-1]
    # ...and the axis limits grew to contain it
    assert fig.axes[0].get_xlim()[1] > fx[-1]
    plt.close(fig)
