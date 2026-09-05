"""`truth=` -- the actual continuation, drawn beside the forecast (GH #285).

The tutorials' stock/projectile figures built three ``np.column_stack([day,
price])`` datasets with ``fmt=['-', '-o', '--x']`` to show train, held-out
and forecast. ``hyp.plot(train, predict=..., t=..., truth=held_out)`` is the
same figure in one call.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import pytest                                             # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402

import hypertools as hyp                                  # noqa: E402

T = 6


@pytest.fixture
def stock():
    """A 40-day price series, its 6-day held-out continuation, and the
    horizon -- the shape of tutorial cell 'stock 8'."""
    rng = np.random.default_rng(11)
    day = np.arange(40.0)
    price = np.cumsum(rng.standard_normal(40)) + 100.0
    train = np.column_stack([day, price])
    held = np.column_stack([np.arange(40.0, 40.0 + T),
                            price[-1] + np.cumsum(rng.standard_normal(T))])
    return train, held


def _by_role(fig, role):
    return [line for line in fig.axes[0].lines
            if getattr(line, '_hyp_forecast_role', None) == role]


def test_truth_is_drawn_role_tagged_and_carries_the_real_values(stock):
    train, held = stock
    fig = hyp.plot(train, reduce=None, ndims=2, axis_scale='data',
                   predict='Kalman', t=T, truth=held, antialias=False,
                   show=False)
    truths = _by_role(fig, 'truth')
    assert truths, 'no artist was tagged truth'
    line = truths[0]
    assert line._hyp_forecast_dataset == 0
    # the seam row (the last observation) is prepended, then the held-out
    # rows, verbatim
    assert np.allclose(np.asarray(line.get_xdata()),
                       np.r_[train[-1, 0], held[:, 0]])
    assert np.allclose(np.asarray(line.get_ydata()),
                       np.r_[train[-1, 1], held[:, 1]])
    plt.close(fig)


def test_truth_is_styled_distinctly_from_the_forecast(stock):
    train, held = stock
    fig = hyp.plot(train, reduce=None, ndims=2, axis_scale='data',
                   predict='Kalman', t=T, truth=held, antialias=False,
                   show=False)
    truth = _by_role(fig, 'truth')[0]
    forecast = _by_role(fig, 'static')[0]
    assert truth.get_alpha() in (None, 1.0)
    assert forecast.get_alpha() == pytest.approx(0.5)
    assert truth.get_linestyle() == '-'
    # markers land on the observations, on their own artist
    markers = _by_role(fig, 'truth')[1]
    assert markers.get_marker() == 'o'
    assert markers.get_linestyle() in ('None', 'none')
    plt.close(fig)


def test_truth_gets_one_legend_entry(stock):
    train, held = stock
    fig = hyp.plot(train, reduce=None, ndims=2, axis_scale='data',
                   predict='Kalman', t=T, truth=held, legend=True,
                   show=False)
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert labels.count('truth') == 1
    plt.close(fig)


def test_truth_without_predict_is_refused(stock):
    train, held = stock
    with pytest.raises(ValueError, match='no.*forecast was requested'):
        hyp.plot(train, reduce=None, ndims=2, truth=held, show=False)


def test_a_wrong_length_truth_names_the_horizon(stock):
    train, held = stock
    with pytest.raises(ValueError, match=f'exactly t={T} rows'):
        hyp.plot(train, reduce=None, ndims=2, predict='Kalman', t=T,
                 truth=held[:-1], show=False)


def test_a_wrong_width_truth_names_the_feature_count(stock):
    train, held = stock
    wide = np.column_stack([held, held[:, :1]])
    with pytest.raises(ValueError, match='3 column'):
        hyp.plot(train, reduce=None, ndims=2, predict='Kalman', t=T,
                 truth=wide, show=False)


def test_a_wrong_count_of_truths_names_the_trace_count(stock):
    train, held = stock
    with pytest.raises(ValueError, match='2 trace'):
        hyp.plot([train, train], reduce=None, ndims=2, predict='Kalman',
                 t=T, truth=[held], show=False)


def test_series_mode_truth_takes_values_only_and_gets_the_index():
    index = pd.Index(np.arange(40.0), name='day')
    rng = np.random.default_rng(12)
    price = np.cumsum(rng.standard_normal(40)) + 100.0
    df = pd.DataFrame({'price': price}, index=index)
    held = price[-1] + np.cumsum(rng.standard_normal(T))
    fig = hyp.plot(df, reduce=None, ndims=1, predict='Kalman', t=T,
                   truth=held, antialias=False, show=False)
    line = _by_role(fig, 'truth')[0]
    assert np.array_equal(np.asarray(line.get_xdata()),
                          np.arange(39.0, 39.0 + T + 1))
    assert np.allclose(np.asarray(line.get_ydata()), np.r_[price[-1], held])
    plt.close(fig)


def test_truth_is_drawn_in_full_during_an_animation(stock):
    train, held = stock
    anim = hyp.plot(train, reduce=None, ndims=2, axis_scale='data',
                    predict='Kalman', t=T, truth=held, animate=True,
                    duration=1, frame_rate=4, antialias=False, show=False)
    fig, ani = anim.figure, anim.animation
    ani._init_draw()
    ani._func(1, *ani._args)
    line = _by_role(fig, 'truth')[0]
    assert np.allclose(np.asarray(line.get_ydata()),
                       np.r_[train[-1, 1], held[:, 1]])
    plt.close(fig)


def test_truth_is_folded_into_the_unit_box(stock):
    train, held = stock
    fig = hyp.plot(train, reduce=None, ndims=2, predict='Kalman', t=T,
                   truth=held, antialias=False, show=False)
    line = _by_role(fig, 'truth')[0]
    pts = np.column_stack([np.asarray(line.get_xdata()),
                           np.asarray(line.get_ydata())])
    assert pts.min() >= -1.0 - 1e-9 and pts.max() <= 1.0 + 1e-9
    plt.close(fig)


def test_plotly_draws_a_role_tagged_truth_trace(stock):
    pytest.importorskip('plotly')
    import plotly.io as pio
    pio.renderers.default = 'json'
    train, held = stock
    fig = hyp.plot(train, reduce=None, ndims=2, axis_scale='data',
                   predict='Kalman', t=T, truth=held, legend=True,
                   backend='plotly', antialias=False, show=False)
    truths = [trace for trace in fig.data
              if (trace.meta or {}).get('hyp_forecast_role') == 'truth']
    assert len(truths) == 1
    trace = truths[0]
    assert trace.name == 'truth'
    assert trace.showlegend is True
    assert np.allclose(np.asarray(trace.x, dtype=float),
                       np.r_[train[-1, 0], held[:, 0]])
    assert np.allclose(np.asarray(trace.y, dtype=float),
                       np.r_[train[-1, 1], held[:, 1]])


def test_a_hierarchy_refuses_truth_rather_than_skipping_its_means():
    """A hierarchy draws derived per-level MEAN traces too, and a mean has
    no observed continuation -- so `truth=` is refused instead of silently
    covering only the leaves."""
    index = pd.MultiIndex.from_product(
        [['a', 'b'], np.arange(10.0)], names=['group', 't'])
    df = pd.DataFrame({'x': np.arange(20.0), 'y': np.arange(20.0) * 2},
                      index=index)
    with pytest.raises(ValueError, match='hierarchical'):
        hyp.plot(df, ndims=2, reduce=None, predict='Kalman', t=2,
                 truth=np.zeros((2, 2)), show=False)


def test_a_static_3d_truth_overlay_does_not_break_tight_layout():
    """An unclipped 3-D line's `get_window_extent()` differs from a clipped
    one, so unclipping on the STATIC path makes `plt.tight_layout()` warn
    and visibly resize the cube (the trap `matplotlib_backend.plot3D`'s NOTE
    records). Truth artists are unclipped only for animations."""
    import warnings
    steps = np.arange(50.0)
    arc = pd.DataFrame({'x': steps * 0.5, 'y': steps * 0.3,
                        'z': -0.05 * (steps - 25) ** 2 + 30})
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        fig = hyp.plot([arc.iloc[:30]], reduce=None, predict='Kalman',
                       t=20, truth=[arc.iloc[30:50]], legend=True,
                       fmt='-o', markersize=3, linewidth=2, show=False)
    truth = _by_role(fig, 'truth')[0]
    assert truth.get_clip_on() is True
    plt.close(fig)


def test_an_animated_3d_truth_overlay_is_unclipped():
    steps = np.arange(50.0)
    arc = pd.DataFrame({'x': steps * 0.5, 'y': steps * 0.3,
                        'z': -0.05 * (steps - 25) ** 2 + 30})
    anim = hyp.plot([arc.iloc[:30]], reduce=None, predict='Kalman', t=20,
                    truth=[arc.iloc[30:50]], animate=True, duration=1,
                    frame_rate=4, show=False)
    fig = anim.figure
    truth = _by_role(fig, 'truth')[0]
    assert truth.get_clip_on() is False
    plt.close(fig)
