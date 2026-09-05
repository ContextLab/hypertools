"""`ndims=1` -- an honest time-series mode (GH #285).

Before 1.2, ``hyp.plot(np.array([5, 7, 6, 9, 8.]), reduce=None, ndims=1)``
drew x = 0..N with the values rescaled into ``[-1, 1]``, discarded a
DataFrame's index, and refused a 2-column frame ("static plots support at
most 1"). It now draws one line per COLUMN against the row index, in the
data's own units.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import pytest                                             # noqa: E402
import matplotlib.dates as mdates                         # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402

import hypertools as hyp                                  # noqa: E402


def _data_lines(fig):
    return [line for line in fig.axes[0].lines
            if getattr(line, '_hyp_forecast_role', None) is None]


def test_a_bare_array_draws_row_positions_and_raw_values():
    fig = hyp.plot(np.array([5.0, 7, 6, 9, 8]), reduce=None, ndims=1,
                   antialias=False, show=False)
    (line,) = _data_lines(fig)
    assert np.array_equal(np.asarray(line.get_xdata()),
                          np.array([0.0, 1, 2, 3, 4]))
    assert np.array_equal(np.asarray(line.get_ydata()),
                          np.array([5.0, 7, 6, 9, 8]))
    plt.close(fig)


def test_a_dataframe_index_becomes_the_x_axis():
    index = pd.Index([10.0, 20.0, 30.0, 40.0], name='t')
    df = pd.DataFrame({'y': [1.0, 4.0, 9.0, 16.0]}, index=index)
    fig = hyp.plot(df, reduce=None, ndims=1, antialias=False, show=False)
    (line,) = _data_lines(fig)
    assert np.array_equal(np.asarray(line.get_xdata()), np.asarray(index))
    assert fig.axes[0].get_xlabel() == 't'
    assert fig.axes[0].get_ylabel() == 'y'
    plt.close(fig)


def test_two_columns_draw_two_lines_named_by_column():
    df = pd.DataFrame({'a': [5.0, 7, 6, 9, 8], 'b': [1.0, 2, 3, 4, 5]})
    fig = hyp.plot(df, reduce=None, ndims=1, legend=True, antialias=False,
                   show=False)
    lines = _data_lines(fig)
    assert len(lines) == 2
    assert [line.get_label() for line in lines] == ['a', 'b']
    for line, column in zip(lines, ('a', 'b')):
        assert np.array_equal(np.asarray(line.get_xdata()),
                              np.arange(5.0))
        assert np.array_equal(np.asarray(line.get_ydata()),
                              df[column].to_numpy())
    # ...and they are drawn in different colours
    assert lines[0].get_color() != lines[1].get_color()
    plt.close(fig)


def test_two_columns_used_to_raise_and_no_longer_do():
    df = pd.DataFrame({'a': [5.0, 7, 6], 'b': [1.0, 2, 3]})
    fig = hyp.plot(df, reduce=None, ndims=1, show=False)
    assert len(_data_lines(fig)) == 2
    plt.close(fig)


def test_names_override_the_column_labels():
    df = pd.DataFrame({'a': [5.0, 7, 6], 'b': [1.0, 2, 3]})
    fig = hyp.plot(df, reduce=None, ndims=1, names=['left', 'right'],
                   show=False)
    assert [line.get_label() for line in _data_lines(fig)] == \
        ['left', 'right']
    plt.close(fig)


def test_a_datetime_index_gives_real_dates_on_the_axis():
    index = pd.date_range('2024-01-01', periods=20, freq='D')
    df = pd.DataFrame({'temp': np.arange(20.0)}, index=index)
    fig = hyp.plot(df, reduce=None, ndims=1, antialias=False, show=False)
    (line,) = _data_lines(fig)
    x = np.asarray(line.get_xdata())
    assert mdates.num2date(x[0]).date() == index[0].date()
    assert mdates.num2date(x[-1]).date() == index[-1].date()
    # a real date axis: matplotlib's own date locator/formatter pair, so
    # the ticks read '2024-01-05' rather than '19727.0'
    assert isinstance(fig.axes[0].xaxis.get_major_locator(),
                      mdates.AutoDateLocator)
    assert isinstance(fig.axes[0].xaxis.get_major_formatter(),
                      mdates.AutoDateFormatter)
    plt.close(fig)


def test_reduce_gives_the_first_component_over_the_index():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((30, 5))
    fig = hyp.plot(data, ndims=1, reduce='PCA', antialias=False, show=False)
    (line,) = _data_lines(fig)
    assert np.array_equal(np.asarray(line.get_xdata()), np.arange(30.0))
    assert len(np.asarray(line.get_ydata())) == 30
    plt.close(fig)


def test_series_mode_defaults_to_the_data_scale_and_can_be_overridden():
    values = np.array([5.0, 7, 6, 9, 8])
    default = hyp.plot(values, reduce=None, ndims=1, antialias=False,
                       show=False)
    assert np.array_equal(
        np.asarray(_data_lines(default)[0].get_ydata()), values)
    plt.close(default)

    unit = hyp.plot(values, reduce=None, ndims=1, axis_scale='unit',
                    antialias=False, show=False)
    y = np.asarray(_data_lines(unit)[0].get_ydata())
    assert y.min() >= -1.0 - 1e-9 and y.max() <= 1.0 + 1e-9
    assert unit.axes[0].get_ylim() == (-1.1, 1.1)
    plt.close(unit)


def test_several_datasets_each_keep_their_own_index():
    a = pd.DataFrame({'v': [1.0, 2, 3]}, index=pd.Index([0.0, 1, 2],
                                                        name='t'))
    b = pd.DataFrame({'v': [4.0, 5, 6]}, index=pd.Index([5.0, 6, 7],
                                                        name='t'))
    fig = hyp.plot([a, b], reduce=None, ndims=1, antialias=False,
                   show=False)
    lines = _data_lines(fig)
    assert len(lines) == 2
    assert np.array_equal(np.asarray(lines[0].get_xdata()),
                          np.array([0.0, 1, 2]))
    assert np.array_equal(np.asarray(lines[1].get_xdata()),
                          np.array([5.0, 6, 7]))
    plt.close(fig)


def test_an_animation_reveals_along_x():
    df = pd.DataFrame({'y': np.arange(30.0)},
                      index=pd.Index(np.arange(30.0), name='t'))
    anim = hyp.plot(df, reduce=None, ndims=1, animate=True, duration=1,
                    frame_rate=4, antialias=False, show=False)
    fig, ani = anim.figure, anim.animation
    ani._init_draw()
    reach = []
    for frame in range(4):
        ani._func(frame, *ani._args)
        x = np.asarray(fig.axes[0].lines[0].get_xdata())
        reach.append(float(x[-1]) if len(x) else -1.0)
    assert reach == sorted(reach)
    assert reach[-1] > reach[0]
    plt.close(fig)


def test_a_forecast_continues_the_index_exactly():
    index = pd.Index(np.arange(0.0, 40.0), name='day')
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {'price': np.cumsum(rng.standard_normal(40)) + 100.0}, index=index)
    fig = hyp.plot(df, reduce=None, ndims=1, predict='Kalman', t=6,
                   antialias=False, show=False)
    forecast = [ln for ln in fig.axes[0].lines
                if getattr(ln, '_hyp_forecast_role', None) == 'static'][0]
    x = np.asarray(forecast.get_xdata())
    # seam row + t forecast steps, one index step apart, exactly
    assert np.array_equal(x, np.arange(39.0, 46.0))
    plt.close(fig)


def test_multi_column_series_refuses_hue_rather_than_mis_assigning():
    df = pd.DataFrame({'a': [1.0, 2, 3], 'b': [4.0, 5, 6]})
    with pytest.raises(ValueError, match='hue= is not supported'):
        hyp.plot(df, reduce=None, ndims=1, hue=[0, 1, 0], show=False)


def test_plotly_series_mode_draws_one_trace_per_column():
    pytest.importorskip('plotly')
    import plotly.io as pio
    pio.renderers.default = 'json'
    df = pd.DataFrame({'a': np.arange(10.0), 'b': np.arange(10.0) * 2},
                      index=pd.Index(np.arange(10.0), name='t'))
    fig = hyp.plot(df, reduce=None, ndims=1, legend=True, backend='plotly',
                   antialias=False, show=False)
    assert len(fig.data) == 2
    assert [trace.name for trace in fig.data] == ['a', 'b']
    assert np.array_equal(np.asarray(fig.data[0].x, dtype=float),
                          np.arange(10.0))
    assert np.array_equal(np.asarray(fig.data[1].y, dtype=float),
                          np.arange(10.0) * 2)
    assert not (fig.layout.shapes or ())


def test_plotly_datetime_series_uses_a_real_date_axis():
    pytest.importorskip('plotly')
    import plotly.io as pio
    pio.renderers.default = 'json'
    index = pd.date_range('2024-01-01', periods=10, freq='D')
    df = pd.DataFrame({'a': np.arange(10.0)}, index=index)
    fig = hyp.plot(df, reduce=None, ndims=1, backend='plotly',
                   antialias=False, show=False)
    assert fig.layout.xaxis.type == 'date'
    x = np.asarray(fig.data[0].x, dtype=float)
    assert pd.to_datetime(x[0], unit='ms') == index[0]
    assert pd.to_datetime(x[-1], unit='ms') == index[-1]


def test_a_timezone_aware_index_reaches_both_backends():
    """A tz-aware `DatetimeIndex` cannot be cast to a naive dtype, which is
    how plotly's epoch-millisecond encoding is built -- so it is converted
    to UTC first rather than raising."""
    pytest.importorskip('plotly')
    import plotly.io as pio
    pio.renderers.default = 'json'
    index = pd.date_range('2024-01-01', periods=8, freq='D',
                          tz='US/Eastern')
    df = pd.DataFrame({'v': np.arange(8.0)}, index=index)

    fig = hyp.plot(df, reduce=None, ndims=1, antialias=False, show=False)
    x = np.asarray(_data_lines(fig)[0].get_xdata())
    assert mdates.num2date(x[0]).date() == index[0].date()
    plt.close(fig)

    pfig = hyp.plot(df, reduce=None, ndims=1, backend='plotly',
                    antialias=False, show=False)
    px = np.asarray(pfig.data[0].x, dtype=float)
    assert pd.to_datetime(px[0], unit='ms', utc=True) == \
        index[0].tz_convert('UTC')


def test_a_pandas_series_is_named_by_its_own_name():
    s = pd.Series([1.0, 2, 3], name='sig',
                  index=pd.Index([0.0, 1, 2], name='t'))
    fig = hyp.plot(s, reduce=None, ndims=1, legend=True, antialias=False,
                   show=False)
    (line,) = _data_lines(fig)
    assert line.get_label() == 'sig'
    assert fig.axes[0].get_xlabel() == 't'
    plt.close(fig)


def test_morph_is_still_refused_for_a_series():
    data = [np.random.default_rng(0).normal(size=(20, 1)) for _ in range(3)]
    with pytest.raises(NotImplementedError, match='2-D or 3-D'):
        hyp.plot(data, '.', animate='morph', ndims=1, reduce=None,
                 duration=1, frame_rate=5, show=False)


def test_a_per_dataset_style_list_follows_its_columns():
    frames = [pd.DataFrame({'p': [1.0, 2, 3], 'q': [2.0, 3, 4]}) + k
              for k in range(3)]
    fig = hyp.plot(frames, reduce=None, ndims=1, color=['r', 'g', 'b'],
                   antialias=False, show=False)
    assert [line.get_color() for line in _data_lines(fig)] == \
        ['r', 'r', 'g', 'g', 'b', 'b']
    plt.close(fig)


def test_a_bare_rgb_tuple_stays_one_colour():
    """Three datasets and a (r, g, b) triple: the triple is ONE colour, not
    three per-dataset values."""
    frames = [pd.DataFrame({'p': [1.0, 2, 3], 'q': [2.0, 3, 4]}) + k
              for k in range(3)]
    fig = hyp.plot(frames, reduce=None, ndims=1, color=(1.0, 0.0, 0.0),
                   antialias=False, show=False)
    for line in _data_lines(fig):
        assert tuple(np.round(np.asarray(line.get_color(), dtype=float), 3)) \
            == (1.0, 0.0, 0.0)
    plt.close(fig)
