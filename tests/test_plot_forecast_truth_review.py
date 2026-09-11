"""Forecast / truth= / time-axis findings from the 1.1 release review.

Each test pins one reproduced finding against real rendered artists
(matplotlib ``Line2D`` colours, linestyles and data; plotly trace
properties), on both backends where the path is shared.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt                           # noqa: E402
import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import pytest                                             # noqa: E402
import seaborn as sns                                     # noqa: E402
from matplotlib.colors import to_hex, to_rgb              # noqa: E402
from matplotlib.dates import date2num                     # noqa: E402

import hypertools as hyp                                  # noqa: E402


def _walks(n=3, rows=20, dims=3, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, dims)).cumsum(0) + 10 * i
            for i in range(n)]


def _role(ax, role):
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) == role]


def _ply_role(fig, role):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') == role]


def _ply_rgb(color):
    """An ``rgb(...)``/``rgba(...)`` plotly colour string as a 0-1 RGB."""
    body = color[color.index('(') + 1:color.rindex(')')]
    vals = [float(v) for v in body.split(',')[:3]]
    return tuple(round(v / 255.0, 3) for v in vals)


# --- F1: a marker+line fmt draws each dataset as TWO artists ----------------

@pytest.mark.parametrize('fmt, dash', [('o-', '-'), ('s--', '--'),
                                       (['o-', '-', 'x:'], None)])
def test_split_fmt_forecasts_and_truths_take_their_own_datasets_style(
        fmt, dash):
    """``fmt='o-'`` draws a smoothed line plus a markers-only artist per
    dataset; the forecast and truth overlays indexed ONE artist per
    dataset, so three datasets' forecasts came out red, red, green (the
    marker artist of dataset 0 was read as dataset 1's line)."""
    data = _walks()
    truth = [d[-4:] + 1.0 for d in data]
    fig = hyp.plot(data, fmt=fmt, predict='Kalman', t=4, truth=truth,
                   show=False)
    ax = fig.axes[0]
    palette = [to_hex(c) for c in sns.color_palette('hls', 3)]
    fmts = fmt if isinstance(fmt, list) else [fmt] * 3
    want_dash = [dash] if dash else None
    fcs = sorted(_role(ax, 'static'), key=lambda a: a._hyp_forecast_dataset)
    assert [a._hyp_forecast_dataset for a in fcs] == [0, 1, 2]
    assert [to_hex(a.get_color()) for a in fcs] == palette
    expected_ls = [f.lstrip('osx') or '-' for f in fmts]
    if want_dash:
        expected_ls = want_dash * 3
    assert [a.get_linestyle() for a in fcs] == expected_ls
    # every truth artist (curve and its markers) wears its OWN dataset's
    # colour
    for tr in _role(ax, 'truth'):
        assert to_hex(tr.get_color()) == palette[tr._hyp_forecast_dataset]
    plt.close(fig)


def test_split_fmt_forecast_under_hue_runs_continues_the_last_run():
    """Under ``hue=`` runs a forecast continues the run holding the last
    observation -- with ``fmt='o-'`` as with ``fmt='-'``."""
    data = _walks(n=1)[0]
    hue = ['a'] * 10 + ['b'] * 10
    colours = {}
    for fmt in ('-', 'o-'):
        fig = hyp.plot(data, fmt=fmt, hue=hue, predict='Kalman', t=4,
                       show=False)
        ax = fig.axes[0]
        (fc,) = _role(ax, 'static')
        drawn = [ln for ln in ax.lines
                 if getattr(ln, '_hyp_forecast_role', None) is None
                 and ln.get_linestyle() not in ('None', 'none', '')]
        # the LAST drawn run holds the final observation
        assert to_hex(fc.get_color()) == to_hex(drawn[-1].get_color())
        colours[fmt] = to_hex(fc.get_color())
        plt.close(fig)
    assert colours['o-'] == colours['-']


def test_split_fmt_forecast_colours_match_on_plotly():
    data = _walks()
    fig = hyp.plot(data, fmt='o-', predict='Kalman', t=4, backend='plotly',
                   show=False)
    palette = [tuple(round(v, 3) for v in to_rgb(c))
               for c in sns.color_palette('hls', 3)]
    fcs = sorted(_ply_role(fig, 'static'),
                 key=lambda tr: tr.meta['hyp_dataset'])
    got = [_ply_rgb(tr.line.color) for tr in fcs]
    assert len(got) == 3
    for g, p in zip(got, palette):
        assert np.allclose(g, p, atol=0.01)


# --- F3: ndims=1 truth= is one column of VALUES per trace -------------------

@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_series_mode_two_column_truth_for_one_trace_raises(backend):
    """In series mode a trace is one plotted column (x is the index); a
    2-column truth= for it matched the trace's internal (x, value) width
    and its first column was drawn as x -- the date axis ran 1970..2020."""
    idx = pd.date_range('2020-01-01', periods=40)
    full = pd.DataFrame(np.sin(np.arange(40) / 4)[:, None] * [[1, 2]],
                        index=idx, columns=['a', 'b'])
    train, test = full.iloc[:30], full.iloc[30:]
    with pytest.raises(ValueError, match='one column of values'):
        hyp.plot(train, ndims=1, predict='Kalman', t=10, truth=test,
                 backend=backend, show=False)
    with pytest.raises(ValueError, match='one column of values'):
        hyp.plot(train, ndims=1, predict='Kalman', t=10,
                 truth=test.values, backend=backend, show=False)
    plt.close('all')


def test_series_mode_one_column_truth_still_lands_on_the_index():
    idx = pd.date_range('2020-01-01', periods=40)
    full = pd.DataFrame({'a': np.sin(np.arange(40) / 4)}, index=idx)
    train, test = full.iloc[:30], full.iloc[30:]
    fig = hyp.plot(train, ndims=1, reduce=None, predict='Kalman', t=10,
                   truth=test, antialias=False, show=False)
    (curve, markers) = _role(fig.axes[0], 'truth')
    want = date2num(pd.date_range('2020-01-30', periods=11).to_pydatetime())
    assert np.allclose(np.asarray(markers.get_xdata(), float), want)
    plt.close(fig)


# --- 1-D data WITHOUT ndims=1: x is the row index, in rows ------------------

def _series_1d(n=40, seed=0):
    return np.cumsum(np.random.default_rng(seed).standard_normal(n))


def _mpl_x_spans(ax):
    out = {}
    for ln in ax.lines:
        role = getattr(ln, '_hyp_forecast_role', None) or 'data'
        x = np.asarray(ln.get_xdata(), float)
        out.setdefault(role, []).append((x.min(), x.max()))
    return out


def _ply_x_spans(fig):
    out = {}
    for tr in fig.data:
        if tr.x is None or not len(tr.x):
            continue
        role = (tr.meta or {}).get('hyp_forecast_role') or 'data'
        x = np.asarray(tr.x, float)
        out.setdefault(role, []).append((x.min(), x.max()))
    return out


@pytest.mark.parametrize('antialias', [True, False])
@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_1d_trace_forecast_and_truth_share_row_units(backend, antialias):
    """A 40-row 1-D array drew its (antialiased) line over x 0..936 -- the
    VERTEX index -- while the forecast continued in rows (936..941), so the
    forecast was squashed 24x at the far end."""
    y = _series_1d()
    out = hyp.plot(y, predict='Kalman', t=5, truth=np.arange(5.0),
                   antialias=antialias, backend=backend, show=False)
    spans = (_mpl_x_spans(out.axes[0]) if backend == 'matplotlib'
             else _ply_x_spans(out))
    assert spans['data'] == [(0.0, 39.0)]
    assert spans['static'] == [(39.0, 44.0)]
    assert all(s == (39.0, 44.0) for s in spans['truth'])
    plt.close('all')


def test_1d_split_fmt_markers_sit_on_the_rows_of_the_line():
    y = _series_1d()
    fig = hyp.plot(y, fmt='o-', show=False)
    line, markers = fig.axes[0].lines
    assert np.asarray(line.get_xdata(), float).max() == 39.0
    assert np.array_equal(np.asarray(markers.get_xdata(), float),
                          np.arange(40.0))
    # every marker sits ON the smoothed line at its row
    lx = np.asarray(line.get_xdata(), float)
    ly = np.asarray(line.get_ydata(), float)
    assert np.allclose(np.interp(np.arange(40.0), lx, ly),
                       np.asarray(markers.get_ydata(), float))
    plt.close(fig)


def test_1d_continuous_hue_line_spans_the_rows():
    y = _series_1d()
    fig = hyp.plot(y, hue=np.linspace(0, 1, 40), fmt='-', show=False)
    from matplotlib.collections import LineCollection
    (coll,) = [c for c in fig.axes[0].collections
               if isinstance(c, LineCollection)]
    xs = np.concatenate([seg[:, 0] for seg in coll.get_segments()])
    assert xs.min() == 0.0 and xs.max() == 39.0
    plt.close(fig)


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_1d_axis_scale_data_puts_the_value_range_on_y_not_x(backend):
    y = _series_1d()
    out = hyp.plot(y, axis_scale='data', predict='Kalman', t=5,
                   backend=backend, show=False)
    if backend == 'matplotlib':
        ax = out.axes[0]
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
    else:
        xr, yr = out.layout.xaxis.range, out.layout.yaxis.range
        # plotly autoranges x (rows); y is pinned to the value range
        xlo, xhi = (xr if xr is not None else (0.0, 44.0))
        ylo, yhi = yr
    assert xlo <= 0.0 and xhi >= 44.0
    assert ylo <= y.min() and yhi >= y.max()
    assert yhi - ylo < 2 * (y.max() - y.min())
    plt.close('all')
