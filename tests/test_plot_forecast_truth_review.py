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


# --- two-column data, animated, without ndims=2 ------------------------------

@pytest.mark.parametrize('extra', [{}, {'forecast_trail': 2},
                                   {'animate': 'window'}])
def test_two_column_animated_forecast_draws_on_a_2d_axes(extra):
    """2-column data under the default ndims draws a 2-D axes, but the live
    forecast artists were built for the REQUESTED 3 dims: ``ax.plot([], [],
    [])`` on a 2-D axes returns two artists and the call crashed with 'too
    many values to unpack'."""
    rng = np.random.default_rng(0)
    data = [np.cumsum(rng.standard_normal((30, 2)), 0) for _ in range(2)]
    kw = dict(animate=True, duration=1, frame_rate=8)
    kw.update(extra)
    anim = hyp.plot(data, predict='Kalman', t=4, show=False, **kw)
    ax = anim.figure.axes[0]
    assert getattr(ax, 'name', None) != '3d'
    anim.draw_frame(7)
    live = _role(ax, 'live')
    assert len(live) == 2
    for art in live:
        assert art.get_visible()
        xy = np.asarray(art.get_xydata())
        assert xy.shape[1] == 2 and len(xy) >= 2 and np.isfinite(xy).all()
    if extra.get('forecast_trail'):
        assert len(_role(ax, 'trail')) == 2 * extra['forecast_trail']
    plt.close(anim.figure)


# --- marker-only categorical regrouping never anchors a forecast ------------

def _equal_count_categories():
    rng = np.random.default_rng(3)
    a, b = (np.cumsum(rng.standard_normal((30, 3)), 0) for _ in range(2))
    # both datasets END in 'x', and 2 categories == 2 datasets
    hue = [['x'] * 10 + ['y'] * 10 + ['x'] * 10, ['y'] * 15 + ['x'] * 15]
    return [a, b], hue


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
@pytest.mark.parametrize('animate', [False, True])
@pytest.mark.parametrize('predict', ['Kalman', ['Kalman', 'AutoRegressor']])
def test_marker_only_categorical_regrouping_refuses_even_at_equal_counts(
        backend, animate, predict):
    """Marker-only categorical regrouping groups GLOBALLY by category, so
    its traces are categories, not datasets. With 2 categories x 2
    datasets the count check passed and dataset 1's forecast was drawn in
    category 'y's colour (though it ends in 'x'), with no warning and
    ``drawn=True``. It must refuse and say so, as it does at 3 x 2."""
    data, hue = _equal_count_categories()
    kw = dict(animate=True, duration=1, frame_rate=4) if animate else {}
    with pytest.warns(UserWarning, match='could not be matched'):
        bundle = hyp.plot(data, hue=hue, fmt='o', predict=predict, t=4,
                          return_model=True, backend=backend, show=False,
                          **kw)
    fig = bundle['fig']
    assert bundle['predict']['drawn'] is False
    assert 'regrouped' in bundle['predict']['draw_reason']
    if backend == 'plotly':
        assert not [tr for tr in fig.data
                    if (tr.meta or {}).get('hyp_forecast_role')]
    else:
        f = fig.figure if hasattr(fig, 'figure') else fig
        assert not [ln for ln in f.axes[0].lines
                    if getattr(ln, '_hyp_forecast_role', None)]
    plt.close('all')


# --- legend_colors= plain list with predict= / truth= ------------------------

@pytest.mark.parametrize('extra', [
    {}, {'animate': True, 'duration': 1, 'frame_rate': 4}, {'truth': 'yes'}])
def test_legend_colors_one_per_data_entry_still_works_with_overlays(extra):
    """``legend_colors=['r', 'b']`` recolours the two data entries; the
    forecast (and truth) entries predict= adds keep their own glyphs.
    It raised 'legend has 3' after drawing (master accepted it)."""
    data = _walks(n=2)
    extra = dict(extra)
    if extra.pop('truth', None):
        extra['truth'] = [d[-5:] for d in data]
    out = hyp.plot(data, legend=['A', 'B'], legend_colors=['r', 'b'],
                   predict='Kalman', t=5, show=False, **extra)
    fig = out.figure if hasattr(out, 'figure') else out
    lg = fig.axes[0].get_legend()
    got = {t.get_text(): to_hex(h.get_color())
           for t, h in zip(lg.get_texts(), lg.legend_handles)}
    assert got['A'] == '#ff0000' and got['B'] == '#0000ff'
    from hypertools.plot.forecast import FORECAST_LEGEND_COLOR
    assert got['Kalman'] == to_hex(FORECAST_LEGEND_COLOR)
    plt.close(fig)


def test_legend_colors_one_per_entry_recolours_the_overlay_entries_too():
    data = _walks(n=2)
    fig = hyp.plot(data, legend=['A', 'B'],
                   legend_colors=['r', 'b', 'g'], predict='Kalman', t=5,
                   show=False)
    lg = fig.axes[0].get_legend()
    assert [to_hex(h.get_color()) for h in lg.legend_handles] == \
        ['#ff0000', '#0000ff', '#008000']
    plt.close(fig)


def test_legend_colors_wrong_count_raises_and_leaves_no_figure_open():
    data = _walks(n=2)
    plt.close('all')
    with pytest.raises(ValueError, match='legend_colors has 4'):
        hyp.plot(data, legend=['A', 'B'], legend_colors=['r', 'b', 'g', 'k'],
                 predict='Kalman', t=5, show=False)
    assert plt.get_fignums() == []


# --- plotly date axes are time-zone invariant ---------------------------------

def _hourly(n=24):
    return pd.DataFrame({'signal': np.sin(np.arange(float(n)) / 4)},
                        index=pd.date_range('2026-01-01', periods=n,
                                            freq='h'))


def _as_dates(values):
    return pd.to_datetime([v for v in values if v is not None])


def test_plotly_date_x_values_are_the_true_dates_as_naive_strings():
    """plotly.js draws a NUMERIC date in the viewer's local time zone, so
    the epoch-ms x hypertools handed it put a series that starts
    2026-01-01 00:00 at 19:00 Dec 31 in New York. Every date x -- data,
    forecast, truth, and the axis range -- is a naive date string equal to
    the input's own dates."""
    data = _hourly()
    held = pd.DataFrame({'signal': np.zeros(4)},
                        index=pd.date_range('2026-01-02', periods=4,
                                            freq='h'))
    fig = hyp.plot(data, backend='plotly', ndims=1, reduce=None,
                   predict='Kalman', t=4, truth=held, antialias=False,
                   show=False)
    assert fig.layout.xaxis.type == 'date'
    (obs,) = [tr for tr in fig.data
              if (tr.meta or {}).get('hyp_trace_index') == 0]
    assert all(isinstance(v, str) for v in obs.x)
    assert list(_as_dates(obs.x)) == list(data.index)
    (fc,) = _ply_role(fig, 'static')
    assert list(_as_dates(fc.x)) == list(
        pd.date_range('2026-01-01 23:00', periods=5, freq='h'))
    (tr,) = _ply_role(fig, 'truth')
    assert list(_as_dates(tr.x)) == list(
        pd.date_range('2026-01-01 23:00', periods=5, freq='h'))
    lo, hi = _as_dates(fig.layout.xaxis.range)
    assert lo < data.index[0] and hi > pd.Timestamp('2026-01-02 03:00')


def test_plotly_animated_date_frames_carry_true_dates():
    data = _hourly()
    fig = hyp.plot(data, backend='plotly', ndims=1, reduce=None,
                   animate=True, duration=1, frame_rate=6, antialias=False,
                   show=False)
    xs = [v for frame in fig.frames for trace in frame.data
          if trace.x is not None for v in trace.x if v is not None]
    assert xs and all(isinstance(v, str) for v in xs)
    stamps = _as_dates(xs)
    assert stamps.min() >= data.index[0] and stamps.max() <= data.index[-1]


_TZ_RENDER = r'''
import sys, warnings
import numpy as np, pandas as pd
import hypertools as hyp
warnings.simplefilter('ignore')
data = pd.DataFrame({'signal': np.sin(np.arange(24.0) / 4)},
                    index=pd.date_range('2026-01-01', periods=24, freq='h'))
fig = hyp.plot(data, backend='plotly', ndims=1, reduce=None,
               predict='Kalman', t=4, show=False)
fig.write_image(sys.argv[1], width=600, height=360)
'''


def test_plotly_date_figure_renders_identically_in_every_time_zone(tmp_path):
    """The real observable: the same figure rendered by Chrome (kaleido)
    under TZ=UTC and TZ=America/New_York. With epoch-ms dates the two
    renders differed by thousands of pixels (the whole trace shifted five
    hours); with date strings they are pixel-identical."""
    pytest.importorskip('kaleido')
    import os
    import subprocess
    import sys
    from PIL import Image
    from hypertools._shared.lazy_import import ensure_kaleido_chrome
    ensure_kaleido_chrome()
    script = tmp_path / 'render.py'
    script.write_text(_TZ_RENDER)
    pixels = []
    for tz in ('UTC', 'America/New_York'):
        out = tmp_path / f'{tz.replace("/", "_")}.png'
        env = dict(os.environ, TZ=tz)
        subprocess.run([sys.executable, str(script), str(out)], check=True,
                       env=env, timeout=300)
        pixels.append(np.asarray(Image.open(out).convert('RGB')))
    assert pixels[0].shape == pixels[1].shape
    assert int((pixels[0] != pixels[1]).any(axis=2).sum()) == 0


# --- xlim=(None, date) on a date axis -----------------------------------------

@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_open_ended_date_xlim_takes_the_data_bound(backend):
    """``xlim=(None, '2020-01-10')`` crashed 'Axis limits cannot be NaN or
    Inf' (NaT from the None); the open side now takes the data bound."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(30, 2)).cumsum(0),
                      index=pd.date_range('2020-01-01', periods=30),
                      columns=['a', 'b'])
    out = hyp.plot(df, ndims=1, xlim=(None, '2020-01-10'), backend=backend,
                   show=False)
    if backend == 'matplotlib':
        from matplotlib.dates import num2date
        lo, hi = out.axes[0].get_xlim()
        lo = pd.Timestamp(num2date(lo)).tz_localize(None)
        hi = pd.Timestamp(num2date(hi)).tz_localize(None)
        plt.close(out)
    else:
        lo, hi = _as_dates(out.layout.xaxis.range)
    assert hi == pd.Timestamp('2020-01-10')
    assert lo <= pd.Timestamp('2020-01-01')
    assert lo > pd.Timestamp('2019-12-01')


# --- forecast_fmt markers sit on the forecast STEPS, not on every vertex -----

def _marked_points(art):
    """The vertices matplotlib draws a marker at (its markevery rule applied
    to the artist's own vertex list)."""
    xy = np.asarray(art.get_xydata()) if not hasattr(art, 'get_data_3d') \
        else np.column_stack(art.get_data_3d())
    every = art.get_markevery()
    idx = np.arange(len(xy))
    if every is None:
        return xy
    if isinstance(every, slice):
        return xy[idx[every]]
    return xy[np.asarray(every)]


@pytest.mark.parametrize('dims', [3, 2, 1])
def test_forecast_fmt_marker_is_drawn_only_at_the_forecast_steps(dims):
    """``forecast_fmt='ro:'`` put a marker on all ~901 antialiased vertices,
    so the dotted forecast drew as a solid red tube. A marker belongs on a
    TRUE observation only (plot()'s antialias contract) -- here the seam
    row and the t forecast steps -- and the dotted line stays smooth."""
    T = 5
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal((40, dims)), 0)
    if dims == 1:
        x = x[:, 0]
    fig = hyp.plot(x, predict='Kalman', t=T, forecast_fmt='ro:', show=False)
    (fc,) = _role(fig.axes[0], 'static')
    assert fc.get_marker() == 'o' and fc.get_linestyle() == ':'
    n_vertices = (len(fc.get_data_3d()[0]) if hasattr(fc, 'get_data_3d')
                  else len(fc.get_xdata()))
    assert n_vertices > 100          # still the smooth curve
    marked = _marked_points(fc)
    assert len(marked) == T + 1      # seam + the t forecast steps
    if dims == 1:
        # ...at the forecast's own rows: x = 39 (the seam) .. 44
        assert np.allclose(marked[:, 0], np.arange(39.0, 45.0))
    plt.close(fig)


def test_forecast_fmt_marker_off_antialias_marks_every_vertex():
    T = 5
    x = np.cumsum(np.random.default_rng(0).standard_normal((40, 2)), 0)
    fig = hyp.plot(x, predict='Kalman', t=T, forecast_fmt='ro:',
                   antialias=False, show=False)
    (fc,) = _role(fig.axes[0], 'static')
    assert len(fc.get_xdata()) == T + 1
    assert len(_marked_points(fc)) == T + 1
    plt.close(fig)


def test_animated_forecast_fmt_marker_is_drawn_only_at_the_forecast_steps():
    T = 4
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal((30, 3)), 0)
    anim = hyp.plot(x, predict='Kalman', t=T, forecast_fmt='ro:',
                    forecast_trail=1, animate=True, duration=1,
                    frame_rate=8, show=False)
    anim.draw_frame(7)
    ax = anim.figure.axes[0]
    for role in ('live', 'trail'):
        for art in _role(ax, role):
            if not art.get_visible():
                continue
            assert len(art.get_data_3d()[0]) > 100
            assert len(_marked_points(art)) == T + 1
    plt.close(anim.figure)
