"""`predict=` legend entries and the collection styling rule (1.1 release
review, feature-tour sections 7.1, 9.10 and 9.11).

Every forecast lists once in the legend under its model's name, on both
backends, static and animated, in the order data / forecasts / truth. A
collection of models keeps each dataset's colour and takes a linestyle per
model; `forecast_palette=` opts into one colour per model. The legend
glyph wears the forecasts' colour when they share one and a neutral gray
otherwise. No mocks: every assertion reads the drawn artists/traces.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from tests._plotly_colors import rgba as effective_rgba
import pytest
from matplotlib.colors import to_rgba

import hypertools as hyp
from hypertools.plot.forecast import (FORECAST_LEGEND_COLOR,
                                      FORECAST_LEGEND_MIN_ALPHA,
                                      FORECAST_MODEL_LINESTYLES)
from hypertools.plot.plotly_backend import _marker_size_px, _to_plotly_color


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _walks(n=2, rows=40):
    return [hyp.load('random_walk', n_samples=rows, n_features=3,
                     random_state=i) for i in range(n)]


def _ax(fig):
    return fig.axes[0]


def _mpl_forecasts(ax, role='static'):
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) == role]


def _legend_texts(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def _legend_handle(ax, label):
    legend = ax.get_legend()
    for handle, text in zip(legend.legend_handles, legend.get_texts()):
        if text.get_text() == label:
            return handle
    raise AssertionError(f'no legend entry {label!r} in {_legend_texts(ax)}')


def _plotly_entries(fig):
    """Legend entries in the order plotly lists them: by `legendrank`
    (default 1000), then trace order."""
    listed = [(tr.legendrank if tr.legendrank is not None else 1000, k, tr)
              for k, tr in enumerate(fig.data) if tr.showlegend]
    return [tr for _, _, tr in sorted(listed, key=lambda t: (t[0], t[1]))]


def _plotly_forecasts(fig, role='static'):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') == role]


def _plotly_legend_traces(fig):
    return [tr for tr in fig.data if (tr.meta or {}).get('hyp_legend_entry')]


def _rgb(rgba_string):
    inner = rgba_string[rgba_string.index('(') + 1:-1].split(',')
    return tuple(round(float(v)) for v in inner[:3])


# --------------------------------------------------------------------------
# the single-model form lists its forecast

def test_single_model_forecast_is_listed_once_under_its_model_name():
    fig = hyp.plot(_walks(2), predict='Kalman', t=4, legend=True, show=False)
    ax = _ax(fig)
    assert _legend_texts(ax) == ['1', '2', 'Kalman']
    handle = _legend_handle(ax, 'Kalman')
    # two datasets, two colours -> the entry names the model, in neutral gray
    assert to_rgba(handle.get_color()) == to_rgba(FORECAST_LEGEND_COLOR)
    assert handle.get_linestyle() == '-'
    # legible: floored, not the forecasts' own 0.5
    assert handle.get_alpha() == pytest.approx(FORECAST_LEGEND_MIN_ALPHA)
    # ...and no forecast artist carries a label of its own
    assert all(ln.get_label() == '_nolegend_' for ln in _mpl_forecasts(ax))


def test_single_dataset_forecast_entry_wears_the_forecast_colour():
    fig = hyp.plot(_walks(1), predict='Kalman', t=4, legend=True,
                   forecast_fmt=':', show=False)
    ax = _ax(fig)
    forecast, = _mpl_forecasts(ax)
    handle = _legend_handle(ax, 'Kalman')
    assert to_rgba(handle.get_color()) == to_rgba(forecast.get_color())
    assert handle.get_linestyle() == forecast.get_linestyle() == ':'


def test_no_legend_means_no_forecast_entry():
    fig = hyp.plot(_walks(2), predict='Kalman', t=4, show=False)
    assert _ax(fig).get_legend() is None
    pl = hyp.plot(_walks(2), predict='Kalman', t=4, show=False,
                  backend='plotly')
    assert not pl.layout.showlegend


def test_plotly_single_model_forecast_is_listed_once():
    fig = hyp.plot(_walks(2), predict='Kalman', t=4, legend=True,
                   show=False, backend='plotly')
    assert [tr.name for tr in _plotly_entries(fig)] == ['1', '2', 'Kalman']
    assert all(not tr.showlegend for tr in _plotly_forecasts(fig))
    entry, = _plotly_legend_traces(fig)
    assert entry.meta['hyp_legend_entry'] == 'Kalman'
    assert 'hyp_forecast_role' not in entry.meta
    assert effective_rgba(entry)[:3] == pytest.approx(to_rgba(FORECAST_LEGEND_COLOR)[:3])
    assert effective_rgba(entry)[-1] == pytest.approx(FORECAST_LEGEND_MIN_ALPHA)
    assert entry.line.dash == 'solid'


def test_plotly_single_dataset_entry_wears_the_forecast_colour():
    fig = hyp.plot(_walks(1), predict='Kalman', t=4, legend=True,
                   forecast_fmt='--', show=False, backend='plotly')
    forecast, = _plotly_forecasts(fig)
    entry, = _plotly_legend_traces(fig)
    assert _rgb(entry.line.color) == _rgb(forecast.line.color)
    assert effective_rgba(entry)[-1] == pytest.approx(FORECAST_LEGEND_MIN_ALPHA)
    assert entry.line.dash == forecast.line.dash == 'dash'


# --------------------------------------------------------------------------
# a collection: dataset colour, linestyle per model

def test_collection_keeps_dataset_colour_and_dashes_per_model():
    fig = hyp.plot(_walks(2), predict=['Kalman', 'ARIMA'], t=4, legend=True,
                   show=False)
    ax = _ax(fig)
    forecasts = _mpl_forecasts(ax)
    assert len(forecasts) == 4
    data_lines = [ln for ln in ax.lines
                  if getattr(ln, '_hyp_forecast_role', None) is None]
    for ln in forecasts:
        source = data_lines[ln._hyp_forecast_dataset]
        assert to_rgba(ln.get_color()) == to_rgba(source.get_color())
        expected = {'Kalman': '-', 'ARIMA': '--'}[ln._hyp_forecast_label]
        assert ln.get_linestyle() == expected
    assert _legend_texts(ax) == ['1', '2', 'Kalman', 'ARIMA']
    for label, style in (('Kalman', '-'), ('ARIMA', '--')):
        handle = _legend_handle(ax, label)
        assert to_rgba(handle.get_color()) == to_rgba(FORECAST_LEGEND_COLOR)
        assert handle.get_linestyle() == style


def test_collection_cycles_the_four_linestyles_in_model_order():
    models = ['Kalman', 'ARIMA', 'GaussianProcess', 'AutoRegressor']
    fig = hyp.plot(_walks(1), predict=models, t=3, legend=True, show=False)
    ax = _ax(fig)
    seen = {ln._hyp_forecast_label: ln.get_linestyle()
            for ln in _mpl_forecasts(ax)}
    assert seen == dict(zip(models, FORECAST_MODEL_LINESTYLES))


def test_plotly_collection_matches_the_matplotlib_rule():
    fig = hyp.plot(_walks(2), predict=['Kalman', 'ARIMA'], t=4, legend=True,
                   show=False, backend='plotly')
    forecasts = _plotly_forecasts(fig)
    assert len(forecasts) == 4
    data = [tr for tr in fig.data if (tr.meta or {}).get('hyp_trace_index')
            is not None]
    for tr in forecasts:
        source = data[tr.meta['hyp_dataset']]
        assert _rgb(tr.line.color) == _rgb(source.line.color)
        assert tr.line.dash == {'Kalman': 'solid', 'ARIMA': 'dash'}[tr.name]
    assert [tr.name for tr in _plotly_entries(fig)] == \
        ['1', '2', 'Kalman', 'ARIMA']
    kalman, arima = _plotly_legend_traces(fig)
    assert kalman.line.dash == 'solid' and arima.line.dash == 'dash'
    neutral = _to_plotly_color(FORECAST_LEGEND_COLOR, FORECAST_LEGEND_MIN_ALPHA)
    assert effective_rgba(kalman) == effective_rgba(arima)
    assert _rgb(kalman.line.color) == _rgb(neutral)
    assert effective_rgba(kalman)[-1] == pytest.approx(FORECAST_LEGEND_MIN_ALPHA)


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_forecast_palette_colours_a_collection_by_model(backend):
    fig = hyp.plot(_walks(2), predict=['Kalman', 'ARIMA'], t=4, legend=True,
                   forecast_palette='Set1', show=False, backend=backend)
    if backend == 'matplotlib':
        ax = _ax(fig)
        by_model = {}
        for ln in _mpl_forecasts(ax):
            by_model.setdefault(ln._hyp_forecast_label, set()).add(
                to_rgba(ln.get_color()))
        assert all(len(colours) == 1 for colours in by_model.values())
        assert by_model['Kalman'] != by_model['ARIMA']
        for label in ('Kalman', 'ARIMA'):
            handle = _legend_handle(ax, label)
            assert {to_rgba(handle.get_color())} == by_model[label]
    else:
        by_model = {}
        for tr in _plotly_forecasts(fig):
            by_model.setdefault(tr.name, set()).add(tr.line.color)
        assert all(len(colours) == 1 for colours in by_model.values())
        assert by_model['Kalman'] != by_model['ARIMA']
        for entry in _plotly_legend_traces(fig):
            assert {entry.line.color} == by_model[entry.name]


def test_forecast_fmt_replaces_the_model_cycle():
    fig = hyp.plot(_walks(2), predict=['Kalman', 'ARIMA'], t=4,
                   forecast_fmt=':', show=False)
    assert {ln.get_linestyle() for ln in _mpl_forecasts(_ax(fig))} == {':'}


# --------------------------------------------------------------------------
# order and truth=

def _series(n=60):
    t = np.linspace(0, 4 * np.pi, n)
    return np.column_stack([np.sin(t), np.cos(t)])


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_legend_lists_data_then_forecast_then_truth(backend):
    data = _series()
    fig = hyp.plot(data[:50], predict='Kalman', t=10, truth=data[50:],
                   reduce=None, ndims=2, names=['observed'], legend=True,
                   show=False, backend=backend)
    if backend == 'matplotlib':
        assert _legend_texts(_ax(fig)) == ['observed', 'Kalman', 'truth']
    else:
        assert [tr.name for tr in _plotly_entries(fig)] == \
            ['observed', 'Kalman', 'truth']


def test_plotly_truth_marks_every_observation_not_every_vertex():
    data = _series()
    t = 10
    fig = hyp.plot(data[:50], predict='Kalman', t=t, truth=data[50:],
                   reduce=None, ndims=2, legend=True, show=False,
                   backend='plotly')
    truth, = _plotly_forecasts(fig, 'truth')
    sizes = np.asarray(truth.marker.size, dtype=float)
    # the antialiased curve has many more vertices than the t + 1 rows
    # (seam included); only the rows carry a marker
    assert sizes.shape[0] > t + 1
    assert int((sizes > 0).sum()) == t + 1
    assert sizes[0] > 0 and sizes[-1] > 0
    assert set(sizes[sizes > 0]) == {_marker_size_px(4, 'o', 2)}
    assert truth.legendrank > 1000


# --------------------------------------------------------------------------
# animated: the live forecast is listed too

def test_animated_matplotlib_forecast_is_listed():
    fig, ani = hyp.plot(_walks(2), predict='Kalman', t=3, animate=True,
                        legend=True, duration=2, frame_rate=4, show=False)
    ax = [a for a in fig.axes if hasattr(a, 'zaxis')][0]
    assert _legend_texts(ax) == ['1', '2', 'Kalman']
    assert len(_mpl_forecasts(ax, 'live')) == 2
    handle = _legend_handle(ax, 'Kalman')
    assert to_rgba(handle.get_color()) == to_rgba(FORECAST_LEGEND_COLOR)


def test_animated_plotly_forecast_is_listed():
    fig = hyp.plot(_walks(2), predict='Kalman', t=3, animate=True,
                   legend=True, duration=2, frame_rate=4, show=False,
                   backend='plotly')
    assert len(fig.frames) > 0
    assert [tr.name for tr in _plotly_entries(fig)] == ['1', '2', 'Kalman']
    entry, = _plotly_legend_traces(fig)
    assert effective_rgba(entry)[:3] == pytest.approx(to_rgba(FORECAST_LEGEND_COLOR)[:3])
    assert effective_rgba(entry)[-1] == pytest.approx(FORECAST_LEGEND_MIN_ALPHA)
    assert all(not tr.showlegend for tr in _plotly_forecasts(fig, 'live'))
