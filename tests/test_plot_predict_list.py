"""Multi-model forecast overlays: ``predict=['Kalman', 'ARIMA', 'GP']``.

GH #285: the 1.1 feature tour drew several forecasters on one trace by hand
(``ax.plot(signal[:, 0], 'k')`` plus one ``ax.plot(f.index, f.iloc[:, 0])``
per model) because `predict=` took a single spec. It now takes a collection,
reusing ``hyp.predict(x, model=[...])``'s ``{name: forecast}`` contract for
both the values and the names.
"""
import matplotlib
import matplotlib.colors
matplotlib.use('Agg')

import numpy as np                                        # noqa: E402
import pytest                                             # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402

import hypertools as hyp                                  # noqa: E402
from hypertools.plot.forecast import FORECAST_MODEL_LINESTYLES  # noqa: E402

MODELS = ['Kalman', 'ARIMA', 'GaussianProcess']
T = 5


@pytest.fixture
def signal():
    rng = np.random.default_rng(21)
    return np.cumsum(rng.standard_normal((40, 2)), axis=0)


def _by_role(fig, role):
    return [line for line in fig.axes[0].lines
            if getattr(line, '_hyp_forecast_role', None) == role]


def test_one_overlay_per_model_with_model_names_in_the_legend(signal):
    fig = hyp.plot(signal, reduce=None, ndims=2, predict=MODELS, t=T,
                   legend=True, antialias=False, show=False)
    overlays = _by_role(fig, 'static')
    assert len(overlays) == len(MODELS)
    # the overlay carries its model's name as a tag (its legend entry is a
    # proxy glyph, so the artist itself stays '_nolegend_')
    assert [line._hyp_forecast_label for line in overlays] == MODELS
    assert all(line.get_label() == '_nolegend_' for line in overlays)
    labels = [text.get_text()
              for text in fig.axes[0].get_legend().get_texts()]
    for name in MODELS:
        assert labels.count(name) == 1
    plt.close(fig)


def test_each_model_gets_its_own_linestyle_in_the_dataset_colour(signal):
    fig = hyp.plot(signal, reduce=None, ndims=2, predict=MODELS, t=T,
                   show=False)
    overlays = _by_role(fig, 'static')
    data_line = [line for line in fig.axes[0].lines
                 if getattr(line, '_hyp_forecast_role', None) is None][0]
    assert [line.get_linestyle() for line in overlays] == \
        list(FORECAST_MODEL_LINESTYLES[:len(MODELS)])
    assert {line.get_color() for line in overlays} == \
        {data_line.get_color()}
    plt.close(fig)


def test_the_overlay_values_match_hyp_predict(signal):
    fig = hyp.plot(signal, reduce=None, ndims=2, axis_scale='data',
                   predict=MODELS, t=T, antialias=False, show=False)
    expected = hyp.predict(signal, model=MODELS, t=T)
    overlays = _by_role(fig, 'static')
    for name, line in zip(MODELS, overlays):
        drawn = np.column_stack([np.asarray(line.get_xdata()),
                                 np.asarray(line.get_ydata())])
        # the drawn trace prepends the seam observation
        assert np.allclose(drawn[0], signal[-1])
        assert np.allclose(drawn[1:], np.asarray(expected[name]))
    plt.close(fig)


def test_the_mapping_form_names_the_overlays(signal):
    fig = hyp.plot(signal, reduce=None, ndims=2,
                   predict={'fast': 'Kalman', 'slow': 'ARIMA'}, t=T,
                   legend=True, show=False)
    assert [line._hyp_forecast_label for line in _by_role(fig, 'static')] == \
        ['fast', 'slow']
    labels = [text.get_text()
              for text in fig.axes[0].get_legend().get_texts()]
    assert labels[-2:] == ['fast', 'slow']
    plt.close(fig)


def test_forecast_fmt_may_be_one_entry_per_model(signal):
    fig = hyp.plot(signal, reduce=None, ndims=2, predict=MODELS, t=T,
                   forecast_fmt=[':', '--', '-'], show=False)
    styles = [line.get_linestyle() for line in _by_role(fig, 'static')]
    assert styles == [':', '--', '-']
    plt.close(fig)


def test_every_dataset_gets_every_model(signal):
    other = signal[::-1] + 1.0
    fig = hyp.plot([signal, other], reduce=None, ndims=2, predict=MODELS,
                   t=T, show=False)
    overlays = _by_role(fig, 'static')
    assert len(overlays) == len(MODELS) * 2
    # ...and each knows which SERIES it continues
    assert sorted(line._hyp_forecast_dataset for line in overlays) == \
        [0, 0, 0, 1, 1, 1]
    # the flat overlay list is MODEL-MAJOR: model m's two datasets sit at
    # positions 2m and 2m+1. Each keeps ITS DATASET'S colour and takes the
    # model's linestyle, so both questions can be read off the figure.
    data_lines = [line for line in fig.axes[0].lines
                  if getattr(line, '_hyp_forecast_role', None) is None]
    by_model = [overlays[m * 2:(m + 1) * 2] for m in range(len(MODELS))]
    for m, pair in enumerate(by_model):
        assert pair[0].get_color() == data_lines[0].get_color()
        assert pair[1].get_color() == data_lines[1].get_color()
        assert {line.get_linestyle() for line in pair} == \
            {FORECAST_MODEL_LINESTYLES[m]}
    plt.close(fig)


def test_a_collection_works_with_truth(signal):
    rng = np.random.default_rng(22)
    truth = signal[-1] + np.cumsum(rng.standard_normal((T, 2)), axis=0)
    fig = hyp.plot(signal, reduce=None, ndims=2, axis_scale='data',
                   predict=MODELS, t=T, truth=truth, legend=True,
                   antialias=False, show=False)
    assert len(_by_role(fig, 'static')) == len(MODELS)
    truth_line = _by_role(fig, 'truth')[0]
    assert np.allclose(np.asarray(truth_line.get_ydata()),
                       np.r_[signal[-1, 1], truth[:, 1]])
    plt.close(fig)


def test_an_animation_gets_one_live_artist_per_model(signal):
    anim = hyp.plot(signal, reduce=None, ndims=2, predict=['Kalman',
                                                           'ARIMA'],
                    t=T, animate=True, duration=1, frame_rate=4,
                    show=False)
    fig, ani = anim.figure, anim.animation
    ani._init_draw()
    for frame in range(3):
        ani._func(frame, *ani._args)
    live = _by_role(fig, 'live')
    assert len(live) == 2
    assert all(len(np.asarray(line.get_xdata())) for line in live)
    plt.close(fig)


def test_the_bundle_mirrors_hyp_predicts_dict(signal):
    bundle = hyp.plot(signal, reduce=None, ndims=2, predict=MODELS, t=T,
                      return_model=True, show=False)
    forecasts = bundle['predict']['forecasts']
    assert set(forecasts) == set(MODELS)
    for name in MODELS:
        assert np.asarray(forecasts[name][0]).shape == (T, 2)


def test_an_empty_collection_is_refused(signal):
    with pytest.raises(ValueError, match='at least one model spec'):
        hyp.plot(signal, reduce=None, ndims=2, predict=[], t=T, show=False)


def test_series_mode_takes_a_collection_too():
    import pandas as pd
    rng = np.random.default_rng(23)
    index = pd.Index(np.arange(40.0), name='day')
    df = pd.DataFrame({'price': np.cumsum(rng.standard_normal(40)) + 100.0},
                      index=index)
    fig = hyp.plot(df, reduce=None, ndims=1, predict=['Kalman', 'ARIMA'],
                   t=T, legend=True, antialias=False, show=False)
    overlays = _by_role(fig, 'static')
    assert len(overlays) == 2
    for line in overlays:
        assert np.array_equal(np.asarray(line.get_xdata()),
                              np.arange(39.0, 39.0 + T + 1))
    plt.close(fig)


def test_plotly_draws_one_named_trace_per_model(signal):
    pytest.importorskip('plotly')
    import plotly.io as pio
    pio.renderers.default = 'json'
    fig = hyp.plot(signal, reduce=None, ndims=2, predict=MODELS, t=T,
                   legend=True, backend='plotly', antialias=False,
                   show=False)
    overlays = [trace for trace in fig.data
                if (trace.meta or {}).get('hyp_forecast_role') == 'static']
    assert [trace.name for trace in overlays] == MODELS
    # the drawn overlays never list themselves; one data-free entry per
    # model does (the plotly twin of matplotlib's proxy handles)
    assert not any(trace.showlegend for trace in overlays)
    entries = [trace for trace in fig.data
               if (trace.meta or {}).get('hyp_legend_entry')]
    assert [trace.name for trace in entries] == MODELS
    assert all(trace.showlegend for trace in entries)
    assert {(trace.meta or {}).get('hyp_dataset') for trace in overlays} == {0}


# --- 1.1 release-review: forecast_hue= is one value per DATASET (F5) ----

def test_F5_forecast_hue_per_dataset_is_shared_across_models(signal):
    other = signal * 0.5 + 3.0
    fig = hyp.plot([signal, other], reduce=None, ndims=2,
                   predict=['Kalman', 'ARIMA'], t=T,
                   forecast_hue=['g1', 'g2'], antialias=False, show=False)
    try:
        overlays = _by_role(fig, 'static')
        assert len(overlays) == 4                       # 2 models x 2 sets
        colours = [tuple(np.round(line.get_color()
                                  if not isinstance(line.get_color(), str)
                                  else matplotlib.colors.to_rgba(
                                      line.get_color()), 6))
                   for line in overlays]
        # model-major order: [Kalman ds0, Kalman ds1, ARIMA ds0, ARIMA ds1]
        assert colours[0] == colours[2]                 # dataset 0, both models
        assert colours[1] == colours[3]                 # dataset 1, both models
        assert colours[0] != colours[1]                 # the two datasets differ
    finally:
        plt.close(fig)


def test_F5_a_mismatched_forecast_hue_names_both_counts(signal):
    with pytest.raises(ValueError) as info:
        hyp.plot([signal, signal * 2.0], reduce=None, ndims=2,
                 predict=['Kalman', 'ARIMA'], t=T,
                 forecast_hue=['g1', 'g2', 'g3'], show=False)
    message = str(info.value)
    assert '3 value(s)' in message
    assert '2 dataset(s) x 2 model(s) = 4 forecast(s)' in message
