"""Codex red-team round 3 on the 1.1 release review (2026-09-07), each
finding pinned at the public API on both backends. No mocks.

1  a collection under hue regrouping continued the wrong run (static) and
   raised IndexError animated (forecast index used as dataset index)
2  a forecaster fitted on several datasets could not animate
3  plotly dropped forecast_fmt's colour letter and markers
4  plotly animations halved a recoloured forecast's alpha
5  a second call into the same plotly cell restarted the palette
6  plotly forecast legend keys compared opacity as colour
7  legend_colors= beside forecasts: pairs gained entries, lists were refused
8  matplotlib panel legends overlapped their colorbars
9  a plotly gutter rebuild discarded multi-line title room
10 plotly cells overwrote an explicit legend position
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.transforms import Bbox

import hypertools as hyp
from hypertools.plot.plotly_backend import _rgb_triplet


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _walk(seed=4, rows=20):
    return np.cumsum(np.random.default_rng(seed).normal(size=(rows, 3)),
                     axis=0)


def _mpl_role(fig, role):
    return [ln for ln in fig.axes[0].lines
            if getattr(ln, '_hyp_forecast_role', None) == role]


def _pl_role(fig, role):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') == role]


def _pl_data(fig):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_trace_index') is not None]


def _pl_entries(fig):
    return [tr for tr in fig.data if (tr.meta or {}).get('hyp_legend_entry')]


HUE = ['a'] * 10 + ['b'] * 10


# --- 1: a collection under regrouping ---------------------------------------

def test_collection_under_hue_regrouping_continues_the_final_run():
    x = _walk()
    fig = hyp.plot(x, hue=HUE, predict=['Kalman', 'ARIMA'], t=3, legend=True,
                   show=False)
    runs = [ln for ln in fig.axes[0].lines
            if getattr(ln, '_hyp_forecast_role', None) is None]
    final = to_rgb(runs[-1].get_color())     # run 'b' holds the last row
    for fc in _mpl_role(fig, 'static'):
        assert to_rgb(fc.get_color()) == final
    pl = hyp.plot(x, hue=HUE, predict=['Kalman', 'ARIMA'], t=3, legend=True,
                  show=False, backend='plotly')
    final = _rgb_triplet(_pl_data(pl)[-1].line.color)
    for fc in _pl_role(pl, 'static'):
        assert _rgb_triplet(fc.line.color) == final


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_animated_collection_under_regrouping_with_a_trail_draws(backend):
    x = _walk()
    kw = dict(predict=['Kalman', 'ARIMA'], t=3, legend=True, animate=True,
              duration=1, frame_rate=4, forecast_trail=2, show=False,
              hue=[HUE] * 2, backend=backend)
    out = hyp.plot([x, x + 2], **kw)
    if backend == 'matplotlib':
        fig, ani = out
        for frame in range(ani._save_count or 4):
            ani._func(frame, *ani._args)        # every frame draws
        assert len(_mpl_role(fig, 'live')) == 4
    else:
        assert len(out.frames) > 0
        assert len(_pl_role(out, 'live')) == 4


# --- 2: a multi-dataset fitted forecaster animates --------------------------

def test_for_dataset_binds_one_fitted_model():
    x = _walk()
    _, model = hyp.predict([x, x + 2], return_model=True, t=3)
    assert len(model.models_) == 2
    view = model.for_dataset(1)
    assert len(view.models_) == 1 and view.models_[0] is model.models_[1]
    assert np.shape(view.data) == x.shape
    one = view.predict_new(x + 2, 3)
    both = model.predict_new([x, x + 2], 3)
    assert np.allclose(np.asarray(one), np.asarray(both[1]))
    with pytest.raises(IndexError):
        model.for_dataset(2)


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_a_fitted_multi_dataset_forecaster_animates(backend):
    x = _walk()
    _, model = hyp.predict([x, x + 2], return_model=True, t=3)
    out = hyp.plot([x, x + 2], predict=model, t=3, animate=True, duration=1,
                   frame_rate=4, show=False, backend=backend)
    if backend == 'matplotlib':
        fig, ani = out
        ani._func(3, *ani._args)
        assert len(_mpl_role(fig, 'live')) == 2
    else:
        assert len(_pl_role(out, 'live')) == 2


# --- 3: plotly honours forecast_fmt's colour and markers --------------------

@pytest.mark.parametrize('animated', [False, True])
def test_plotly_forecast_fmt_colour_and_markers(animated):
    x = _walk()
    kw = dict(predict=['Kalman', 'ARIMA'], t=3, legend=True, show=False,
              forecast_fmt='ro:', duration=1, frame_rate=4)
    fig = hyp.plot(x, animate=animated, backend='plotly', **kw)
    role = 'live' if animated else 'static'
    for tr in _pl_role(fig, role):
        assert _rgb_triplet(tr.line.color) == (255, 0, 0)
        assert tr.line.dash == 'dot'
        assert tr.mode == 'lines+markers'
        assert tr.marker.symbol == 'circle'
    for entry in _pl_entries(fig):
        assert entry.mode == 'lines+markers'
        assert _rgb_triplet(entry.line.color) == (255, 0, 0)
    # matplotlib parity on the same call
    mfig = hyp.plot(x, animate=animated, **kw)
    if animated:
        mfig = mfig[0]
    for ln in _mpl_role(mfig, role):
        assert to_rgb(ln.get_color()) == to_rgb('r')
        assert ln.get_linestyle() == ':' and ln.get_marker() == 'o'


# --- 4: plotly animations keep a recoloured forecast's alpha -----------------

def test_plotly_animated_recoloured_forecast_keeps_its_alpha():
    x = _walk()
    fig = hyp.plot(x, alpha=.7, forecast_palette='Set1', predict='Kalman',
                   t=3, animate=True, duration=1, frame_rate=4,
                   forecast_trail=2, show=False, backend='plotly')
    live, = _pl_role(fig, 'live')
    assert live.meta['hyp_forecast_alpha'] == pytest.approx(0.7)
    trails = [tr.meta['hyp_forecast_alpha'] for tr in _pl_role(fig, 'trail')]
    assert max(trails) < 0.7 and min(trails) > 0.0
    mfig, ani = hyp.plot(x, alpha=.7, forecast_palette='Set1',
                         predict='Kalman', t=3, animate=True, duration=1,
                         frame_rate=4, forecast_trail=2, show=False)
    mlive, = _mpl_role(mfig, 'live')
    assert mlive.get_alpha() == pytest.approx(0.7)
    assert sorted(ln.get_alpha() for ln in _mpl_role(mfig, 'trail')) == \
        pytest.approx(sorted(trails))


# --- 5: a second call into the same plotly cell continues the palette ------

def test_second_call_into_the_same_plotly_cell_continues_the_palette():
    x = _walk()
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    hyp.plot(x, ax=cells[0], show=False, backend='plotly')
    hyp.plot(x + 2, ax=cells[0], show=False, backend='plotly')
    hyp.plot(x, ax=cells[1], show=False, backend='plotly')
    colours = [tr.line.color for tr in _pl_data(fig)]
    assert colours[0] != colours[1]        # cell 0's two datasets differ
    assert colours[2] == colours[0]        # cell 1 starts its own palette
    mfig, axes = hyp.subplots(1, 1)
    hyp.plot(x, ax=axes[0], show=False)
    hyp.plot(x + 2, ax=axes[0], show=False)
    mcolours = [to_rgb(ln.get_color()) for ln in axes[0].lines[:2]]
    assert [_rgb_triplet(c) for c in colours[:2]] == \
        [tuple(round(v * 255) for v in c) for c in mcolours]


# --- 6: legend keys compare colour, not opacity -----------------------------

def test_plotly_legend_keys_ignore_alpha_when_comparing_colours():
    x = _walk()
    fig = hyp.plot([x, x + 2], alpha=[1, .4], predict=['Kalman', 'ARIMA'],
                   t=3, forecast_palette=['red', 'red'], legend=True,
                   show=False, backend='plotly')
    for entry in _pl_entries(fig):
        assert _rgb_triplet(entry.line.color) == (255, 0, 0)


# --- 7: legend_colors= beside forecasts -------------------------------------

@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_explicit_legend_pairs_define_the_whole_legend(backend):
    x = _walk()
    fig = hyp.plot(x, predict='Kalman', t=3, truth=x[-3:] + 1,
                   legend=True, legend_colors=[('Custom', 'red')],
                   show=False, backend=backend)
    if backend == 'matplotlib':
        texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    else:
        texts = [tr.name for tr in fig.data if tr.showlegend]
    assert texts == ['Custom']


def test_plain_legend_colour_list_recolours_the_final_legend():
    x = _walk()
    fig = hyp.plot(x, predict='Kalman', t=3, truth=x[-3:] + 1, legend=True,
                   legend_colors=['navy', 'gold', 'green'], show=False)
    legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == \
        ['1', 'Kalman', 'truth']
    assert [to_rgb(h.get_color()) for h in legend.legend_handles] == \
        [to_rgb('navy'), to_rgb('gold'), to_rgb('green')]


# --- 8: panel legends clear their colorbars ---------------------------------

@pytest.mark.parametrize('ndims', [1, 2, 3])
def test_panel_legend_and_colorbar_do_not_overlap(ndims):
    x = _walk()
    fig = hyp.plot([x, x + 2], panels=True, ndims=ndims, legend=True,
                   colorbar=True, size=[9, 4], show=False,
                   cluster={'model': 'KMeans',
                            'kwargs': {'n_clusters': 2, 'random_state': 0}})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panels = [a for a in fig.axes if a.get_legend() is not None]
    bars = [a for a in fig.axes if a.get_label() == '<colorbar>']
    assert len(panels) == len(bars) == 2
    for panel, bar in zip(panels, bars):
        legend_box = panel.get_legend().get_window_extent(renderer)
        assert Bbox.intersection(legend_box,
                                 bar.get_window_extent(renderer)) is None


# --- 9 and 10: plotly cells keep title room and explicit legend positions --

def test_gutter_rebuild_keeps_multiline_title_room():
    x = _walk()
    fig, cells = hyp.subplots(2, 1, backend='plotly', size=[6, 6])
    hyp.plot(x, ax=cells[0], title='three\nlarge\nlines',
             title_kwargs={'fontsize': 30}, show=False, backend='plotly')
    top_before = fig.layout.margin.t
    assert top_before > 40
    hyp.plot([x, x + 2], ax=cells[1], legend=True, names=['a', 'b'],
             show=False, backend='plotly')
    assert fig.layout.margin.t >= top_before
    assert fig.layout.meta['hyp_grid']['gutter_px'] > 0


def test_plotly_panels_keep_an_explicit_legend_position():
    x = _walk()
    fig = hyp.plot([x, x + 2], panels=True, legend=True, show=False,
                   legend_kwargs={'x': .02, 'y': .98, 'xanchor': 'left',
                                  'yanchor': 'top'}, backend='plotly')
    from hypertools.plot.plotly_backend import cell_layout_keys
    for i in range(2):
        keys = cell_layout_keys(i)
        d = fig.layout[keys['scene']].domain
        legend = fig.layout[keys['legend']]
        assert legend.x == pytest.approx(d.x[0] + .02 * (d.x[1] - d.x[0]))
        assert legend.y == pytest.approx(d.y[0] + .98 * (d.y[1] - d.y[0]))
        assert legend.yanchor == 'top'
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    hyp.plot(x, ax=cells[0], legend=True, names=['a'], show=False,
             backend='plotly',
             legend_kwargs={'x': .02, 'y': .98, 'yanchor': 'top'})
    d0 = fig.layout.scene.domain
    assert fig.layout.legend.x == pytest.approx(
        d0.x[0] + .02 * (d0.x[1] - d0.x[0]))
    hyp.plot(x, ax=cells[1], colorbar=True, hue=np.arange(20.0),
             show=False, backend='plotly')       # grows the gutters
    d0 = fig.layout.scene.domain
    assert fig.layout.legend.x == pytest.approx(
        d0.x[0] + .02 * (d0.x[1] - d0.x[0]))     # re-placed inside cell 0
