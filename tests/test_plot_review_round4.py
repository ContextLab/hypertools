"""Codex red-team round 4 on the 1.1 release review (2026-09-07), each
finding pinned at the public API on both backends. No mocks.

1 regrouped animations repainted a `forecast_fmt` colour letter (and
  plotly halved a recoloured forecast's per-frame alpha)
2 mixture-hue legends lost their forecast and truth entries
3 forecasts/truth on a reused matplotlib axes styled themselves from an
  earlier call's lines
4 repeated calls into one axes/figure duplicated legend entries
5 a legend and a colorbar arriving in separate calls shared one gutter
6 a multi-line title widened the top margin but not the rows
7 plotly drew a marker-only `forecast_fmt` as connected lines
8 animated plotly collection traces tagged the forecast index as dataset
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex, to_rgb

import hypertools as hyp
from hypertools.plot.plotly_backend import _rgb_triplet, cell_layout_keys


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _walk(seed=4, rows=20):
    return np.cumsum(np.random.default_rng(seed).normal(size=(rows, 3)),
                     axis=0)


HUE = ['a'] * 10 + ['b'] * 10


def _mpl_role(fig, role):
    return [ln for ln in fig.axes[0].lines
            if getattr(ln, '_hyp_forecast_role', None) == role]


def _pl_role(fig, role):
    return [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_forecast_role') == role]


def _mpl_legend(fig):
    return [t.get_text() for t in fig.axes[0].get_legend().get_texts()]


def _pl_legend(fig):
    listed = [(tr.legendrank if tr.legendrank is not None else 1000, k, tr)
              for k, tr in enumerate(fig.data) if tr.showlegend]
    return [tr.name for _, _, tr in sorted(listed, key=lambda t: t[:2])]


# --- 1: a format-string colour survives a regrouped animation ---------------

def test_regrouped_animation_keeps_a_forecast_fmt_colour():
    x = _walk()
    kw = dict(hue=HUE, predict=['Kalman', 'ARIMA'], forecast_fmt='ro:',
              alpha=.7, t=3, animate=True, forecast_trail=2, antialias=False,
              duration=1, frame_rate=4, legend=True, show=False)
    fig, ani = hyp.plot(x, **kw)
    fig.canvas.draw()
    for frame in range(ani._save_count):
        ani._func(frame, *ani._args)
    for ln in _mpl_role(fig, 'live'):
        assert to_hex(ln.get_color()) == '#ff0000'
        assert ln.get_alpha() == pytest.approx(0.7)
    pl = hyp.plot(x, backend='plotly', **kw)
    live = _pl_role(pl, 'live')
    assert [_rgb_triplet(tr.line.color) for tr in live] == [(255, 0, 0)] * 2
    assert all(tr.line.color.endswith(',0.7)') for tr in live)
    # no frame repaints a pinned colour
    for frame in pl.frames:
        for tr in frame.data:
            if tr.line is not None and tr.line.color is not None:
                assert _rgb_triplet(tr.line.color) == (255, 0, 0)
                assert tr.line.color.endswith(',0.7)')


# --- 2: mixture-hue legends list forecasts and truth --------------------

@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
@pytest.mark.parametrize('animated', [False, True])
def test_mixture_hue_legend_lists_forecasts_and_truth(backend, animated):
    x = _walk()
    mix = np.column_stack([np.linspace(0, 1, 20), np.linspace(1, 0, 20)])
    out = hyp.plot(x, hue=mix, predict=['Kalman', 'ARIMA'], t=3,
                   truth=x[-3:], legend=True, animate=animated, duration=1,
                   frame_rate=4, show=False, backend=backend)
    if backend == 'matplotlib':
        fig = out[0] if animated else out
        assert _mpl_legend(fig) == ['1', '2', 'Kalman', 'ARIMA', 'truth']
    else:
        assert _pl_legend(out) == ['1', '2', 'Kalman', 'ARIMA', 'truth']


# --- 3 and 4: repeated calls into one axes / figure --------------------------

def test_reused_matplotlib_axes_styles_overlays_from_its_own_call():
    x = _walk()
    fig, axes = hyp.subplots(1, 1)
    for k in range(3):
        hyp.plot(x + 2 * k, ax=axes[0], predict='Kalman', t=3,
                 truth=x[-3:] + 2 * k, legend=True, names=[f'data{k}'],
                 show=False)
    ax = axes[0]
    data = [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) is None]
    forecasts = _mpl_role(fig, 'static')
    assert [to_rgb(ln.get_color()) for ln in forecasts] == \
        [to_rgb(ln.get_color()) for ln in data]
    truths = [ln for ln in _mpl_role(fig, 'truth')
              if ln.get_linestyle() != 'None']
    assert [to_rgb(ln.get_color()) for ln in truths] == \
        [to_rgb(ln.get_color()) for ln in data]
    assert _mpl_legend(fig) == ['data0', 'data1', 'data2', 'Kalman', 'truth']


def test_reused_plotly_figure_consolidates_legend_entries():
    x = _walk()
    fig = None
    for k in range(3):
        fig = hyp.plot(x + 2 * k, ax=fig, predict='Kalman', t=3,
                       truth=x[-3:] + 2 * k, legend=True,
                       names=[f'data{k}'], show=False, backend='plotly')
    assert _pl_legend(fig) == ['data0', 'data1', 'data2', 'Kalman', 'truth']
    data = [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_trace_index') is not None]
    assert [_rgb_triplet(tr.line.color) for tr in _pl_role(fig, 'static')] \
        == [_rgb_triplet(tr.line.color) for tr in data]


def test_reused_plotly_cell_consolidates_legend_entries():
    x = _walk()
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    for k in range(2):
        hyp.plot(x + 2 * k, ax=cells[0], predict='Kalman', t=3, legend=True,
                 names=[f'data{k}'], show=False, backend='plotly')
    hyp.plot(x, ax=cells[1], predict='Kalman', t=3, legend=True,
             names=['other'], show=False, backend='plotly')
    key0 = cell_layout_keys(0)['legend']
    cell0 = [tr.name for tr in fig.data if tr.showlegend
             and getattr(tr, 'legend', None) == key0]
    assert cell0 == ['data0', 'data1', 'Kalman']


# --- 5: a legend and a colorbar from separate calls sit side by side ---------

@pytest.mark.parametrize('ndims', [1, 2, 3])
def test_legend_then_colorbar_in_separate_calls_share_no_gutter(ndims):
    x = _walk()
    fig, cells = hyp.subplots(1, 2, backend='plotly', ndims=ndims)
    hyp.plot([x, x + 2], ax=cells[0], legend=True, names=['a', 'b'],
             ndims=ndims, show=False, backend='plotly')
    hyp.plot(x, ax=cells[0], hue=np.arange(20.0), colorbar=True,
             ndims=ndims, show=False, backend='plotly')
    from hypertools.plot.plotly_backend import PANEL_LEGEND_PX
    assert fig.layout.meta['hyp_grid']['gutter_px'] >= 2 * PANEL_LEGEND_PX
    legend_x = fig.layout.legend.x
    bars = [tr.marker.colorbar.x for tr in fig.data
            if getattr(getattr(tr, 'marker', None), 'showscale', None)]
    plot_w = fig.layout.width - fig.layout.margin.l - fig.layout.margin.r
    assert all((bx - legend_x) * plot_w >= PANEL_LEGEND_PX - 1
               for bx in bars)


# --- 6: a multi-line title makes room between the rows ----------------------

def test_multiline_cell_title_rebuilds_the_rows():
    x = _walk()
    fig, cells = hyp.subplots(2, 1, backend='plotly', size=[6, 6])
    for cell in cells:
        hyp.plot(x, ax=cell, title='three\nlarge\nlines',
                 title_kwargs={'fontsize': 30}, show=False, backend='plotly')
    mg = fig.layout.margin
    plot_h = fig.layout.height - mg.t - mg.b
    top, bottom = fig.layout.scene.domain, fig.layout.scene2.domain
    gap_px = (top.y[0] - bottom.y[1]) * plot_h
    title_px = fig.layout.meta['hyp_grid']['title_px']
    assert title_px > 100          # three 30 pt lines
    assert gap_px >= title_px      # the row gap holds the whole title


# --- 7: marker-only forecast_fmt on plotly ----------------------------------

@pytest.mark.parametrize('animated', [False, True])
def test_plotly_marker_only_forecast_fmt_draws_markers(animated):
    x = _walk()
    fig = hyp.plot(x, predict='Kalman', t=3, forecast_fmt='ro',
                   antialias=False, animate=animated, duration=1,
                   frame_rate=4, show=False, backend='plotly')
    role = 'live' if animated else 'static'
    tr, = _pl_role(fig, role)
    assert tr.mode == 'markers'
    mfig = hyp.plot(x, predict='Kalman', t=3, forecast_fmt='ro',
                    antialias=False, animate=animated, duration=1,
                    frame_rate=4, show=False)
    mfig = mfig[0] if animated else mfig
    ln, = _mpl_role(mfig, role)
    assert ln.get_linestyle() == 'None' and ln.get_marker() == 'o'


# --- 8: animated plotly collection traces name their SOURCE dataset --------

def test_animated_plotly_collection_tags_source_datasets():
    x = _walk()
    fig = hyp.plot([x, x + 2], predict=['Kalman', 'ARIMA'], t=3,
                   animate=True, duration=1, frame_rate=4, show=False,
                   backend='plotly')
    assert [tr.meta['hyp_dataset'] for tr in _pl_role(fig, 'live')] == \
        [0, 1, 0, 1]
