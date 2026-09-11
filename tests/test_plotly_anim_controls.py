# -*- coding: utf-8 -*-
"""Plotly's Play/Pause controls sit clear of the x axis' tick labels.

1.1 release review: the controls hung at paper y=-0.06 -- right where a
visible x axis (``axis_scale='data'``, an ``ndims=1`` series, a date axis
with its second tick-label line) draws its tick labels and title -- so on a
dated animated forecast they covered the "2020" of the first date tick.
They now go below the tick labels and the axis title, with the bottom
margin opened so they are not clipped.

Checked on kaleido pixels: the buttons' footprint (what changes when they
are hidden) must not cover any ink the figure draws without them, and must
lie inside the canvas.
"""

import io

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp

go = pytest.importorskip('plotly.graph_objects')


def _png(fig, width=640, height=480):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(fig.to_image(
        format='png', width=width, height=height))).convert('RGB')).astype(
            int)


def _controls_clear_of_the_axis(fig):
    """Dark ink (tick labels, axis title) inside the controls' footprint,
    in ONE render: the controls are drawn in a marker colour (pure green
    border and text) so their footprint is readable off the same image --
    hiding them instead would let plotly re-lay the margins out."""
    shown = go.Figure(fig)
    shown.frames = ()
    menu = shown.layout.updatemenus[0]
    menu.bordercolor = 'rgb(0,255,0)'
    menu.borderwidth = 2
    menu.font.color = 'rgb(0,255,0)'
    # see-through buttons, so ink they would COVER shows in the render
    menu.bgcolor = 'rgba(0,0,0,0)'
    img = _png(shown)
    green = (img[..., 1] > 200) & (img[..., 0] < 90) & (img[..., 2] < 90)
    rows, cols = np.nonzero(green)
    assert rows.size, 'the controls did not render'
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    # inside the canvas, not clipped at its bottom edge
    assert r1 < img.shape[0] - 1
    box = img[r0:r1 + 1, c0:c1 + 1]
    dark = (box.max(axis=2) < 150) & ~green[r0:r1 + 1, c0:c1 + 1]
    return int(dark.sum())


def _dated_forecast(**kw):
    rng = np.random.default_rng(0)
    idx = pd.date_range('2020-01-01', periods=60, freq='D')
    df = pd.DataFrame({'price': np.cumsum(rng.standard_normal(60))},
                      index=idx)
    return hyp.plot(df, ndims=1, reduce=None, backend='plotly', show=False,
                    predict='Kalman', t=8, animate=True, duration=2,
                    frame_rate=5, **kw)


def test_controls_clear_the_date_tick_labels():
    assert _controls_clear_of_the_axis(_dated_forecast()) == 0


def test_controls_clear_the_tick_labels_and_axis_title():
    assert _controls_clear_of_the_axis(_dated_forecast(xlabel='date')) == 0


def test_controls_clear_a_numeric_data_axis():
    x = np.cumsum(np.random.default_rng(1).standard_normal((40, 2)), 0)
    fig = hyp.plot(x, backend='plotly', show=False, axis_scale='data',
                   xlabel='x', animate=True, duration=1, frame_rate=4)
    assert _controls_clear_of_the_axis(fig) == 0


def test_axisless_animation_keeps_its_controls_where_they_were():
    """3-D (and unit-scale 2-D) figures draw no tick labels: the controls
    keep their historical spot and margin."""
    x = np.cumsum(np.random.default_rng(1).standard_normal((30, 3)), 0)
    fig = hyp.plot(x, backend='plotly', show=False, animate=True,
                   duration=1, frame_rate=4)
    menu = fig.layout.updatemenus[0]
    assert (menu.y, menu.yanchor) == (-0.06, 'top')
    assert fig.layout.margin.b == 64
