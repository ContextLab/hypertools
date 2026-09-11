# -*- coding: utf-8 -*-
"""Figure-QA expectation findings on the plotly backend (1.1 release
review): each figure is checked against what `plot()`'s docstring promises
and what the matplotlib backend draws.

* ``frame_kwargs=`` was ignored -- the cube/square stayed black;
* a STATIC plot applied ``zoom=``, which the docstring calls
  animation-only and the matplotlib static view ignores;
* ``legend_kwargs={'x': 0, 'y': 1}`` kept hypertools' ``yanchor='middle'``,
  so the legend straddled the top edge;
* ``'^'`` draws a diamond in 3-D (plotly's Scatter3d has no triangle) --
  unavoidable, so the mapping is documented.

Real figures and kaleido-rendered pixels.
"""

import io
import warnings

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp

pytest.importorskip('plotly')


def _walk(d=3, n=40, seed=0):
    return np.cumsum(np.random.default_rng(seed).standard_normal((n, d)), 0)


def _png(fig, width=640, height=480):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(
        fig.to_image(format='png', width=width, height=height)))
        .convert('RGB')).astype(int)


def _cube(fig):
    cubes = [t for t in fig.data
             if t.type == 'scatter3d' and t.mode == 'lines'
             and t.hoverinfo == 'skip' and t.x is not None
             and len(t.x) == 36]
    assert len(cubes) == 1
    return cubes[0]


# --------------------------------------------------------------------------
# frame_kwargs


def test_frame_kwargs_colour_the_3d_cube():
    fig = hyp.plot(_walk(), backend='plotly', show=False,
                   frame_kwargs={'color': 'red', 'linewidth': 2})
    cube = _cube(fig)
    assert cube.line.color.replace(' ', '') in ('rgba(255,0,0,1.0)',
                                                'rgb(255,0,0)')
    default = _cube(hyp.plot(_walk(), backend='plotly', show=False))
    assert default.line.color == 'black'
    assert cube.line.width == pytest.approx(2 * default.line.width)
    # pixels: a red wireframe, and no black one
    img = _png(fig)
    red = (img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] < 60)
    black = img.sum(axis=2) < 60
    assert red.sum() > 500
    assert black.sum() < 50


def test_frame_kwargs_style_the_2d_square():
    fig = hyp.plot(_walk(d=2), backend='plotly', show=False,
                   frame_kwargs={'edgecolor': 'blue', 'linestyle': '--',
                                 'alpha': 0.5})
    square, = fig.layout.shapes
    assert square.line.color.replace(' ', '') == 'rgba(0,0,255,0.5)'
    assert square.line.dash == 'dash'
    assert square.fillcolor == 'rgba(0,0,0,0)'


def test_frame_kwargs_colour_fills_the_square_like_matplotlib():
    """matplotlib's `plot_square` hands `color=` to a Rectangle, which
    fills with it; `fill=False` keeps the outline only."""
    fig = hyp.plot(_walk(d=2), backend='plotly', show=False,
                   frame_kwargs={'color': 'lightgray'})
    square, = fig.layout.shapes
    assert square.fillcolor.replace(' ', '') == 'rgba(211,211,211,1.0)'
    fig2 = hyp.plot(_walk(d=2), backend='plotly', show=False,
                    frame_kwargs={'color': 'lightgray', 'fill': False})
    assert fig2.layout.shapes[0].fillcolor == 'rgba(0,0,0,0)'


def test_unmappable_frame_kwargs_are_named_in_a_warning():
    with pytest.warns(UserWarning, match="frame_kwargs.*zorder"):
        hyp.plot(_walk(), backend='plotly', show=False,
                 frame_kwargs={'color': 'red', 'zorder': 3})


def test_default_frame_is_unchanged_and_silent():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fig = hyp.plot(_walk(d=2), backend='plotly', show=False)
    square, = fig.layout.shapes
    assert square.line.color == 'black' and square.line.dash is None


# --------------------------------------------------------------------------
# zoom is animation-only


def _radius(eye):
    return float(np.sqrt(eye.x ** 2 + eye.y ** 2 + eye.z ** 2))


def test_static_plot_ignores_zoom_like_matplotlib():
    near = hyp.plot(_walk(), zoom=3, backend='plotly', show=False)
    far = hyp.plot(_walk(), zoom=1, backend='plotly', show=False)
    assert _radius(near.layout.scene.camera.eye) == pytest.approx(
        _radius(far.layout.scene.camera.eye))


def test_animation_still_zooms():
    near = hyp.plot(_walk(), zoom=3, backend='plotly', show=False,
                    animate='spin', duration=1, frame_rate=2)
    far = hyp.plot(_walk(), zoom=1, backend='plotly', show=False,
                   animate='spin', duration=1, frame_rate=2)
    assert _radius(near.layout.scene.camera.eye) < _radius(
        far.layout.scene.camera.eye)


# --------------------------------------------------------------------------
# legend_kwargs position -> anchor


@pytest.mark.parametrize('pos, anchors', [
    ({'x': 0, 'y': 1}, ('left', 'top')),
    ({'x': 0.02, 'y': 0.98}, ('left', 'top')),        # the docstring's own
    ({'x': 1, 'y': 0}, ('right', 'bottom')),
    ({'x': 0.5, 'y': 0.5}, ('center', 'middle')),
    ({'x': 1.05, 'y': 1.1}, ('left', 'bottom')),      # outside: hang away
])
def test_legend_anchor_follows_the_given_position(pos, anchors):
    fig = hyp.plot([_walk(), _walk(seed=1)], backend='plotly', show=False,
                   legend=['a', 'b'], legend_kwargs=pos)
    assert (fig.layout.legend.xanchor, fig.layout.legend.yanchor) == anchors


def test_explicit_legend_anchor_is_kept():
    fig = hyp.plot([_walk(), _walk(seed=1)], backend='plotly', show=False,
                   legend=['a', 'b'],
                   legend_kwargs={'x': 0, 'y': 1, 'yanchor': 'bottom'})
    assert fig.layout.legend.yanchor == 'bottom'
    assert fig.layout.legend.xanchor == 'left'


def test_default_legend_keeps_its_outside_right_anchor():
    fig = hyp.plot([_walk(), _walk(seed=1)], backend='plotly', show=False,
                   legend=['a', 'b'])
    assert (fig.layout.legend.x, fig.layout.legend.y) == (1.02, 0.5)
    assert (fig.layout.legend.xanchor, fig.layout.legend.yanchor) == \
        ('left', 'middle')


def test_top_left_legend_renders_inside_the_figure():
    """Pixels: with x=0, y=1 the legend's text sits BELOW the top edge of
    the plotting area (it used to straddle it, half clipped above)."""
    fig = hyp.plot([_walk(d=2), _walk(d=2, seed=1)], backend='plotly',
                   show=False, legend=['alpha', 'beta'],
                   legend_kwargs={'x': 0, 'y': 1, 'bgcolor': 'yellow'})
    img = _png(fig)
    yellow = (img[..., 0] > 240) & (img[..., 1] > 240) & (img[..., 2] < 40)
    rows = np.flatnonzero(yellow.any(axis=1))
    top_px = fig.layout.margin.t
    assert rows.size and rows.min() >= top_px - 1


# --------------------------------------------------------------------------
# '^' in 3-D: documented mapping


def test_triangle_marker_maps_to_diamond_in_3d_and_is_documented():
    fig = hyp.plot(_walk(), '^', backend='plotly', show=False)
    tr = [t for t in fig.data
          if (t.meta or {}).get('hyp_trace_index') == 0][0]
    assert tr.marker.symbol == 'diamond'
    fig2 = hyp.plot(_walk(d=2), '^', backend='plotly', show=False)
    tr2 = [t for t in fig2.data
           if (t.meta or {}).get('hyp_trace_index') == 0][0]
    assert tr2.marker.symbol == 'triangle-up'
    assert "triangles (``'^'``" in hyp.plot.__doc__
