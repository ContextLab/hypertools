# -*- coding: utf-8 -*-
"""Plotly 3-D data lines render at the width asked for, and an animation's
default width is the documented 1 pt.

1.1 release review (L1): plotly's WebGL (Scatter3d) line renderer draws a
line at HALF its requested width -- measured in kaleido (2026-09-11) as
exactly 0.50x from 1.4 to 12 px, at device scale 1 and 2, while the SVG 2-D
line draws the width asked for -- so every 3-D data line was about half as
thick as the same line in 2-D (and as matplotlib's). And `plot()` documents
the default ``linewidth`` as 1.5 for static plots and 1 for animations;
matplotlib animates at 1 pt, plotly animated at 1.5.

Rendered with kaleido; widths are measured as ink area / stroke length.
"""

import io

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools.plot.plotly_backend import PT_TO_PX

pytest.importorskip('plotly')


def _straight(ndims):
    t = np.linspace(-1, 1, 40)
    cols = [t, 0.5 * t] + ([0.3 * t] if ndims == 3 else [])
    return np.column_stack(cols)


def _stroke_width(fig, width=700, height=560):
    """Effective rendered stroke width (px) of the one dark data line: the
    summed ink coverage divided by the stroke's length (its extent along
    the principal axis of the inked pixels)."""
    from PIL import Image
    fig = type(fig)(fig)
    # the frame (cube/square) is drawn in black too; paint the data line
    # red and measure only red ink
    img = np.asarray(Image.open(io.BytesIO(fig.to_image(
        format='png', width=width, height=height))).convert('RGB')).astype(
            float) / 255.0
    red_ink = np.clip(img[..., 0] - np.maximum(img[..., 1], img[..., 2]),
                      0, 1)
    # coverage of pure red over white: 1 - G (== 1 - B)
    cover = np.where(red_ink > 0.02, 1 - img[..., 1], 0.0)
    ys, xs = np.nonzero(cover > 0.3)
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    along = pts @ vt[0]
    length = along.max() - along.min()
    return cover.sum() / length


def _plot(ndims, **kw):
    return hyp.plot(_straight(ndims), backend='plotly', show=False,
                    color='red', reduce=None, **kw)


@pytest.mark.parametrize('lw', [1.5, 4])
def test_3d_line_renders_as_wide_as_the_same_2d_line(lw):
    w3 = _stroke_width(_plot(3, linewidth=lw))
    w2 = _stroke_width(_plot(2, linewidth=lw))
    requested = lw * PT_TO_PX
    assert w2 == pytest.approx(requested, rel=0.15)
    assert w3 == pytest.approx(requested, rel=0.15), (w3, w2, requested)


def test_3d_trace_requests_the_gl_compensated_width():
    from hypertools.plot.plotly_backend import _GL_LINE_WIDTH_BOOST
    fig = _plot(3, linewidth=2)
    tr = [t for t in fig.data
          if (t.meta or {}).get('hyp_trace_index') == 0][0]
    assert tr.line.width == pytest.approx(2 * PT_TO_PX * _GL_LINE_WIDTH_BOOST)
    fig2 = _plot(2, linewidth=2)
    tr2 = [t for t in fig2.data
           if (t.meta or {}).get('hyp_trace_index') == 0][0]
    assert tr2.line.width == pytest.approx(2 * PT_TO_PX)


@pytest.mark.parametrize('ndims', [2, 3])
def test_animation_default_linewidth_is_the_documented_one_point(ndims):
    from hypertools.plot.plotly_backend import _GL_LINE_WIDTH_BOOST
    boost = _GL_LINE_WIDTH_BOOST if ndims == 3 else 1.0
    anim = _plot(ndims, animate=True, duration=1, frame_rate=4)
    static = _plot(ndims)

    def width(fig):
        return [t for t in fig.data
                if (t.meta or {}).get('hyp_trace_index') == 0][0].line.width

    assert width(anim) == pytest.approx(1.0 * PT_TO_PX * boost)
    assert width(static) == pytest.approx(1.5 * PT_TO_PX * boost)
    # an explicit linewidth= still wins in an animation
    assert width(_plot(ndims, animate=True, duration=1, frame_rate=4,
                       linewidth=3)) == pytest.approx(3 * PT_TO_PX * boost)
    # matplotlib animates at the same documented 1 pt
    mpl = hyp.plot(_straight(ndims), backend='matplotlib', show=False,
                   reduce=None, animate=True, duration=1, frame_rate=4)
    ax = mpl.figure.axes[0]
    assert ax.lines[0].get_linewidth() == pytest.approx(1.0)
    assert '1 for animations' in hyp.plot.__doc__


def test_trails_and_forecasts_get_the_same_gl_compensation():
    from hypertools.plot.plotly_backend import _GL_LINE_WIDTH_BOOST
    x = np.cumsum(np.random.default_rng(0).standard_normal((30, 3)), 0)
    fig = hyp.plot(x, backend='plotly', show=False, animate=True,
                   chemtrails=True, duration=1, frame_rate=4, linewidth=2)
    trail = [t for t in fig.data if (t.meta or {}).get('hyp_trail_index') == 0]
    assert trail[0].line.width == pytest.approx(
        2 * PT_TO_PX * _GL_LINE_WIDTH_BOOST)
    fc = hyp.plot(x, backend='plotly', show=False, predict='Kalman', t=4,
                  linewidth=2)
    static_fc = [t for t in fc.data
                 if (t.meta or {}).get('hyp_forecast_role') == 'static']
    assert static_fc[0].line.width == pytest.approx(
        2 * PT_TO_PX * _GL_LINE_WIDTH_BOOST)
