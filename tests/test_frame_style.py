# -*- coding: utf-8 -*-
"""The 2-D square frame and 3-D wireframe cube must look like the same toolkit:
matching outline weight, on both backends (maintainer report, Andy -- the 2-D
box's outline rendered visibly heavier than the 3-D box's).

matplotlib renders both frames at the same width already (both default to
linewidth 1 -> ~2px). plotly did NOT: the 2-D square is an SVG `shape` (honors
its stroke width) but the 3-D cube is a `Scatter3d` line, which plotly's gl line
renderer draws at ~0.6x the requested width -- so at the same requested width the
cube came out ~1px while the square came out ~2px. The cube's requested width is
now boosted so both render at the same ~2px.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import plotly.graph_objects as go

import hypertools as hyp
from hypertools.plot import plotly_backend as pb


def _pts(n=40, seed=0):
    return np.cumsum(np.random.default_rng(seed).standard_normal((n, 3)), axis=0)


def test_plotly_cube_width_is_boosted_over_the_square():
    # both frames derive from the SAME base thickness (CUBE_LINEWIDTH_PT); the
    # cube requests a boosted width purely to counter gl under-rendering, so the
    # two render at matching on-screen weight
    cube = pb._cube_trace(go)
    square = pb._square_shape()
    assert pb._CUBE_GL_WIDTH_BOOST > 1
    assert cube.line.width == pytest.approx(
        square['line']['width'] * pb._CUBE_GL_WIDTH_BOOST)


def test_plotly_cube_and_square_share_the_base_linewidth():
    # a future edit to CUBE_LINEWIDTH_PT must move BOTH frames together
    cube = pb._cube_trace(go, linewidth_pt=3.0)
    square = pb._square_shape(linewidth_pt=3.0)
    base_cube = pb._cube_trace(go, linewidth_pt=1.0).line.width
    base_square = pb._square_shape(linewidth_pt=1.0)['line']['width']
    assert cube.line.width == pytest.approx(3.0 * base_cube)
    assert square['line']['width'] == pytest.approx(3.0 * base_square)


def test_plotly_2d_and_3d_frames_are_present_and_black():
    fig3 = hyp.plot(_pts(), ndims=3, backend='plotly', show=False)
    cube = [t for t in fig3.data
            if isinstance(t, go.Scatter3d) and t.mode == 'lines'
            and t.line.color == 'black']
    assert len(cube) == 1
    fig2 = hyp.plot(_pts(), ndims=2, backend='plotly', show=False)
    squares = [s for s in (fig2.layout.shapes or []) if s.type == 'rect']
    assert len(squares) == 1
    assert squares[0].line.color == 'black'


def test_matplotlib_2d_square_frame_linewidth():
    # the 2-D frame is a fill=False Rectangle patch; it should use the same
    # linewidth (1) as the 3-D cube's wireframe, so the two match (~2px)
    import matplotlib.patches as mpatches
    fig = hyp.plot(_pts(), ndims=2, show=False)
    ax = fig.axes[0]
    rects = [p for p in ax.patches
             if isinstance(p, mpatches.Rectangle) and not p.get_fill()]
    assert rects, 'no square frame patch found'
    assert all(r.get_linewidth() == 1 for r in rects)
