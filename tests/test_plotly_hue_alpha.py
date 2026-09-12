# -*- coding: utf-8 -*-
"""A continuous `hue=` honours `alpha=` on plotly, markers included, and a
translucent Scatter3d keeps its hue.

1.1 release review: (a) plotly's continuous-hue MARKERS ignored `alpha=`
(opaque ``rgb(...)`` colours, no opacity) although a single-coloured trace's
`alpha=` dims its markers; (b) Plotly's WebGL path composites an ``rgba``
colour additively, so a 3-D hue line at alpha 0.5 drawn with opaque markers
('o-') rendered steelblue as a pale cyan (197, 255, 255) instead of
(162, 192, 217) -- the earlier fix only handled a trace whose alpha was
uniform. Real figures, kaleido-rendered pixels.
"""

import io

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools.plot.plotly_backend import _normalize_scatter3d_alpha

go = pytest.importorskip('plotly.graph_objects')

STEELBLUE = np.array([70, 130, 180])


def _over_white(rgb, alpha):
    return alpha * np.asarray(rgb, float) + (1 - alpha) * 255.0


def _png(fig, width=700, height=500):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(
        fig.to_image(format='png', width=width, height=height)))
        .convert('RGB')).astype(int)


def _colored_pixels(img):
    flat = img.reshape(-1, 3)
    return flat[(flat.sum(1) < 740)
                & ~((flat[:, 0] == flat[:, 1]) & (flat[:, 1] == flat[:, 2]))]


def _dominant(img):
    px = _colored_pixels(img)
    vals, counts = np.unique(px, axis=0, return_counts=True)
    return vals[np.argmax(counts)]


def _data_trace(fig):
    return [t for t in fig.data
            if (t.meta or {}).get('hyp_trace_index') is not None][0]


def _walk(n=60, seed=1):
    return np.cumsum(np.random.default_rng(seed).standard_normal((n, 3)), 0)


@pytest.mark.parametrize('ndims', [1, 2, 3])
def test_continuous_hue_markers_honour_alpha(ndims):
    x = _walk()[:, :ndims] if ndims > 1 else _walk()[:, 0]
    fig = hyp.plot(x, backend='plotly', show=False, fmt='o',
                   hue=np.linspace(0, 1, 60), alpha=0.7, palette='viridis')
    tr = _data_trace(fig)
    colors = list(tr.marker.color)
    if ndims >= 3:
        # uniform alpha -> WebGL's native opacity, colours made opaque
        assert tr.opacity == pytest.approx(0.7)
        assert all(c.startswith('rgb(') for c in colors)
    else:
        assert all(c.startswith('rgba(') and c.endswith(',0.7)')
                   for c in colors)


def test_continuous_hue_markers_render_translucent_2d():
    """Pixels: one steelblue hue marker at alpha=.5 renders as steelblue
    over white at half opacity, as a single-coloured marker does."""
    x = np.column_stack([np.linspace(-1, 1, 5), np.zeros(5)])
    fig = hyp.plot(x, backend='plotly', show=False, fmt='o', reduce=None,
                   hue=np.linspace(0, 1, 5), alpha=0.5,
                   palette=['steelblue', 'steelblue'], markersize=20)
    got = _dominant(_png(fig))
    assert np.abs(got - _over_white(STEELBLUE, 0.5)).max() <= 3, got


@pytest.mark.parametrize('fmt', ['-', 'o-', 'o'])
def test_scatter3d_hue_alpha_keeps_its_hue(fmt):
    """The reviewer's case: continuous hue + alpha=.5 in 3-D. Every fmt
    renders steelblue at half opacity over white, never the additive cyan
    (197, 255, 255)."""
    fig = hyp.plot(_walk(), backend='plotly', show=False,
                   hue=np.linspace(0, 1, 60), alpha=0.5, fmt=fmt,
                   palette=['steelblue', 'steelblue'], linewidth=6,
                   markersize=4, antialias=False)
    got = _dominant(_png(fig))
    assert np.abs(got - _over_white(STEELBLUE, 0.5)).max() <= 4, got


def _mixed_trace():
    z = np.linspace(-1, 1, 40)
    return go.Scatter3d(
        x=np.sin(3 * z), y=np.cos(3 * z), z=z, mode='lines+markers',
        line=dict(color='rgba(70,130,180,0.5)', width=10),
        marker=dict(color='rgb(70,130,180)', size=2))


def test_nonuniform_alpha_is_blended_toward_white():
    """A translucent line with opaque markers cannot share one trace
    opacity: the line is pre-blended toward the white paper instead."""
    tr = _mixed_trace()
    _normalize_scatter3d_alpha(tr)
    assert tr.opacity == 1
    assert tr.line.color == 'rgb(162,192,218)'
    assert tr.marker.color == 'rgb(70,130,180)'
    # idempotent
    _normalize_scatter3d_alpha(tr)
    assert tr.line.color == 'rgb(162,192,218)' and tr.opacity == 1


def test_nonuniform_alpha_renders_its_hue():
    """Pixels: before, the mixed-alpha trace rendered additively (cyan);
    normalized, the thick line is steelblue at half opacity over white."""
    raw = go.Figure(_mixed_trace())
    fixed_trace = _mixed_trace()
    _normalize_scatter3d_alpha(fixed_trace)
    fixed = go.Figure(fixed_trace)
    for fig in (raw, fixed):
        fig.update_layout(paper_bgcolor='white', showlegend=False,
                          scene=dict(xaxis_visible=False, yaxis_visible=False,
                                     zaxis_visible=False))
    want = _over_white(STEELBLUE, 0.5)
    assert np.abs(_dominant(_png(raw)) - want).max() > 20   # the bug
    assert np.abs(_dominant(_png(fixed)) - want).max() <= 4


def test_partial_alpha_keeps_translucency_share():
    """Line .3 and markers .6: opacity .6, line pre-blended by .3/.6, so
    over white each part composites to its own requested alpha."""
    tr = _mixed_trace()
    tr.line.color = 'rgba(70,130,180,0.3)'
    tr.marker.color = 'rgba(70,130,180,0.6)'
    _normalize_scatter3d_alpha(tr)
    assert tr.opacity == pytest.approx(0.6)
    line = np.array([float(v) for v in tr.line.color[4:-1].split(',')])
    composed = 0.6 * line + 0.4 * 255
    np.testing.assert_allclose(composed, _over_white(STEELBLUE, 0.3),
                               atol=1)
