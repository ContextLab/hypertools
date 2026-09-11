# -*- coding: utf-8 -*-
"""Plotly composition (`ax=` a figure or a `hyp.subplots` cell, and
`panels=`): 1.1 release-review findings.

* a colorbar beside a legend-less cell sat a whole legend's width further
  right -- on top of the next cell -- because every DRAWN cell counted as
  having a legend;
* a second call into a cell deleted the cell's earlier title (matplotlib
  keeps an axes title a later call does not replace);
* data whose dimensionality does not match the cell/figure raised a bare
  ``TypeError: cannot unpack non-iterable NoneType`` (cell) or silently
  overlaid a 2-D trace on a 3-D figure (``ax=<Figure>``);
* drawing into a caller's figure rewrote the caller's OWN traces;
* ``ax=<plotly cell>`` with the default ``backend='auto'`` raised instead of
  drawing with plotly.

Real `make_subplots` figures, real layout objects and kaleido renders.
"""

import io

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools.plot.plotly_backend import PANEL_GUTTER_PAD_PX

go = pytest.importorskip('plotly.graph_objects')


def _walks(k=3, n=40, d=3, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.standard_normal((n, d)), 0) for _ in range(k)]


def _colorbar_traces(fig):
    return [t for t in fig.data
            if t.marker is not None and t.marker.showscale]


def _plot_w(fig):
    return fig.layout.width - fig.layout.margin.l - fig.layout.margin.r


# --------------------------------------------------------------------------
# finding 4: colorbar beside a legend-less cell


def test_subplots_colorbar_cell_without_legend_sits_beside_its_cell():
    X = _walks()
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    hyp.plot(X[0], ax=cells[0], backend='plotly', show=False,
             hue=np.linspace(0, 1, 40), colorbar=True)
    hyp.plot(X[1:], ax=cells[1], backend='plotly', show=False,
             legend=['a', 'b'])
    cb, = _colorbar_traces(fig)
    x1 = fig.layout.scene.domain.x[1]
    next_x0 = fig.layout.scene2.domain.x[0]
    assert cb.marker.colorbar.x == pytest.approx(
        x1 + PANEL_GUTTER_PAD_PX / _plot_w(fig))
    # the whole bar (15 px + its tick labels, ~45 px) clears the next cell
    assert cb.marker.colorbar.x + 60 / _plot_w(fig) < next_x0


def test_panels_colorbars_do_not_land_on_the_next_panel():
    X = _walks(k=2)
    fig = hyp.plot(X, panels=True, backend='plotly', show=False,
                   hue=[np.linspace(0, 1, 40)] * 2, colorbar=True)
    cbs = {t.scene: t.marker.colorbar for t in _colorbar_traces(fig)}
    x1 = fig.layout.scene.domain.x[1]
    assert cbs['scene'].x == pytest.approx(
        x1 + PANEL_GUTTER_PAD_PX / _plot_w(fig))
    assert cbs['scene'].x + 60 / _plot_w(fig) < fig.layout.scene2.domain.x[0]


def test_colorbar_after_a_legend_in_the_same_cell_still_moves_right():
    """The fix must not lose the case the legend push exists for."""
    X = _walks()
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    hyp.plot(X[:2], ax=cells[0], backend='plotly', show=False,
             legend=['a', 'b'])
    hyp.plot(X[2], ax=cells[0], backend='plotly', show=False,
             hue=np.linspace(0, 1, 40), colorbar=True)
    cb, = _colorbar_traces(fig)
    from hypertools.plot.plotly_backend import PANEL_LEGEND_PX
    assert cb.marker.colorbar.x == pytest.approx(
        fig.layout.scene.domain.x[1]
        + (PANEL_GUTTER_PAD_PX + PANEL_LEGEND_PX) / _plot_w(fig))


# --------------------------------------------------------------------------
# finding 6: a later untitled call keeps the cell's title


def _cell_titles(fig):
    return [a.text for a in fig.layout.annotations
            if (a.name or '').startswith('hyp-cell-title-')]


def test_second_untitled_draw_keeps_the_cell_title():
    A, B = _walks(k=2), _walks(k=2, seed=1)
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    hyp.plot(A, ax=cells[0], backend='plotly', show=False,
             title='first title')
    hyp.plot(B, ax=cells[0], backend='plotly', show=False)
    assert _cell_titles(fig) == ['first title']
    assert fig.layout.margin.t >= 40


def test_second_titled_draw_replaces_the_cell_title():
    A, B = _walks(k=2), _walks(k=2, seed=1)
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    hyp.plot(A, ax=cells[0], backend='plotly', show=False, title='first')
    hyp.plot(B, ax=cells[0], backend='plotly', show=False, title='second')
    assert _cell_titles(fig) == ['second']


# --------------------------------------------------------------------------
# finding 7: dimensionality mismatch


@pytest.mark.parametrize('cell_nd, cols, kw', [
    (3, 2, {}),                  # 2-D data into a 3-D cell
    (2, 5, {}),                  # 3-D plot into a 2-D cell
    (2, 5, dict(ndims=3)),
])
def test_cell_dimensionality_mismatch_raises_clearly(cell_nd, cols, kw):
    rng = np.random.default_rng(0)
    data = np.cumsum(rng.standard_normal((40, cols)), 0)
    fig, cells = hyp.subplots(1, 2, ndims=cell_nd, backend='plotly')
    with pytest.raises(ValueError, match=r'ax= is a [23]-D'):
        hyp.plot(data, ax=cells[0], backend='plotly', show=False, **kw)
    # nothing was drawn into the grid
    assert len(fig.data) == 0


def test_figure_dimensionality_mismatch_raises_clearly():
    X3 = _walks(k=1)[0]
    fig = hyp.plot(X3, backend='plotly', show=False)
    n_before = len(fig.data)
    with pytest.raises(ValueError, match=r'ax= is a 3-D plotly figure'):
        hyp.plot(X3[:, :2], backend='plotly', show=False, ax=fig)
    assert len(fig.data) == n_before
    fig2 = hyp.plot(X3[:, :2], backend='plotly', show=False)
    with pytest.raises(ValueError, match=r'ax= is a 2-D plotly figure'):
        hyp.plot(X3, backend='plotly', show=False, ax=fig2)


def test_matching_dimensionality_still_composes():
    X3 = _walks(k=1)[0]
    fig = hyp.plot(X3, backend='plotly', show=False)
    out = hyp.plot(X3 + 1, backend='plotly', show=False, ax=fig)
    assert out is fig
    empty = go.Figure()
    assert hyp.plot(X3[:, :2], backend='plotly', show=False,
                    ax=empty) is empty


# --------------------------------------------------------------------------
# finding 9: the caller's own traces are left alone


def test_drawing_into_a_figure_leaves_the_callers_traces_untouched():
    theirs = go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode='lines',
                          line=dict(color='rgba(255,0,0,0.5)', width=4),
                          name='mine')
    fig = go.Figure(theirs)
    before = fig.data[0].to_plotly_json()
    hyp.plot(_walks(k=1)[0], backend='plotly', show=False, ax=fig,
             alpha=0.5)
    assert fig.data[0].to_plotly_json() == before
    # ...while hypertools' own translucent trace is still normalized
    ours = [t for t in fig.data
            if (t.meta or {}).get('hyp_trace_index') is not None]
    assert ours and ours[0].opacity == pytest.approx(0.5)


# --------------------------------------------------------------------------
# finding B: a plotly ax= implies the plotly backend


def test_plotly_cell_with_default_backend_draws_with_plotly():
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    out = hyp.plot(_walks(k=1)[0], ax=cells[0], show=False)
    assert out is fig
    assert any((t.meta or {}).get('hyp_trace_index') == 0 for t in fig.data)


def test_plotly_figure_with_default_backend_draws_with_plotly():
    fig = hyp.plot(_walks(k=1)[0], backend='plotly', show=False)
    n = len(fig.data)
    out = hyp.plot(_walks(k=1, seed=2)[0], ax=fig, show=False)
    assert out is fig and len(fig.data) > n


def test_plotly_cell_with_explicit_matplotlib_still_raises():
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    with pytest.raises(TypeError, match="pass backend='plotly'"):
        hyp.plot(_walks(k=1)[0], ax=cells[0], show=False,
                 backend='matplotlib')


def test_rendered_colorbar_does_not_overlap_next_cell(tmp_path):
    """Pixels: the colorbar's coloured bar sits between the two cubes, not
    inside the second cell."""
    from PIL import Image
    X = _walks()
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    hyp.plot(X[0], ax=cells[0], backend='plotly', show=False,
             hue=np.linspace(0, 1, 40), colorbar=True, palette='viridis')
    hyp.plot(X[1], ax=cells[1], backend='plotly', show=False)
    img = np.asarray(Image.open(io.BytesIO(fig.to_image(format='png')))
                     .convert('RGB')).astype(int)
    W = fig.layout.width
    x1_px = fig.layout.margin.l + fig.layout.scene.domain.x[1] * _plot_w(fig)
    x0_next = fig.layout.margin.l \
        + fig.layout.scene2.domain.x[0] * _plot_w(fig)
    # saturated (viridis) columns = the colorbar and the hue line
    sat = (img.max(axis=2) - img.min(axis=2)) > 80
    col_counts = sat.sum(axis=0)
    bar_cols = np.flatnonzero(col_counts > 0.3 * fig.layout.height * 0.75
                              * 0.5)
    assert bar_cols.size, 'no colorbar found'
    assert bar_cols.min() >= x1_px - 2
    assert bar_cols.max() < min(x0_next, W)
