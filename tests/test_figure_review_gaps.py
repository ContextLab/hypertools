"""Gaps found by the three-role figure review of the 1.1 feature tour
(2026-09-07: one agent wrote each figure's EXPECTED appearance from the
notebook's code and prose, one described the OBSERVED render, one
adjudicated), each pinned here at the public API on both backends.

G1  a caller's matplotlib axes (`ax=`, every `panels=` cell) took its
    figure's default colour cycle instead of the palette; a second call
    into the same axes/figure restarted the palette on both backends.
G2  `panels=` at the default size shrank its cells to fit per-panel legends.
G3  a 3-D figure's axis labels fell outside its tight bbox.
G5  a recoloured forecast was still faded and vanished among translucent
    traces.
G7  the 2-D frame square sat on the data's extreme points.
G8  `legend_kwargs={'loc': ...}` kept the outside-right anchor.
G10 plotly legend keys reproduced a 2 px '.' marker.
No mocks: every assertion reads the drawn artists, traces or pixels.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from tests._plotly_colors import rgba as effective_rgba
import pytest
from matplotlib.colors import to_rgb

import hypertools as hyp
from hypertools._shared.helpers import UNIT_FRAME_LIMIT, UNIT_FRAME_SCALE


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _walks(n=2, rows=40):
    return [hyp.load('random_walk', n_samples=rows, n_features=3,
                     random_state=i) for i in range(n)]


def _data_lines(ax):
    return [ln for ln in ax.lines
            if getattr(ln, '_hyp_forecast_role', None) is None]


def _own_colors(n):
    fig = hyp.plot(_walks(n), show=False)
    return [to_rgb(ln.get_color()) for ln in _data_lines(fig.axes[0])[:n]]


# --- G1: palette on a caller's axes, continued across calls ---------------

def test_caller_axes_draw_in_the_palette():
    fig, axes = hyp.subplots(1, 2)
    hyp.plot(_walks(2), ax=axes[0], show=False)
    drawn = [to_rgb(ln.get_color()) for ln in _data_lines(axes[0])[:2]]
    assert drawn == _own_colors(2)
    assert drawn[0] != to_rgb('C0')


def test_panels_draw_in_the_palette():
    fig = hyp.plot(_walks(3), panels=True, show=False)
    first = _own_colors(1)[0]
    for ax in fig.axes[:3]:
        assert to_rgb(_data_lines(ax)[0].get_color()) == first


def test_explicit_palette_reaches_a_caller_axes():
    fig, axes = hyp.subplots(1, 1)
    hyp.plot(_walks(2), ax=axes[0], palette=['navy', 'gold'], show=False)
    drawn = [to_rgb(ln.get_color()) for ln in _data_lines(axes[0])[:2]]
    assert drawn == [to_rgb('navy'), to_rgb('gold')]


def test_a_second_call_into_the_same_axes_continues_the_palette():
    fig, axes = hyp.subplots(1, 1)
    hyp.plot(_walks(1), ax=axes[0], show=False)
    hyp.plot(_walks(1), ax=axes[0], show=False)
    first, second = [to_rgb(ln.get_color()) for ln in _data_lines(axes[0])]
    assert first != second
    # the pair is what one call with both datasets draws
    assert [first, second] == _own_colors(2)


def test_a_second_call_into_the_same_plotly_figure_continues_the_palette():
    fig = hyp.plot(_walks(1), show=False, backend='plotly')
    fig = hyp.plot(_walks(1), ax=fig, show=False, backend='plotly')
    data = [tr for tr in fig.data
            if (tr.meta or {}).get('hyp_trace_index') is not None]
    assert len(data) == 2
    assert data[0].line.color != data[1].line.color
    both = hyp.plot(_walks(2), show=False, backend='plotly')
    expected = [tr.line.color for tr in both.data
                if (tr.meta or {}).get('hyp_trace_index') is not None]
    assert [tr.line.color for tr in data] == expected


# --- G2: panel figures make room for their legends ------------------------

def test_default_size_panels_widen_for_per_panel_legends():
    plain = hyp.plot(_walks(3), panels=True, show=False)
    with_legend = hyp.plot(_walks(3), panels=True, legend=True, show=False)
    assert with_legend.get_size_inches()[0] > plain.get_size_inches()[0]
    # ...so the panels themselves stay as large as without a legend
    def cell_in(fig):
        ax = fig.axes[0]
        return ax.get_position().width * fig.get_size_inches()[0]
    assert cell_in(with_legend) >= 0.9 * cell_in(plain)
    sized = hyp.plot(_walks(3), panels=True, legend=True, size=[9, 3],
                     show=False)
    assert tuple(sized.get_size_inches()) == (9.0, 3.0)


# --- G3: 3-D axis labels are inside the tight bbox ------------------------

def test_three_d_axis_labels_are_inside_the_tight_bbox(tmp_path):
    fig = hyp.plot(_walks(1)[0], xlabel='PC 1', ylabel='PC 2',
                   zlabel='PC 3', show=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)          # inches
    dpi = fig.dpi
    for axis in fig.axes[0]._axis_map.values():
        box = axis.label.get_window_extent(renderer)   # pixels
        # every label box lies inside the tight bbox on all four sides
        assert tight.x0 * dpi - 1e-6 <= box.x0
        assert box.x1 <= tight.x1 * dpi + 1e-6
        assert tight.y0 * dpi - 1e-6 <= box.y0
        assert box.y1 <= tight.y1 * dpi + 1e-6
    # and a bbox-tight save keeps the z-label's pixels (right of the cube)
    from PIL import Image
    loose = tmp_path / 'loose.png'
    tight_png = tmp_path / 'tight.png'
    fig.savefig(loose, dpi=100)
    fig.savefig(tight_png, dpi=100, bbox_inches='tight')
    zbox = fig.axes[0].zaxis.label.get_window_extent(renderer)
    img = np.asarray(Image.open(tight_png).convert('L'))
    # the tight image must be at least as wide as the label's right edge
    # minus the tight bbox's left edge
    assert img.shape[1] >= int(zbox.x1 - tight.x0 * fig.dpi) - 2


# --- G5: a recoloured forecast keeps its trace's alpha --------------------

@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_recoloured_forecasts_are_not_faded(backend):
    kw = dict(predict='Kalman', t=4, alpha=0.7, show=False, backend=backend)
    inherited = hyp.plot(_walks(2), **kw)
    recoloured = hyp.plot(_walks(2), forecast_palette=['black', 'orange'],
                          **kw)
    if backend == 'matplotlib':
        fc = [ln for ln in inherited.axes[0].lines
              if getattr(ln, '_hyp_forecast_role', None) == 'static']
        assert fc[0].get_alpha() == pytest.approx(0.35)
        fc = [ln for ln in recoloured.axes[0].lines
              if getattr(ln, '_hyp_forecast_role', None) == 'static']
        assert fc[0].get_alpha() == pytest.approx(0.7)
        assert to_rgb(fc[0].get_color()) == to_rgb('black')
    else:
        def alpha(fig):
            tr = [t for t in fig.data
                  if (t.meta or {}).get('hyp_forecast_role') == 'static'][0]
            return tr.meta['hyp_forecast_alpha'], effective_rgba(tr)[-1]
        assert alpha(inherited)[0] == pytest.approx(0.35)
        a, color = alpha(recoloured)
        assert a == pytest.approx(0.7)
        assert color == pytest.approx(.7)


def test_a_dash_only_override_keeps_the_fade():
    fig = hyp.plot(_walks(2), predict='Kalman', t=4, forecast_fmt=':',
                   show=False)
    fc = [ln for ln in fig.axes[0].lines
          if getattr(ln, '_hyp_forecast_role', None) == 'static']
    assert fc[0].get_alpha() == pytest.approx(0.5)


# --- G7: the 2-D frame square clears the data -----------------------------

def test_two_d_frame_square_clears_the_data_matplotlib():
    fig = hyp.plot(_walks(2), ndims=2, show=False)
    ax = fig.axes[0]
    square = [p for p in ax.patches if p.get_width() > 2][0]
    assert square.get_width() == pytest.approx(2 * UNIT_FRAME_SCALE)
    assert UNIT_FRAME_SCALE > 1.0
    xs = np.concatenate([ln.get_xdata() for ln in _data_lines(ax)])
    ys = np.concatenate([ln.get_ydata() for ln in _data_lines(ax)])
    assert max(abs(xs).max(), abs(ys).max()) <= 1.0 + 1e-9
    assert ax.get_xlim() == (-UNIT_FRAME_LIMIT, UNIT_FRAME_LIMIT)


def test_two_d_frame_square_clears_the_data_plotly():
    fig = hyp.plot(_walks(2), ndims=2, show=False, backend='plotly')
    square = fig.layout.shapes[0]
    assert square.x1 == pytest.approx(UNIT_FRAME_SCALE)
    assert list(fig.layout.xaxis.range) == [-UNIT_FRAME_LIMIT,
                                            UNIT_FRAME_LIMIT]
    xs = np.concatenate([np.asarray(tr.x, dtype=float) for tr in fig.data
                         if (tr.meta or {}).get('hyp_trace_index')
                         is not None])
    assert abs(xs).max() <= 1.0 + 1e-9


# --- G8: legend_kwargs loc= places the legend on the axes -----------------

def test_legend_kwargs_loc_drops_the_outside_anchor():
    fig = hyp.plot(_walks(2), legend=True,
                   legend_kwargs={'loc': 'upper left'}, show=False)
    fig.canvas.draw()
    legend = fig.axes[0].get_legend()
    box = legend.get_window_extent(fig.canvas.get_renderer())
    axes_box = fig.axes[0].get_window_extent(fig.canvas.get_renderer())
    # inside the axes' left half and upper half
    assert box.x0 >= axes_box.x0 - 1 and box.x1 <= axes_box.x0 + axes_box.width / 2
    assert box.y1 <= axes_box.y1 + 1 and box.y0 >= axes_box.y0 + axes_box.height / 2
    default = hyp.plot(_walks(2), legend=True, show=False)
    default.canvas.draw()
    dbox = default.axes[0].get_legend().get_window_extent(
        default.canvas.get_renderer())
    daxes = default.axes[0].get_window_extent(default.canvas.get_renderer())
    assert dbox.x0 >= daxes.x1  # the default stays outside right


# --- G10: plotly legend keys are readable for tiny markers -----------------

def test_plotly_legend_keys_use_a_constant_item_size():
    fig = hyp.plot(_walks(2), '.', legend=True, show=False, backend='plotly')
    assert fig.layout.legend.itemsizing == 'constant'
    grid = hyp.plot(_walks(2), '.', panels=True, legend=True, show=False,
                    backend='plotly')
    assert grid.layout.legend.itemsizing == 'constant'
    assert grid.layout.legend2.itemsizing == 'constant'


# --- G9: a plotly hyp.subplots grid grows its gutters only when needed ---

def _plotly_cells_px(fig):
    from hypertools.plot.plotly_backend import cell_layout_keys
    mg = fig.layout.margin
    plot_w = fig.layout.width - mg.l - mg.r
    out = []
    for i in range(2):
        d = fig.layout[cell_layout_keys(i)['scene']].domain
        out.append((mg.l + d.x[0] * plot_w, mg.l + d.x[1] * plot_w))
    return out


def test_plotly_subplots_grid_starts_without_gutters():
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    hyp.plot(_walks(1)[0], ax=cells[0], show=False, backend='plotly')
    hyp.plot(_walks(1)[0], ax=cells[1], show=False, backend='plotly')
    grid = hyp.plot(_walks(2), panels=True, show=False, backend='plotly')
    # the same cells as a legend-less panels= grid draws
    assert fig.layout.width == grid.layout.width
    assert _plotly_cells_px(fig) == pytest.approx(_plotly_cells_px(grid),
                                                  abs=1.0)
    assert fig.layout.margin.r == grid.layout.margin.r


def test_plotly_subplots_grid_grows_gutters_for_a_legend():
    from hypertools.plot.plotly_backend import PANEL_GUTTER_PAD_PX
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    hyp.plot(_walks(1)[0], ax=cells[0], title='first', show=False,
             backend='plotly')
    before = _plotly_cells_px(fig)
    width_before = fig.layout.width
    hyp.plot(_walks(2), ax=cells[1], legend=True, names=['a', 'b'],
             show=False, backend='plotly')
    after = _plotly_cells_px(fig)
    # the default-size figure widened by one gutter per column, each
    # cell kept its width, and cell 0 (drawn earlier) moved with the grid
    assert fig.layout.width > width_before
    for (b0, b1), (a0, a1) in zip(before, after):
        assert (a1 - a0) == pytest.approx(b1 - b0, abs=1.0)
    assert after[1][0] - after[0][1] > before[1][0] - before[0][1]
    # cell 1's legend sits just right of cell 1, inside the figure
    mg = fig.layout.margin
    plot_w = fig.layout.width - mg.l - mg.r
    legend_x = mg.l + fig.layout.legend2.x * plot_w
    assert legend_x == pytest.approx(after[1][1] + PANEL_GUTTER_PAD_PX,
                                     abs=1.0)
    assert legend_x < fig.layout.width
    # cell 0's title followed its cell
    title = [a for a in fig.layout.annotations
             if a.name == 'hyp-cell-title-0'][0]
    assert mg.l + title.x * plot_w == pytest.approx(
        0.5 * (after[0][0] + after[0][1]), abs=1.0)
    assert fig.layout.meta['hyp_grid']['gutter_px'] > 0
