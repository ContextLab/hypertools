"""`hyp.subplots(..., backend='plotly')` + `hyp.plot(..., ax=cell)`: the
plotly form of composing a panel grid from separate calls (backend parity
for the matplotlib ``fig, axes = hyp.subplots(); hyp.plot(d, ax=axes[i])``
loop; 1.1 release review). Real `make_subplots` figures, real plotly
layout objects, real kaleido renders -- no mocks.
"""
import os

import numpy as np
import pytest

import hypertools as hyp

pytest.importorskip('plotly')


def _walk(seed, n=40):
    return hyp.load('random_walk', n_samples=n, n_features=6,
                    random_state=seed)


def test_plotly_subplots_returns_the_grid_figure_and_flat_cells():
    from hypertools.plot.plotly_backend import PlotlyCell
    fig, cells = hyp.subplots(2, 3, backend='plotly')
    assert type(fig).__name__ == 'HyperPlotlyFigure'
    assert cells.shape == (6,)
    assert all(isinstance(c, PlotlyCell) for c in cells)
    assert [(c.row, c.col, c.index) for c in cells] == [
        (1, 1, 0), (1, 2, 1), (1, 3, 2), (2, 1, 3), (2, 2, 4), (2, 3, 5)]
    assert all(c.figure is fig for c in cells)
    # 3-D grid: one scene per cell, already laid out
    for key in ('scene', 'scene2', 'scene3', 'scene4', 'scene5', 'scene6'):
        assert fig.layout[key].domain is not None


def test_plotly_subplots_1x1_still_returns_an_array():
    fig, cells = hyp.subplots(backend='plotly')
    assert cells.shape == (1,)
    assert cells[0].index == 0


def test_plotly_subplots_ndims_2_gives_xy_cells():
    fig, cells = hyp.subplots(1, 2, ndims=2, backend='plotly')
    assert fig.layout.xaxis.domain is not None
    assert fig.layout.xaxis2.domain is not None
    assert fig.layout.scene.to_plotly_json() == {}     # no 3-D cell
    assert cells[1].ndims == 2


def test_plotly_subplots_size_sets_pixels():
    fig, _ = hyp.subplots(1, 2, size=[8, 4], backend='plotly')
    assert (fig.layout.width, fig.layout.height) == (800, 400)


def test_plotly_subplots_rejects_bad_ndims():
    with pytest.raises(ValueError, match='ndims must be 1, 2 or 3'):
        hyp.subplots(1, 1, ndims=4, backend='plotly')


def test_plotly_subplots_forwards_make_subplots_kwargs():
    fig, _ = hyp.subplots(2, 1, backend='plotly', vertical_spacing=0.3)
    y_top_of_lower = fig.layout.scene2.domain.y[1]
    y_bottom_of_upper = fig.layout.scene.domain.y[0]
    assert y_bottom_of_upper - y_top_of_lower == pytest.approx(0.3)


def test_ax_cell_moves_the_whole_panel_into_its_cell_3d():
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    out = hyp.plot(_walk(0), ax=cells[0], title='PCA', legend=True,
                   names=['walk'], backend='plotly', show=False)
    out2 = hyp.plot(_walk(1), ax=cells[1], reduce='PCA', title='again',
                    backend='plotly', show=False)
    assert out is fig and out2 is fig
    # traces landed in their own scenes...
    scenes = {trace.scene for trace in fig.data}
    assert scenes == {'scene', 'scene2'}
    # ...each cell's scene carries the drawn cube's axis ranges/camera
    assert fig.layout.scene.camera.eye is not None
    assert fig.layout.scene2.xaxis.range is not None
    # the titles are per-cell annotations, and the legend is the first
    # cell's own (no entry for the second cell, which has no name)
    assert [a.text for a in fig.layout.annotations] == ['PCA', 'again']
    assert all(t.legend == 'legend' for t in fig.data if t.scene == 'scene')
    assert all(t.legend == 'legend2' for t in fig.data
               if t.scene == 'scene2')
    assert fig.layout.margin.t >= 40


def test_ax_cell_2d_keeps_the_frame_labels_and_colorbar():
    fig, cells = hyp.subplots(1, 2, ndims=2, backend='plotly')
    hyp.plot(_walk(0)[:, :2], ax=cells[0], ndims=2, reduce=None,
             xlabel='a', ylabel='b', backend='plotly', show=False)
    hyp.plot(_walk(1), ax=cells[1], ndims=2, hue=np.arange(40.0),
             colorbar=True, backend='plotly', show=False)
    assert list(fig.layout.xaxis.range) == [-1.1, 1.1]
    assert list(fig.layout.yaxis2.range) == [-1.1, 1.1]
    assert fig.layout.xaxis.title.text == 'a'
    assert fig.layout.yaxis.title.text == 'b'
    # one frame square per cell, on that cell's axes
    assert sorted(shape.xref for shape in fig.layout.shapes) == ['x', 'x2']
    colorbars = [t.marker.colorbar for t in fig.data
                 if t.marker is not None and t.marker.showscale]
    assert len(colorbars) == 1
    x1 = fig.layout.xaxis2.domain[1]
    assert colorbars[0].x > x1


def test_ax_cell_refuses_animate_and_the_matplotlib_backend():
    fig, cells = hyp.subplots(1, 1, backend='plotly')
    with pytest.raises(ValueError, match='cannot be combined with animate'):
        hyp.plot(_walk(0), ax=cells[0], animate=True, backend='plotly',
                 show=False)
    with pytest.raises(TypeError, match='draws with matplotlib'):
        hyp.plot(_walk(0), ax=cells[0], backend='matplotlib', show=False)


def test_ax_cell_grid_renders_to_png(tmp_path):
    fig, cells = hyp.subplots(1, 2, backend='plotly')
    for cell, seed in zip(cells, (0, 1)):
        hyp.plot(_walk(seed), ax=cell, title=f'walk {seed}',
                 backend='plotly', show=False)
    target = tmp_path / 'grid.png'
    fig.write_image(str(target))
    assert os.path.getsize(target) > 0


def test_ax_cell_queues_the_grid_for_display_once():
    """Three cell calls in one notebook cell must display the grid once
    (matplotlib's inline hook shows a grid figure once too). Exercised on
    a real IPython InteractiveShell, whose `post_execute` queue
    `_display_at_cell_end` fills."""
    from IPython.core.interactiveshell import InteractiveShell
    from hypertools.plot import plotly_backend as pb
    shell = InteractiveShell.instance()
    try:
        fig, cells = hyp.subplots(1, 3, backend='plotly')
        pb._PENDING_DISPLAY[:] = []
        for _ in cells:
            pb._display_at_cell_end(fig)
        assert len(pb._PENDING_DISPLAY) == 1
        assert pb._PENDING_DISPLAY[0] is fig
    finally:
        pb._PENDING_DISPLAY[:] = []
        try:
            shell.events.unregister('post_execute', pb._flush_pending_display)
        except ValueError:
            pass
        InteractiveShell.clear_instance()


def test_drawing_into_a_cell_twice_keeps_the_earlier_labels():
    fig, cells = hyp.subplots(1, 1, backend='plotly')
    hyp.plot(_walk(0), ax=cells[0], labels=['first'], label_anchor='first',
             backend='plotly', show=False)
    hyp.plot(_walk(1), ax=cells[0], labels=['second'], label_anchor='first',
             backend='plotly', show=False)
    assert len(fig.data) >= 2
    texts = [a.text for a in fig.layout.scene.annotations]
    assert texts == ['first', 'second']
