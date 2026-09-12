"""Multi-panel static plots: `panels=`/`subplots=` and `hyp.subplots` (GH #285).

Every hand-built panel grid in the examples/tutorials repeats the same five
steps -- `plt.subplots(nrows, ncols, subplot_kw={'projection': '3d'})`,
`ravel()`, a loop of `hyp.plot(x, ax=ax, show=False)`, hide the spares,
`tight_layout()`. `panels=` does all five, and additionally fits the analysis
pipeline ONCE across every dataset so the panels share one set of components.

Real figures, rendered headless; positions and artists are read back off the
returned Figure (no mocks).
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import hypertools as hyp
from hypertools._shared.helpers import UNIT_FRAME_LIMIT
from hypertools.plot.plot import subplots as hyp_subplots


def _datasets(n=3, rows=30, cols=6, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.normal(size=(rows, cols)), axis=0) for _ in range(n)]


# --- hyp.subplots -------------------------------------------------------

def test_subplots_returns_flat_3d_axes():
    fig, axes = hyp_subplots(2, 3)
    try:
        assert axes.shape == (6,)
        assert all(ax.name == '3d' for ax in axes)
        assert all(ax.figure is fig for ax in axes)
    finally:
        matplotlib.pyplot.close(fig)


def test_subplots_1x1_still_returns_an_array():
    fig, axes = hyp_subplots()
    try:
        assert axes.shape == (1,)
        assert axes[0].name == '3d'
    finally:
        matplotlib.pyplot.close(fig)


def test_subplots_ndims_2_gives_rectilinear_axes():
    fig, axes = hyp_subplots(1, 2, ndims=2)
    try:
        assert [ax.name for ax in axes] == ['rectilinear', 'rectilinear']
    finally:
        matplotlib.pyplot.close(fig)


def test_subplots_size_sets_figsize():
    fig, axes = hyp_subplots(1, 2, size=[8, 4])
    try:
        assert tuple(fig.get_size_inches()) == (8.0, 4.0)
    finally:
        matplotlib.pyplot.close(fig)


def test_subplots_rejects_bad_ndims():
    with pytest.raises(ValueError, match='ndims'):
        hyp_subplots(1, 1, ndims=7)


def test_subplots_accepts_matplotlib_figure_kwargs():
    fig, axes = hyp_subplots(1, 2, dpi=123)
    try:
        assert fig.dpi == 123
    finally:
        matplotlib.pyplot.close(fig)


# --- panels= : one panel per dataset ------------------------------------

def test_panels_true_draws_one_axes_per_dataset():
    data = _datasets(3)
    fig = hyp.plot(data, panels=True, reduce='PCA', show=False)
    # 3 datasets -> one row of three (1.1 release review: the grid follows
    # the figure's aspect and avoids a spare cell; it used to be 2x2)
    assert len(fig.axes) == 3
    assert all(ax.get_visible() for ax in fig.axes)
    assert all(ax.name == '3d' for ax in fig.axes)
    # one trajectory drawn per panel
    for ax in fig.axes:
        assert len(ax.lines) == 1


def test_panels_true_hides_the_spare_cell_when_no_grid_fits_exactly():
    data = _datasets(5)
    fig = hyp.plot(data, panels=True, reduce='PCA', show=False)
    # 5 datasets -> 2x3 with one spare, hidden
    assert len(fig.axes) == 6
    assert [ax.get_visible() for ax in fig.axes] == [True] * 5 + [False]
    for ax in fig.axes[:5]:
        assert len(ax.lines) == 1


def test_panels_titles_are_per_panel():
    data = _datasets(3)
    fig = hyp.plot(data, panels=True, title=['a', 'b', 'c'], reduce='PCA',
                   show=False)
    assert [ax.get_title() for ax in fig.axes[:3]] == ['a', 'b', 'c']


def test_panels_single_title_names_every_panel():
    data = _datasets(3)
    fig = hyp.plot(data, panels=(1, 3), title='shared', reduce='PCA',
                   show=False)
    assert [ax.get_title() for ax in fig.axes] == ['shared'] * 3


def test_panels_title_length_mismatch_raises():
    data = _datasets(3)
    with pytest.raises(ValueError, match='one title per panel'):
        hyp.plot(data, panels=True, title=['a', 'b'], reduce='PCA',
                 show=False)


def test_panels_int_is_the_column_count():
    data = _datasets(5)
    fig = hyp.plot(data, panels=2, reduce='PCA', show=False)
    # 5 panels, 2 columns -> 3 rows = 6 cells, 1 hidden
    assert len(fig.axes) == 6
    assert sum(ax.get_visible() for ax in fig.axes) == 5
    positions = [ax.get_position().bounds for ax in fig.axes]
    # two distinct x positions (two columns), three distinct y positions
    assert len({round(p[0], 3) for p in positions}) == 2
    assert len({round(p[1], 3) for p in positions}) == 3


def test_panels_explicit_grid_is_used_verbatim():
    data = _datasets(4)
    fig = hyp.plot(data, panels=(4, 1), reduce='PCA', show=False)
    assert len(fig.axes) == 4
    xs = {round(ax.get_position().bounds[0], 3) for ax in fig.axes}
    assert len(xs) == 1          # one column


def test_panels_grid_too_small_raises():
    data = _datasets(5)
    with pytest.raises(ValueError, match='cells'):
        hyp.plot(data, panels=(2, 2), reduce='PCA', show=False)


def test_panels_ndims_2_makes_2d_panels():
    data = _datasets(3)
    fig = hyp.plot(data, panels=(1, 3), ndims=2, reduce='PCA', show=False)
    assert [ax.name for ax in fig.axes] == ['rectilinear'] * 3


def test_panels_share_one_pipeline_fit():
    """The whole point of `panels=`: every panel is drawn from ONE
    reduction fit across all datasets, so the panels are comparable.

    Verified against the components themselves: the shared fit's per-dataset
    output must match what a single-axes `hyp.plot(..., return_model=True)`
    call produces, and must NOT match independent per-dataset reductions.
    """
    data = _datasets(3, rows=40, cols=8)
    shared = hyp.plot(data, reduce='PCA', return_model=True, show=False)
    joint = [np.asarray(a) for a in shared['xform_data']]

    bundle = hyp.plot(data, panels=True, reduce='PCA', return_model=True,
                      show=False)
    per_panel = [np.asarray(a) for a in bundle['xform_data']]
    assert len(per_panel) == 3
    for got, want in zip(per_panel, joint):
        np.testing.assert_allclose(got, want, atol=1e-10)

    # ... and an independently-fit reduction of dataset 1 alone genuinely
    # differs, so the assertion above is not vacuous
    alone = np.asarray(
        hyp.plot([data[1]], reduce='PCA', return_model=True,
                 show=False)['xform_data'][0])
    assert not np.allclose(alone, joint[1], atol=1e-6)


def test_panels_return_model_carries_axes_and_grid():
    data = _datasets(3)
    bundle = hyp.plot(data, panels=True, reduce='PCA', return_model=True,
                      show=False)
    assert bundle['panels'] == (1, 3)
    assert len(bundle['axes']) == 3
    assert all(ax.figure is bundle['fig'] for ax in bundle['axes'])
    assert len(bundle['panel_models']) == 3
    assert bundle['colors'] is not None


def test_panels_hue_is_narrowed_per_panel():
    data = _datasets(3, rows=20)
    hue = list(np.arange(60, dtype=float))       # flat, per observation
    fig = hyp.plot(data, panels=(1, 3), hue=hue, reduce='PCA', show=False)
    assert len(fig.axes) == 3
    # a continuous hue draws a LineCollection per panel, not a plain line
    from matplotlib.collections import LineCollection
    for ax in fig.axes:
        assert any(isinstance(c, LineCollection)
                   and getattr(c, '_hyp_trace_index', None) is not None
                   for c in ax.collections)


def test_panels_per_dataset_labels_land_in_their_own_panel():
    data = _datasets(3, rows=20)
    fig = hyp.plot(data, panels=(1, 3), labels=['one', 'two', 'three'],
                   reduce='PCA', show=False)
    texts = [[t.get_text() for t in ax.texts] for ax in fig.axes]
    assert texts == [['one'], ['two'], ['three']]


def test_panels_names_are_narrowed_per_panel():
    data = _datasets(3, rows=20)
    fig = hyp.plot(data, panels=(1, 3), names=['a', 'b', 'c'], legend=True,
                   reduce='PCA', show=False)
    got = [[t.get_text() for t in ax.get_legend().get_texts()]
           for ax in fig.axes]
    assert got == [['a'], ['b'], ['c']]


def test_panels_color_list_is_narrowed_per_panel():
    data = _datasets(3, rows=20)
    fig = hyp.plot(data, panels=(1, 3), color=['red', 'green', 'blue'],
                   reduce='PCA', show=False)
    import matplotlib.colors as mcolors
    got = [mcolors.to_hex(ax.lines[0].get_color()) for ax in fig.axes]
    assert got == [mcolors.to_hex(c) for c in ('red', 'green', 'blue')]


# --- panels= : one panel per reducer ------------------------------------

def test_reduce_list_gives_one_panel_per_reducer():
    data = _datasets(2, rows=30, cols=8)
    fig = hyp.plot(data, panels=True, reduce=['PCA', 'FastICA', 'PCA'],
                   title=['pca', 'ica', 'pca again'], show=False)
    assert sum(ax.get_visible() for ax in fig.axes) == 3
    assert [ax.get_title() for ax in fig.axes[:3]] == ['pca', 'ica',
                                                       'pca again']
    # each reducer panel draws EVERY dataset
    for ax in fig.axes[:3]:
        assert len(ax.lines) == 2


def test_reduce_list_panels_actually_differ():
    data = _datasets(2, rows=30, cols=8)
    bundle = hyp.plot(data, panels=(1, 2), reduce=['PCA', 'FastICA'],
                      return_model=True, show=False)
    first = np.asarray(bundle['panel_models'][0]['xform_data'][0])
    second = np.asarray(bundle['panel_models'][1]['xform_data'][0])
    assert first.shape == second.shape
    assert not np.allclose(first, second)


def test_empty_reduce_list_raises():
    with pytest.raises(ValueError, match='empty list'):
        hyp.plot(_datasets(2), panels=True, reduce=[], show=False)


# --- refusals -----------------------------------------------------------

def test_panels_with_animate_raises():
    with pytest.raises(ValueError, match='animate'):
        hyp.plot(_datasets(3), panels=True, animate=True, show=False)


def test_panels_with_ax_raises():
    fig, axes = hyp_subplots(1, 1)
    try:
        with pytest.raises(ValueError, match='ax='):
            hyp.plot(_datasets(3), panels=True, ax=axes[0], show=False)
    finally:
        matplotlib.pyplot.close(fig)


def test_panels_with_explore_raises():
    with pytest.raises(ValueError, match='explore'):
        hyp.plot(_datasets(3), panels=True, explore=True, show=False)


def test_panels_needs_a_list_of_datasets():
    one = _datasets(1)[0]
    with pytest.raises(ValueError, match='list'):
        hyp.plot(one, panels=True, show=False)


def test_panels_and_subplots_together_raise():
    with pytest.raises(ValueError, match='aliases'):
        hyp.plot(_datasets(3), panels=True, subplots=True, show=False)


def test_subplots_alias_behaves_like_panels():
    data = _datasets(3)
    fig = hyp.plot(data, subplots=(1, 3), title=['a', 'b', 'c'],
                   reduce='PCA', show=False)
    assert [ax.get_title() for ax in fig.axes] == ['a', 'b', 'c']


def test_bad_panels_value_raises():
    with pytest.raises(ValueError, match='panels='):
        hyp.plot(_datasets(3), panels='grid', show=False)


def test_panels_false_is_the_ordinary_single_axes_call():
    data = _datasets(3)
    fig = hyp.plot(data, panels=False, reduce='PCA', show=False)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 3


# --- saving / show ------------------------------------------------------

def test_panels_save_path_writes_the_whole_grid(tmp_path):
    out = tmp_path / 'panels.png'
    data = _datasets(3)
    fig = hyp.plot(data, panels=(1, 3), reduce='PCA', show=False,
                   save_path=str(out))
    assert out.exists() and out.stat().st_size > 1000
    assert len(fig.axes) == 3


def test_panels_figure_stays_renderable_after_show_false():
    data = _datasets(3)
    fig = hyp.plot(data, panels=(1, 3), reduce='PCA', show=False)
    fig.canvas.draw()          # would raise if plt.close detached the canvas
    assert fig.canvas.get_renderer() is not None


# --- plotly parity ------------------------------------------------------

def test_panels_under_plotly_builds_a_scene_grid():
    pytest.importorskip('plotly')
    data = _datasets(3, rows=20)
    fig = hyp.plot(data, panels=(1, 3), reduce='PCA', backend='plotly',
                   show=False, title=['a', 'b', 'c'])
    layout = fig.layout
    assert layout.scene is not None
    assert layout.scene2 is not None
    assert layout.scene3 is not None
    assert len(fig.data) >= 3
    assert {a.text for a in layout.annotations} == {'a', 'b', 'c'}


# --- 1.1 release-review fixes (P4-P6) -----------------------------------

@pytest.mark.parametrize('panel_fit', ['shared', 'independent'])
def test_P4_ndims_above_3_draws_3d_panels_on_matplotlib(panel_fit):
    fig = hyp.plot(_datasets(2), ndims=4, panels=True, panel_fit=panel_fit,
                   show=False)
    try:
        assert [ax.name for ax in fig.axes] == ['3d', '3d']
        for ax in fig.axes:
            assert len(ax.lines) >= 1
    finally:
        matplotlib.pyplot.close(fig)


def test_P4_ndims_above_3_draws_3d_panels_on_plotly():
    pytest.importorskip('plotly')
    fig = hyp.plot(_datasets(2), ndims=4, panels=True, backend='plotly',
                   show=False)
    assert all(trace.type == 'scatter3d' for trace in fig.data)
    assert fig.layout.scene is not None and fig.layout.scene2 is not None


def test_P5_panel_save_path_expands_tilde(tmp_path):
    import os
    home = os.path.expanduser('~')
    name = f'.hyp_panels_p5_{os.getpid()}.png'
    target = os.path.join(home, name)
    fig = hyp.plot(_datasets(2), panels=True, save_path=f'~/{name}',
                   show=False)
    try:
        assert os.path.isfile(target) and os.path.getsize(target) > 0
    finally:
        matplotlib.pyplot.close(fig)
        if os.path.exists(target):
            os.remove(target)


def test_P5_panel_save_path_accepts_path_objects_on_both_backends(tmp_path):
    fig = hyp.plot(_datasets(2), panels=True, save_path=tmp_path / 'p.png',
                   show=False)
    try:
        assert (tmp_path / 'p.png').stat().st_size > 0
    finally:
        matplotlib.pyplot.close(fig)
    pytest.importorskip('plotly')
    hyp.plot(_datasets(2), panels=True, backend='plotly',
             save_path=tmp_path / 'p.html', show=False)
    assert (tmp_path / 'p.html').stat().st_size > 0


def test_P5_a_missing_directory_fails_before_any_panel_is_drawn(tmp_path):
    before = set(matplotlib.pyplot.get_fignums())
    with pytest.raises(FileNotFoundError, match='directory does not exist'):
        hyp.plot(_datasets(2), panels=True,
                 save_path=tmp_path / 'missing' / 'p.png', show=False)
    assert set(matplotlib.pyplot.get_fignums()) == before
    with pytest.raises(FileNotFoundError, match='directory does not exist'):
        hyp.plot(_datasets(2), panels=True, backend='plotly',
                 save_path=tmp_path / 'missing' / 'p.html', show=False)


def test_P6_plotly_panels_return_the_single_axes_figure_type(capsys):
    """The plotly panel path returned a bare ``go.Figure`` and called
    ``fig.show()`` itself, bypassing the one-shot end-of-cell display queue
    -- a notebook cell ending in the call displayed the grid twice."""
    pytest.importorskip('plotly')
    import json
    import IPython
    import plotly.io as pio
    from hypertools.plot import plotly_backend

    assert IPython.get_ipython() is None
    single = hyp.plot(_datasets(1)[0], backend='plotly', show=False)
    panels = hyp.plot(_datasets(2), panels=True, backend='plotly',
                      show=False)
    assert type(panels) is type(single)
    assert type(panels).__name__ == 'HyperPlotlyFigure'
    assert len(panels.data) >= 2

    saved_renderer = pio.renderers.default
    pio.renderers.default = 'json'
    try:
        capsys.readouterr()
        hyp.plot(_datasets(2), panels=True, backend='plotly', show=False)
        assert capsys.readouterr().out == ''            # show=False: no show
        fig = hyp.plot(_datasets(2), panels=True, backend='plotly',
                       show=True)
        out = capsys.readouterr().out
        assert out.count("'application/json'") == 1     # shown exactly once
        assert str({'application/json': json.loads(fig.to_json())}) in out
        assert plotly_backend._PENDING_DISPLAY == []
    finally:
        pio.renderers.default = saved_renderer


# --- plotly cell parity (1.1 release review: transplant_panel) -----------

def _df2(seed, cols=('a', 'b')):
    import pandas as pd
    arr = hyp.load('random_walk', n_samples=30, n_features=2,
                   random_state=seed)
    return pd.DataFrame(arr, columns=list(cols))


def test_plotly_2d_panels_keep_the_unit_frame_and_column_labels():
    pytest.importorskip('plotly')
    fig = hyp.plot([_df2(0), _df2(1)], panels=True, ndims=2, reduce=None,
                   backend='plotly', show=False)
    assert list(fig.layout.xaxis.range) == [-UNIT_FRAME_LIMIT, UNIT_FRAME_LIMIT]
    assert list(fig.layout.yaxis2.range) == [-UNIT_FRAME_LIMIT, UNIT_FRAME_LIMIT]
    assert fig.layout.xaxis2.title.text == 'a'
    assert fig.layout.yaxis2.title.text == 'b'
    assert fig.layout.xaxis2.showticklabels is False
    # one frame square per panel, each on its own cell's axes
    assert sorted(s.xref for s in fig.layout.shapes) == ['x', 'x2']
    assert sorted(s.yref for s in fig.layout.shapes) == ['y', 'y2']


def test_plotly_2d_panels_axis_scale_data_keep_visible_axes():
    pytest.importorskip('plotly')
    fig = hyp.plot([_df2(0), _df2(1)], panels=True, ndims=2, reduce=None,
                   axis_scale='data', backend='plotly', show=False)
    assert fig.layout.xaxis2.showticklabels is True
    # the data's own range, not the unit frame
    assert list(fig.layout.xaxis2.range) != [-UNIT_FRAME_LIMIT, UNIT_FRAME_LIMIT]
    assert fig.layout.xaxis2.range[1] - fig.layout.xaxis2.range[0] > 2.2
    assert len(fig.layout.shapes) == 0


def test_plotly_panels_get_one_legend_each_with_their_own_entries():
    pytest.importorskip('plotly')
    data = _datasets(3, rows=20)
    fig = hyp.plot(data, panels=True, legend=True, names=['p', 'q', 'r'],
                   backend='plotly', show=False)
    by_legend = {}
    for trace in fig.data:
        if trace.showlegend:
            by_legend.setdefault(trace.legend, []).append(trace.name)
    assert by_legend == {'legend': ['p'], 'legend2': ['q'],
                         'legend3': ['r']}
    # each legend sits just right of its own cell, not at the figure edge
    for key, scene in (('legend', 'scene'), ('legend2', 'scene2'),
                       ('legend3', 'scene3')):
        x1 = fig.layout[scene].domain.x[1]
        assert fig.layout[key].x > x1
        assert fig.layout[key].x < x1 + 0.2
    assert fig.layout.showlegend is True


def test_plotly_panels_hue_legend_lists_each_group_once_per_panel():
    pytest.importorskip('plotly')
    data = _datasets(2, rows=20)
    hue = [['x'] * 10 + ['y'] * 10] * 2
    fig = hyp.plot(data, panels=True, hue=hue, legend=True,
                   backend='plotly', show=False)
    names = {}
    for trace in fig.data:
        if trace.showlegend:
            names.setdefault(trace.legend, []).append(trace.name)
    assert names == {'legend': ['x', 'y'], 'legend2': ['x', 'y']}


def test_plotly_panels_without_legend_have_none():
    pytest.importorskip('plotly')
    fig = hyp.plot(_datasets(2, rows=20), panels=True, backend='plotly',
                   show=False)
    assert fig.layout.showlegend is False


def test_plotly_panels_place_each_colorbar_beside_its_own_panel():
    pytest.importorskip('plotly')
    data = _datasets(2, rows=20)
    hue = [np.arange(20.0), np.arange(20.0)]
    fig = hyp.plot(data, panels=True, hue=hue, colorbar=True,
                   backend='plotly', show=False)
    colorbars = [t.marker.colorbar for t in fig.data
                 if t.marker is not None and t.marker.showscale]
    assert len(colorbars) == 2
    assert colorbars[0].x < colorbars[1].x
    assert colorbars[0].x > fig.layout.scene.domain.x[1]
    assert colorbars[1].x > fig.layout.scene2.domain.x[1]
    assert colorbars[0].x < fig.layout.scene2.domain.x[0]


def test_plotly_panels_reserve_a_gutter_only_when_needed():
    pytest.importorskip('plotly')
    from hypertools.plot.plotly_backend import DEFAULT_FIGSIZE
    plain = hyp.plot(_datasets(2, rows=20), panels=True, backend='plotly',
                     show=False)
    with_legend = hyp.plot(_datasets(2, rows=20), panels=True, legend=True,
                           backend='plotly', show=False)
    assert plain.layout.width == int(DEFAULT_FIGSIZE[0] * 100)
    assert with_legend.layout.width > plain.layout.width
    assert with_legend.layout.margin.r > plain.layout.margin.r
    # an explicit size= is honoured verbatim on both
    sized = hyp.plot(_datasets(2, rows=20), panels=True, legend=True,
                     size=[9, 3], backend='plotly', show=False)
    assert (sized.layout.width, sized.layout.height) == (900, 300)


def test_plotly_3d_panels_keep_the_single_axes_camera_in_square_cells():
    """plotly sizes a scene by its domain's height, so a cube in a tall
    narrow cell spilled out of the cell's sides. The grid's 3-D cells are
    SQUARE (1.1 release review, like the matplotlib grid's), so no cell of
    `panels=` needs the camera backed off: three panels in a default-sized
    figure, and two in a wide one, keep the single-axes distance. (A
    genuinely narrow cell -- a caller's own `column_widths=` -- is backed
    off; see tests/test_plot_panels_geometry.py.)"""
    pytest.importorskip('plotly')
    single = hyp.plot(_datasets(1, rows=20)[0], backend='plotly',
                      show=False)
    eye = single.layout.scene.camera.eye
    r_single = (eye.x ** 2 + eye.y ** 2 + eye.z ** 2) ** 0.5
    fig = hyp.plot(_datasets(3, rows=20), panels=(1, 3), backend='plotly',
                   show=False)
    for key in ('scene', 'scene2', 'scene3'):
        e = fig.layout[key].camera.eye
        r = (e.x ** 2 + e.y ** 2 + e.z ** 2) ** 0.5
        assert r == pytest.approx(r_single)
        d = fig.layout[key].domain
        plot_w = fig.layout.width - fig.layout.margin.l - fig.layout.margin.r
        plot_h = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
        assert (d.x[1] - d.x[0]) * plot_w == pytest.approx(
            (d.y[1] - d.y[0]) * plot_h, abs=2.0)
    wide = hyp.plot(_datasets(2, rows=20), panels=(1, 2), size=[16, 4],
                    backend='plotly', show=False)
    e = wide.layout.scene.camera.eye
    assert (e.x ** 2 + e.y ** 2 + e.z ** 2) ** 0.5 == pytest.approx(r_single)


def test_plotly_panels_labels_annotations_follow_their_cell():
    pytest.importorskip('plotly')
    fig = hyp.plot([_df2(0), _df2(1)], panels=True, ndims=2, reduce=None,
                   labels=[['first'], ['second']], label_anchor='first',
                   backend='plotly', show=False)
    labels = [a for a in fig.layout.annotations if a.text in ('first',
                                                              'second')]
    assert [(a.text, a.xref, a.yref) for a in labels] == [
        ('first', 'x', 'y'), ('second', 'x2', 'y2')]


# --- matplotlib panels: colorbars take room from their own panel ---------

def test_matplotlib_panels_draw_one_colorbar_per_panel_without_warnings():
    import warnings
    data = _datasets(2, rows=20)
    hue = [np.arange(20.0), np.arange(20.0)]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fig = hyp.plot(data, panels=True, hue=hue, colorbar=True,
                       show=False)
    try:
        panels = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
        cbars = [ax for ax in fig.axes if ax.get_label() == '<colorbar>']
        assert len(panels) == 2 and len(cbars) == 2
        # each colorbar sits to the right of its own panel and left of the
        # next panel: no two share a position
        xs = sorted(ax.get_position().x0 for ax in panels + cbars)
        assert len(set(round(x, 3) for x in xs)) == 4
        order = sorted(panels + cbars, key=lambda ax: ax.get_position().x0)
        assert [ax.get_label() == '<colorbar>' for ax in order] == [
            False, True, False, True]
    finally:
        matplotlib.pyplot.close(fig)


def test_matplotlib_ax_grid_colorbars_do_not_widen_the_figure():
    fig, axes = hyp.subplots(1, 2, size=[8, 4])
    try:
        for ax, d in zip(axes, _datasets(2, rows=20)):
            hyp.plot(d, ax=ax, hue=np.arange(20.0), colorbar=True,
                     show=False)
        assert tuple(fig.get_size_inches()) == (8.0, 4.0)
        assert sum(ax.get_label() == '<colorbar>' for ax in fig.axes) == 2
    finally:
        matplotlib.pyplot.close(fig)


# --- release review round 2 ------------------------------------------------

def test_matplotlib_panels_ignore_an_active_plotly_preference():
    """`panels=True, backend='matplotlib'` under
    `hyp.set_interactive_backend('plotly')` built its grid with `subplots`'
    new `backend='auto'` default -- plotly cells the matplotlib panel calls
    then rejected."""
    pytest.importorskip('plotly')
    with hyp.set_interactive_backend('plotly'):
        fig = hyp.plot(_datasets(2, rows=20), panels=True,
                       backend='matplotlib', show=False)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert [ax.name for ax in fig.axes] == ['3d', '3d']
    finally:
        matplotlib.pyplot.close(fig)


def test_plotly_panels_keep_a_left_colorbar_on_the_left():
    pytest.importorskip('plotly')
    data = _datasets(2, rows=20)
    hue = [np.arange(20.0), np.arange(20.0)]
    fig = hyp.plot(data, panels=True, hue=hue,
                   colorbar={'location': 'left'}, backend='plotly',
                   show=False)
    colorbars = [t.marker.colorbar for t in fig.data
                 if t.marker is not None and t.marker.showscale]
    assert len(colorbars) == 2
    assert colorbars[0].xanchor == 'right'
    assert colorbars[0].x <= fig.layout.scene.domain.x[0]
    assert colorbars[1].xanchor == 'right'
    assert colorbars[1].x <= fig.layout.scene2.domain.x[0]


def test_plotly_panel_titles_go_through_the_title_path():
    """A newline in a plotly title becomes ``<br>`` and `title_kwargs=`
    styles it on the single-axes path; panel titles used to bypass both."""
    pytest.importorskip('plotly')
    fig = hyp.plot(_datasets(2, rows=20), panels=True,
                   title=['first\nline', 'second'],
                   title_kwargs={'fontsize': 20, 'color': 'red'},
                   backend='plotly', show=False)
    titles = {a.text: a for a in fig.layout.annotations}
    assert set(titles) == {'first<br>line', 'second'}
    single = hyp.plot(_datasets(1, rows=20)[0], title='x',
                      title_kwargs={'fontsize': 20, 'color': 'red'},
                      backend='plotly', show=False)
    assert titles['second'].font.size == single.layout.title.font.size
    assert titles['second'].font.color == single.layout.title.font.color
    assert fig.layout.margin.t >= 40


def test_plotly_panels_carry_the_font_into_the_grid():
    pytest.importorskip('plotly')
    fig = hyp.plot(_datasets(2, rows=20), panels=True, legend=True,
                   names=['p', 'q'], font='DejaVu Serif', backend='plotly',
                   show=False)
    single = hyp.plot(_datasets(1, rows=20)[0], legend=True, names=['p'],
                      font='DejaVu Serif', backend='plotly', show=False)
    assert fig.layout.font.family == single.layout.font.family
    assert fig.layout.legend.font.family == single.layout.font.family
    assert fig.layout.legend2.font.family == single.layout.font.family


def test_plotly_grid_png_draws_something_in_every_cell(tmp_path):
    """A rendered 1x2 plotly grid has ink in BOTH halves (not just a
    nonempty file)."""
    pytest.importorskip('plotly')
    from PIL import Image
    fig = hyp.plot(_datasets(2, rows=20), panels=(1, 2), backend='plotly',
                   show=False)
    target = tmp_path / 'grid.png'
    fig.write_image(str(target))
    img = np.asarray(Image.open(target).convert('L'))
    ink = img < 128
    left = ink[:, : img.shape[1] // 2].sum()
    right = ink[:, img.shape[1] // 2:].sum()
    assert left > 200 and right > 200


def test_plotly_panels_reserve_the_multiline_title_margin():
    pytest.importorskip('plotly')
    single = hyp.plot(_datasets(1, rows=20)[0], title='a\nb\nc',
                      title_kwargs={'fontsize': 20}, backend='plotly',
                      show=False)
    grid = hyp.plot(_datasets(2, rows=20), panels=True,
                    title=['a\nb\nc', 'x'], title_kwargs={'fontsize': 20},
                    backend='plotly', show=False)
    assert single.layout.margin.t > 40
    # the grid's first row reserves at least what the single figure did
    # (more when square cells leave vertical slack that centres the grid);
    # a one-line title next to it does not shrink the reservation
    assert grid.layout.margin.t >= single.layout.margin.t
    plain = hyp.plot(_datasets(2, rows=20), panels=True, title=['a', 'b'],
                     backend='plotly', show=False)
    assert grid.layout.margin.t - plain.layout.margin.t >= \
        (single.layout.margin.t - 40) / 2
