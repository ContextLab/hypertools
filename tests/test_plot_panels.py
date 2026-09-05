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
    # 3 datasets -> a 2x2 grid with one spare, hidden
    assert len(fig.axes) == 4
    assert [ax.get_visible() for ax in fig.axes] == [True, True, True, False]
    assert all(ax.name == '3d' for ax in fig.axes)
    # one trajectory drawn per visible panel
    for ax in fig.axes[:3]:
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
    assert bundle['panels'] == (2, 2)
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
