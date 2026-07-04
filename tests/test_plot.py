# -*- coding: utf-8 -*-

import pytest

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from hypertools.plot import plot
from hypertools.reduce.reduce import reduce as reducer
from hypertools.io.load import load

data = [np.random.multivariate_normal(np.zeros(4), np.eye(4), size=100) for i
        in range(2)]
weights = load('weights_avg')

# To prevent warning about 20+ figs being open
mpl.rcParams['figure.max_open_warning'] = 25

## STATIC ##


def test_plot_1d():
    data_reduced_1d = reducer(data, ndims=1)
    fig = plot.plot(data_reduced_1d, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    assert all([i.shape[1]==1 for i in data_reduced_1d])


def test_plot_2d():
    data_reduced_2d = reducer(data, ndims=2)
    fig = plot.plot(data_reduced_2d, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    assert all([i.shape[1]==2 for i in data_reduced_2d])


def test_plot_3d():
    data_reduced_3d = reducer(data, ndims=3)
    fig = plot.plot(data_reduced_3d, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    assert all([i.shape[1]==3 for i in data_reduced_3d])


def test_plot_reduce_none():
    # Should return same dimensional data if ndims is None
    fig = plot.plot(data, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_reduce3d():
    # should return 3d data since ndims=3
    result = plot.plot(data, ndims=3, show=False, return_model=True)
    assert all([i.shape[1] == 3 for i in result['xform_data']])


def test_plot_reduce2d():
    # should return 2d data since ndims=2
    result = plot.plot(data, ndims=2, show=False, return_model=True)
    assert all([i.shape[1] == 2 for i in result['xform_data']])


def test_plot_reduce1d():
    # should return 1d data since ndims=1
    result = plot.plot(data, ndims=1, show=False, return_model=True)
    assert all([i.shape[1] == 1 for i in result['xform_data']])


def test_plot_reduce_align5d():
    # should return 5d data since ndims=5
    result = plot.plot(weights, ndims=5, align='hyper', show=False,
                       return_model=True)
    assert all([i.shape[1] == 5 for i in result['xform_data']])


def test_plot_reduce10d():
    # should return 10d data since ndims=10
    result = plot.plot(weights, ndims=10, show=False, return_model=True)
    assert all([i.shape[1] == 10 for i in result['xform_data']])


def test_plot_model_dict():
    fig = plot.plot(weights, reduce={'model' : 'PCA', 'params' : {'whiten' : True}}, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_cluster_str():
    fig = plot.plot(weights, cluster='KMeans', show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_cluster_dict():
    fig = plot.plot(weights, cluster={'model' : 'KMeans', 'params' : {'n_clusters' : 3}}, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_cluster_n_clusters():
    fig = plot.plot(weights, n_clusters=3, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_cluster_hdbscan_dict_with_n_clusters():
    # Regression: cluster={'model': 'HDBSCAN'} + n_clusters used to crash
    # with "HDBSCAN.__init__() got an unexpected keyword argument
    # 'n_clusters'" because the guard checked the raw `cluster` arg (a dict
    # in this branch) instead of the resolved model name, so n_clusters
    # leaked into params instead of being dropped with a warning.
    with pytest.warns(UserWarning, match="n_clusters is not a valid parameter"):
        fig = plot.plot(weights, cluster={'model': 'HDBSCAN'}, n_clusters=2,
                        show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_nd():
    fig = plot.plot(data, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_data_is_list():
    # list input still plots and returns a figure
    fig = plot.plot(data, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_check_fig():
    fig = plot.plot(data, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_check_ax():
    fig = plot.plot(data, show=False)
    assert isinstance(fig.axes[0], mpl.axes._axes.Axes)


def test_plot_text():
    text_data = [['i like cats alot', 'cats r pretty cool', 'cats are better than dogs'],
            ['dogs rule the haus', 'dogs are my jam', 'dogs are a mans best friend']]
    fig = plot.plot(text_data, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_ax():
    parent = plt.figure()
    ax = parent.add_subplot(111, projection='3d')
    fig = plot.plot(data, ax=ax, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    assert fig is ax.figure


def test_plot_ax_2d():
    parent = plt.figure()
    ax = parent.add_subplot(111)
    fig = plot.plot(data, ax=ax, show=False, ndims=2)
    assert isinstance(fig, mpl.figure.Figure)
    assert fig is ax.figure


def test_plot_ax_error():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError) as e_info:
        plot.plot(data, ax=ax, show=False)


def test_plot_geo():
    # re-plotting the same raw data twice both succeed (geo replay retired)
    fig = plot.plot(data, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    fig2 = plot.plot(data, show=False)
    assert isinstance(fig2, mpl.figure.Figure)


# ## ANIMATED ##
def test_plot_1d_animate():
    d = reducer(data, ndims=1)
    with pytest.raises(Exception) as e_info:
        plot.plot(d, animate=True, show=False)


def test_plot_2d_animate():
    data_reduced_2d = reducer(data, ndims=2)
    with pytest.raises(Exception) as e_info:
        plot.plot(data_reduced_2d, animate=True, show=False)


def test_plot_3d_animate():
    data_reduced_3d = reducer(data,ndims=3)
    fig, ani = plot.plot(data_reduced_3d, animate=True, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    assert all([i.shape[1]==3 for i in data_reduced_3d])


def test_plot_nd_animate():
    fig, ani = plot.plot(data, animate=True, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_data_animate_is_list():
    # list input still animates and returns a (fig, animation) tuple
    fig, ani = plot.plot(data, animate=True, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_animate_check_fig():
    fig, ani = plot.plot(data, animate=True, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_animate_check_ax():
    fig, ani = plot.plot(data, animate=True, show=False)
    assert isinstance(fig.axes[0], mpl.axes._axes.Axes)


def test_plot_animate_check_line_ani():
    fig, ani = plot.plot(data, animate=True, show=False)
    assert isinstance(ani, mpl.animation.FuncAnimation)


def test_plot_animate_return_model_includes_animation():
    # Regression: return_model=True combined with animate=True used to
    # return the {'fig','xform_data','models'} bundle *before* the
    # (fig, line_ani) return path, silently dropping the only reference to
    # the FuncAnimation (so it could be garbage-collected before it ever
    # played). The bundle must carry the animation handle too.
    result = plot.plot(data, animate=True, return_model=True, show=False)
    assert isinstance(result, dict)
    assert isinstance(result['animation'], mpl.animation.Animation)


def test_plot_animate_static_return_model_animation_is_none():
    # Non-animated return_model bundles should carry animation=None rather
    # than omitting the key.
    result = plot.plot(data, animate=False, return_model=True, show=False)
    assert result['animation'] is None


def test_plot_animate_parallel_honors_elev():
    # Regression: update_lines_parallel (the animate=True/'parallel' style)
    # hardcoded ax.view_init(elev=10, ...) instead of using the `elev`
    # fargs parameter, so a non-default elev was silently ignored.
    fig, ani = plot.plot(data, animate=True, elev=45, show=False)
    ax = fig.axes[0]
    # drive one animation frame directly to exercise update_lines_parallel
    ani._func(2, *ani._args)
    assert ax.elev == 45


def test_anim_box_zoom_is_zoomed_out():
    # the animated set_box_aspect zoom is slightly LOWER than the historical
    # 1.25 (10/8) so the wireframe box keeps a margin at every rotation angle
    from hypertools.plot.matplotlib_backend import _anim_box_zoom
    assert _anim_box_zoom(1) == pytest.approx(1.125)   # 9/8, was 10/8=1.25
    assert _anim_box_zoom(1) < 1.25


def test_spin_box_never_clipped():
    # Regression: a full-rotation spin must keep the wireframe box (and data)
    # fully inside the figure -- no edge is clipped at any azimuth. Drives the
    # real update_lines_spin closure and checks the inked bounding box has a
    # margin to every edge across a full 360 deg rotation.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    d = reducer(data, ndims=3)
    fig, ani = plot.plot(d, animate='spin', show=False)
    # plot(show=False) closes the figure; matplotlib >= 3.11 then resets its
    # canvas (dropping buffer_rgba), so rasterize through a fresh Agg canvas.
    canvas = FigureCanvasAgg(fig)
    total = ani._save_count            # frame_rate * duration (default 900)
    for num in np.linspace(0, total - 1, 12, dtype=int):
        ani._func(int(num), *ani._args)
        canvas.draw()
        rgb = np.asarray(canvas.buffer_rgba())[..., :3]
        inked = np.any(rgb < 250, axis=-1)
        ys, xs = np.where(inked)
        h, w = inked.shape
        left, right = int(xs.min()), w - 1 - int(xs.max())
        top, bottom = int(ys.min()), h - 1 - int(ys.max())
        assert min(left, right, top, bottom) > 10, (
            f'box clipped at azim frame {int(num)}: '
            f'L={left} R={right} T={top} B={bottom}')
    plt.close('all')


## LEGEND PLACEMENT ##


def _legend_and_axes_fraction(fig):
    """(legend_bbox, axes_bbox) in figure-fraction coordinates."""
    ax = fig.axes[0]
    lg = ax.get_legend()
    assert lg is not None, "no legend was drawn"
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    return (lg.get_window_extent().transformed(inv),
            ax.get_window_extent().transformed(inv))


@pytest.mark.parametrize("ndims", [2, 3])
def test_legend_is_right_of_plot_and_within_figure(ndims):
    # the legend must sit to the RIGHT of the axes AND stay fully inside the
    # figure (not clipped off the right edge). tight_layout reserves room for
    # an outside legend on 2D axes but not on 3D axes, so wide 3D legends used
    # to overflow the canvas.
    d = [np.random.default_rng(0).random((15, ndims)) for _ in range(3)]
    fig = plot.plot(d, legend=['first label', 'second label', 'third label'],
                    show=False)
    lb, ab = _legend_and_axes_fraction(fig)
    # to the right of the axes
    assert lb.x0 >= ab.x1 - 0.05, "legend is not to the right of the axes"
    # not clipped off the figure's right edge
    assert lb.x1 <= 1.001, f"legend clipped off right edge (x1={lb.x1:.3f})"
    plt.close('all')


def test_legend_right_with_long_labels_3d_not_clipped():
    # a 3D plot with long legend labels is the case that used to clip: the
    # legend must remain fully within the figure.
    d = [np.random.default_rng(0).random((15, 3)) for _ in range(3)]
    fig = plot.plot(
        d,
        legend=['very long label number one',
                'another quite long label two',
                'third extremely long legend label'],
        show=False)
    lb, ab = _legend_and_axes_fraction(fig)
    assert lb.x0 >= ab.x1 - 0.05
    assert lb.x1 <= 1.001, f"legend clipped off right edge (x1={lb.x1:.3f})"
    plt.close('all')


@pytest.mark.parametrize("ndims,labels", [
    (3, ['very long label number one', 'another quite long label two',
         'third extremely long legend label']),
    (2, [f'condition number {i}' for i in range(6)]),
])
def test_legend_not_clipped_in_saved_pixels(ndims, labels, tmp_path):
    # Regression: wide legends (long labels / many entries) clipped off the
    # right edge of the SAVED image. The earlier fit measured the legend under
    # seaborn's (narrower) font while the figure is actually saved under the
    # default (wider) font, so it looked like it fit yet clipped. Measure the
    # real saved pixels -- this fails on the pre-fix code and passes now.
    from PIL import Image
    d = [np.random.default_rng(0).random((15, ndims)) for _ in range(len(labels))]
    fig = plot.plot(d, '.', legend=labels, show=False)
    out = str(tmp_path / 'legend.png')
    fig.savefig(out)
    im = np.asarray(Image.open(out).convert('L'))
    ink_cols = np.where((im < 245).any(axis=0))[0]
    right_margin = im.shape[1] - 1 - int(ink_cols.max())
    plt.close('all')
    assert right_margin > 4, \
        f"legend clipped in saved image: right margin {right_margin}px"
