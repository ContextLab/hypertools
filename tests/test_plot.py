# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import dill
import os

import hypertools as hyp


weights = hyp.load('weights')
fig_dir = os.path.join(os.path.dirname(__file__), 'reference_figures')


def compare_figs(f, name):
    def compare_helper(ref, x):
        if not type(ref) == type(x):
            return False

        if type(ref) is dict:
            for k in ref.keys():
                if k not in x.keys():
                    return False
                elif not compare_helper(ref[k], x[k]):
                    return False
        elif type(ref) is list:
            if not all([compare_helper(a, b) for a, b in zip(ref, x)]):
                return False
        elif dw.zoo.is_array(ref) or dw.zoo.is_dataframe(ref):
            if not np.allclose(ref, x):
                return False
        elif not (ref == x):
            return False
        return True

    with open(os.path.join(fig_dir, f'{name}.fig'), 'rb') as fd:
        reference = dill.load(fd)
    return compare_helper(reference.to_dict(), f.to_dict()) and compare_helper(f.to_dict(), reference.to_dict())


def plot_test(name, *args, **kwargs):
    np.random.seed(1234)
    fig = hyp.plot(*args, **kwargs)
    # save_fig(name, fig)  # FIXME: REMOVE ONCE REFERENCE FIGS ARE GENERATED
    assert compare_figs(fig, name), f'Figure failed to replicate: {name}'


def save_fig(name, fig):
    with open(os.path.join(fig_dir, f'{name}.fig'), 'wb') as f:
        dill.dump(fig, f)


# noinspection DuplicatedCode
def test_static_plot2d():
    # test lines, markers
    # test different strategies for managing color
    # test various manipulations (align, cluster, manip, reduce)
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 2}}

    # lines
    plot_test('fig1', weights, reduce=pca)
    plot_test('fig2', weights, '-', reduce=pca)
    plot_test('fig3', weights, reduce=pca, color=weights)

    # markers
    plot_test('fig4', weights, ',', reduce=pca)
    plot_test('fig5', weights, '.', reduce=pca)
    plot_test('fig6', weights, 'o', reduce=pca, color=weights)

    # lines + markers
    plot_test('fig7', weights, '-.', reduce=pca)
    plot_test('fig8', weights, ':o', reduce=pca, color=weights)

    # zscore, align, cluster
    kmeans = {'model': 'KMeans', 'args': [], 'kwargs': {'n_clusters': 5}}
    plot_test('fig9', weights, '*-', reduce=pca, cluster=kmeans, align='HyperAlign', manip='ZScore')
    plot_test('fig10', weights, 'D--', reduce=pca, align='SharedResponseModel', manip=['ZScore', 'Resample', 'Smooth'])

    # pipeline
    pca_10d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 10}}
    pca_5d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 5}}
    plot_test('fig11', weights, 'star-triangle-down-open-dot--', reduce=pca, pipeline=[pca_10d, pca_5d])


# noinspection DuplicatedCode
def test_static_plot3d():
    # test lines, markers
    # test different strategies for managing color
    # test various manipulations (align, cluster, manip, reduce)
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 3}}

    # lines
    plot_test('fig12', weights, reduce=pca)
    plot_test('fig13', weights, '-', reduce=pca)
    plot_test('fig14', weights, reduce=pca, color=weights)

    # markers
    plot_test('fig15', weights, ',', reduce=pca)
    plot_test('fig16', weights, '.', reduce=pca)
    plot_test('fig17', weights, 'o', reduce=pca, color=weights)

    # lines + markers
    plot_test('fig18', weights, '-.', reduce=pca)
    plot_test('fig19', weights, ':o', reduce=pca, color=weights)

    # zscore, align, cluster
    kmeans = {'model': 'KMeans', 'args': [], 'kwargs': {'n_clusters': 5}}
    plot_test('fig20', weights, 'x-', reduce=pca, cluster=kmeans, align='HyperAlign', manip='ZScore')
    plot_test('fig21', weights, 'd--', reduce=pca, align='SharedResponseModel', manip=['ZScore', 'Resample', 'Smooth'])

    # pipeline
    pca_10d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 10}}
    pca_5d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 5}}
    plot_test('fig22', weights, 'diamond-open:', reduce=pca, pipeline=[pca_10d, pca_5d])


def test_animated_plot2d():
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 2}}

    # test each animation type: window, chemtrails, precog, bullettime, grow, shrink, spin
    fig12 = hyp.plot(weights[:5], reduce=pca, animate=True)

    # split for now (so that smoothing and plotting can be debugged separately if needed)
    smoothed_weights = hyp.manip(weights,
                                 model=['ZScore', {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 1000}},
                                        'Smooth'])
    fig13 = hyp.plot(smoothed_weights, reduce=pca, animate=True)

    # also test each combination of lines, markers, and lines + markers
    # use different line styles and marker shapes
    # verify that resampling and smoothing change animations correctly
    # test timing: total duration, window length, tail length (noting for 3d: also test number of rotations)
    # (for 3d: test zoom)
    pass


test_animated_plot2d()


def test_animated_plot3d():
    # test each type of animation
    # also test saving in various formats
    pass


def test_backend_management():
    # not implemented
    pass
