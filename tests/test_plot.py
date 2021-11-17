# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import dill
import os

import hypertools as hyp


data = hyp.load('weights')[:5]
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
    save_fig(name, fig)  # FIXME: COMMENT OUT ONCE REFERENCE FIGS ARE GENERATED
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
    plot_test('fig1', data, reduce=pca)
    plot_test('fig2', data, '-', reduce=pca)
    plot_test('fig3', data, reduce=pca, color=data)

    # markers
    plot_test('fig4', data, ',', reduce=pca)
    plot_test('fig5', data, '.', reduce=pca)
    plot_test('fig6', data, 'o', reduce=pca, color=data)

    # lines + markers
    plot_test('fig7', data, '-.', reduce=pca)
    plot_test('fig8', data, ':o', reduce=pca, color=data)

    # zscore, align, cluster
    kmeans = {'model': 'KMeans', 'args': [], 'kwargs': {'n_clusters': 5}}
    plot_test('fig9', data, '*-', reduce=pca, cluster=kmeans, align='HyperAlign', manip='ZScore')
    plot_test('fig10', data, 'D--', reduce=pca, align='SharedResponseModel', manip=['ZScore', 'Resample', 'Smooth'])

    # pipeline
    pca_10d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 10}}
    pca_5d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 5}}
    plot_test('fig11', data, 'star-triangle-down-open-dot--', reduce=pca, pipeline=[pca_10d, pca_5d])


# noinspection DuplicatedCode
def test_static_plot3d():
    # test lines, markers
    # test different strategies for managing color
    # test various manipulations (align, cluster, manip, reduce)
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 3}}

    # lines
    plot_test('fig12', data, reduce=pca)
    plot_test('fig13', data, '-', reduce=pca)
    plot_test('fig14', data, reduce=pca, color=data)

    # markers
    plot_test('fig15', data, ',', reduce=pca)
    plot_test('fig16', data, '.', reduce=pca)
    plot_test('fig17', data, 'o', reduce=pca, color=data)

    # lines + markers
    plot_test('fig18', data, '-.', reduce=pca)
    plot_test('fig19', data, ':o', reduce=pca, color=data)

    # zscore, align, cluster
    kmeans = {'model': 'KMeans', 'args': [], 'kwargs': {'n_clusters': 5}}
    plot_test('fig20', data, 'x-', reduce=pca, cluster=kmeans, align='HyperAlign', manip='ZScore')
    plot_test('fig21', data, 'd--', reduce=pca, align='SharedResponseModel', manip=['ZScore', 'Resample', 'Smooth'])

    # pipeline
    pca_10d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 10}}
    pca_5d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 5}}
    plot_test('fig22', data, 'diamond-open:', reduce=pca, pipeline=[pca_10d, pca_5d])

    # manipulate and align
    manip = ['ZScore', {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 1000}}, 'Smooth']
    plot_test('fig23', data, manip=manip, align='HyperAlign')

    # pipeline, manipulate, align
    plot_test('fig24', data, manip=manip, align='SharedResponseModel', pipeline=[pca_10d, pca_5d])

    # colormap
    plot_test('fig25', data, cmap='flare')

    # mixture model
    gaussian_mixture = {'model': 'GaussianMixture', 'args': [], 'kwargs': {'n_components': 10}}
    plot_test(f'fig26', data[0], cluster=gaussian_mixture, cmap='husl')

test_static_plot3d()

def test_animated_plot2d():
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 2}}

    # basic animations of each style
    styles = ['window', 'chemtrails', 'precog', 'bullettime', 'grow', 'shrink', 'spin']
    fig_num = 27
    for s in styles:
        plot_test(f'fig{fig_num}', data, reduce=pca, animate=True, style=s)
        fig_num += 1

    # zscore + resampling + smoothing + hyperalign, with each style
    manip = ['ZScore', {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 500}}, 'Smooth']
    hyperalign = {'model': 'HyperAlign', 'args': [], 'kwargs': {'n_iter': 3}}

    # for s in styles:
    #     plot_test(f'fig{fig_num}', data, reduce=pca, animate=True, style=s, manip=manip, align=hyperalign)
    #     fig_num += 1

    # also test each combination of lines, markers, and lines + markers
    # use different line styles and marker shapes
    styles = ['.', ':o']
    # for s in styles:
    #     plot_test(f'fig{fig_num}', data, s, reduce=pca, animate=True, style='window')
    #     fig_num += 1
    #
    # # verify that resampling and smoothing change animations correctly
    # for s in styles:
    #     plot_test(f'fig{fig_num}', data, s, reduce=pca, manip=manip, animate=True, style='chemtrails')
    #     fig_num += 1

    # # verify that coloring works with animations
    # for s in styles:
    #     plot_test(f'fig{fig_num}', data, s, reduce=pca, animate=True, style='grow', color=data)
    #     fig_num += 1
    #
    # # verify that single-line animations work, also try mixture-based coloring and a custom colormap
    # gaussian_mixture = {'model': 'GaussianMixture', 'args': [], 'kwargs': {'n_components': 10}}
    # plot_test(f'fig{fig_num}', data[0], reduce=pca, animate=True, style='shrink', cluster=gaussian_mixture, cmap='husl')
    # fig_num += 1

    # test timing: total duration, window length, tail length (noting for 3d: also test number of rotations)
    # (for 3d: test zoom)
    plot_test(f'fig{fig_num}', data, reduce=pca, animate=True, style='precog', cmap='light:seagreen', duration=20,
              focused=1, unfocused=5)


test_animated_plot2d()


def test_animated_plot3d():
    # test each type of animation
    # also test saving in various formats
    pass


def test_backend_management():
    # not implemented
    pass
