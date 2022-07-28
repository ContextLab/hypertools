# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import dill
import os

import pytest
import hypertools as hyp


data = hyp.load('weights')[:5]


def compare_figs(fig_dir, f, name, tol=1e-3):
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
            if not np.allclose(ref, x, atol=tol):
                return False
        elif not (ref == x):
            return False
        return True

    with open(os.path.join(fig_dir, f'{name}.fig'), 'rb') as fd:
        reference = dill.load(fd)
    return compare_helper(reference.to_dict(), f.to_dict()) and compare_helper(f.to_dict(), reference.to_dict())


def plot_test(name, fig_dir, *args, **kwargs):
    np.random.seed(1234)
    fig = hyp.plot(*args, **kwargs)
    # save_fig(name, fig, fig_dir)
    assert compare_figs(fig_dir, fig, name), f'Figure failed to replicate: {name}'


def save_fig(name, fig, fig_dir):
    with open(os.path.join(fig_dir, f'{name}.fig'), 'wb') as f:
        dill.dump(fig, f)


# noinspection DuplicatedCode
def test_static_plot2d(fig_dir):
    # test lines, markers
    # test different strategies for managing color
    # test various manipulations (align, cluster, manip, reduce)
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 2}}

    # lines
    plot_test('fig1', fig_dir, data, reduce=pca)
    plot_test('fig2', fig_dir, data, '-', reduce=pca)
    plot_test('fig3', fig_dir, data, reduce=pca, color=data)

    # markers
    plot_test('fig4', fig_dir, data, ',', reduce=pca)
    plot_test('fig5', fig_dir, data, '.', reduce=pca)
    plot_test('fig6', fig_dir, data, 'o', reduce=pca, color=data)

    # lines + markers
    plot_test('fig7', fig_dir, data, '-.', reduce=pca)
    plot_test('fig8', fig_dir, data, ':o', reduce=pca, color=data)

    # zscore, align, cluster
    kmeans = {'model': 'KMeans', 'args': [], 'kwargs': {'n_clusters': 5}}
    plot_test('fig9', fig_dir, data, '*-', reduce=pca, cluster=kmeans, align='HyperAlign', pre='ZScore')
    plot_test('fig10', fig_dir, data, 'D--', reduce=pca, align='SharedResponseModel', pre=['ZScore', 'Resample', 'Smooth'])

    # pipeline
    pca_10d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 10}}
    pca_5d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 5}}
    plot_test('fig11', fig_dir, data, 'star-triangle-down-open-dot--', reduce=pca, pipeline=[pca_10d, pca_5d])


# noinspection DuplicatedCode
def test_static_plot3d(fig_dir):
    # test lines, markers
    # test different strategies for managing color
    # test various manipulations (align, cluster, manip, reduce)
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 3}}

    # lines
    plot_test('fig12', fig_dir, data, reduce=pca)
    plot_test('fig13', fig_dir, data, '-', reduce=pca)
    plot_test('fig14', fig_dir, data, reduce=pca, color=data)

    # markers
    plot_test('fig15', fig_dir, data, ',', reduce=pca)
    plot_test('fig16', fig_dir, data, '.', reduce=pca)
    plot_test('fig17', fig_dir, data, 'o', reduce=pca, color=data)

    # lines + markers
    plot_test('fig18', fig_dir, data, '-.', reduce=pca)
    plot_test('fig19', fig_dir, data, ':o', reduce=pca, color=data)

    # zscore, align, cluster
    kmeans = {'model': 'KMeans', 'args': [], 'kwargs': {'n_clusters': 5}}
    plot_test('fig20', fig_dir, data, 'x-', reduce=pca, cluster=kmeans, align='HyperAlign', pre='ZScore')
    plot_test('fig21', fig_dir, data, 'd--', reduce=pca, align='SharedResponseModel', pre=['ZScore'], post=['Resample', 'Smooth'])

    # pipeline
    pca_10d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 10}}
    pca_5d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 5}}
    plot_test('fig22', fig_dir, data, 'diamond-open:', reduce=pca, pipeline=[pca_10d, pca_5d])

    # manipulate and align
    pre = 'ZScore'
    post = [{'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 1000}}, 'Smooth']
    plot_test('fig23', fig_dir, data, pre=pre, align='HyperAlign', post=post)

    # pipeline, manipulate, align
    plot_test('fig24', fig_dir, data, pre=pre, align='SharedResponseModel', pipeline=[pca_10d, pca_5d], post=post)

    # colormap
    plot_test('fig25', fig_dir, data, cmap='flare')

    # mixture model
    gaussian_mixture = {'model': 'GaussianMixture', 'args': [], 'kwargs': {'n_components': 3,
                                                                           'mode': 'fit_predict_proba'}}
    plot_test(f'fig26', fig_dir, data[0], cluster=gaussian_mixture, cmap='husl')
    plot_test(f'fig27', fig_dir, data, cluster=gaussian_mixture, cmap='hls')


def test_animated_plot2d(fig_dir):
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 2}}

    # basic animations of each style
    styles = ['window', 'chemtrails', 'precog', 'bullettime', 'grow', 'shrink', 'spin']
    fig_num = 28
    for s in styles:
        plot_test(f'fig{fig_num}', fig_dir, data, reduce=pca, animate=s)
        fig_num += 1

    # zscore + hyperalign + resampling + smoothing, with each style
    pre = 'ZScore'
    post = [{'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 500}}, 'Smooth']
    hyperalign = {'model': 'HyperAlign', 'args': [], 'kwargs': {'n_iter': 3}}

    for s in styles:
        plot_test(f'fig{fig_num}', fig_dir, data, reduce=pca, animate=s, pre=pre, post=post, align=hyperalign)
        fig_num += 1

    # also test each combination of lines, markers, and lines + markers
    # use different line styles and marker shapes
    styles = ['.', ':o']
    for s in styles:
        plot_test(f'fig{fig_num}', fig_dir, data, s, reduce=pca, animate='window')
        fig_num += 1

    # verify that resampling and smoothing change animations correctly
    for s in styles:
        plot_test(f'fig{fig_num}', fig_dir, data, s, reduce=pca, pre=pre, post=post, animate='chemtrails')
        fig_num += 1

    # verify that single-line animations work
    plot_test(f'fig{fig_num}', fig_dir, data[0], reduce=pca, animate='shrink', cmap='husl')
    fig_num += 1

    # test timing: total duration, window length, tail length (noting for 3d: also test number of rotations)
    # (for 3d: test zoom)
    plot_test(f'fig{fig_num}', fig_dir, data, reduce=pca, animate='precog', cmap='light:seagreen', duration=20,
              focused=1, unfocused=5)


def test_animated_plot3d(fig_dir):
    # basic animations of each style
    styles = ['window', 'chemtrails', 'precog', 'bullettime', 'grow', 'shrink', 'spin']
    fig_num = 48
    for s in styles:
        plot_test(f'fig{fig_num}', fig_dir, data, animate=s)  # FIXME: the camera view is too zoomed out...
        fig_num += 1

    # zscore + resampling + smoothing + hyperalign, with each style
    manip = ['ZScore', {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 500}}, 'Smooth']
    hyperalign = {'model': 'HyperAlign', 'args': [], 'kwargs': {'n_iter': 3}}

    for s in styles: # FIXME: manip is now specified as pre and post processing
        plot_test(f'fig{fig_num}', fig_dir, data, animate=s, manip=manip, align=hyperalign)
        fig_num += 1

    # also test each combination of lines, markers, and lines + markers
    # use different line styles and marker shapes
    styles = ['o', '-:.']
    for s in styles:
        plot_test(f'fig{fig_num}', fig_dir, data, s, animate='grow')
        fig_num += 1

    # verify that resampling and smoothing change animations correctly
    for s in styles:  # FIXME: manip is now specified as pre and post processing
        plot_test(f'fig{fig_num}', fig_dir, data, s, manip=manip, animate='bullettime')
        fig_num += 1

    # verify that single-line animations work
    plot_test(f'fig{fig_num}', fig_dir, data[0], animate='spin', cmap='cubehelix')
    fig_num += 1

    # test timing: total duration, window length, tail length (noting for 3d: also test number of rotations)
    # (for 3d: test zoom)
    plot_test(f'fig{fig_num}', fig_dir, data, animate='chemtrails', cmap='dark:blue', duration=20,
              focused=1, unfocused=5)


def test_advanced_plot_color_manipulations(fig_dir):
    # 2d plots with "spin animation"
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 2}}
    kmeans = {'model': 'KMeans', 'args': [], 'kwargs': {'n_clusters': 4}}
    gaussian_mixture = {'model': 'GaussianMixture', 'args': [],
                        'kwargs': {'n_components': 4, 'mode': 'fit_predict_proba'}}

    # color by weights
    plot_test('fig68', fig_dir, data, 'bowtie-open-dashdot', animate='spin', reduce=pca, cmap='mako', color=data)

    # color by k-means cluster
    plot_test('fig69', fig_dir, data, '8-dash', animate='spin', reduce=pca, cmap='rocket', cluster=kmeans)

    # color by mixture component
    plot_test('fig70', fig_dir, data, 'H-dot', animate='spin', reduce=pca, cmap='crest', cluster=gaussian_mixture)

    # 3d plots with spin animations
    # color by weights
    plot_test('fig71', fig_dir, data, 'dashdot', animate='spin', cmap='dark:salmon_r', color=data)

    # color by k-means cluster
    plot_test('fig72', fig_dir, data, '.-dash', animate='spin', cmap='YlOrBr', cluster=kmeans)

    # color by mixture component
    plot_test('fig73', fig_dir, data, ',-dot', animate='spin', cmap='icefire', cluster=gaussian_mixture)


def test_bounding_box(fig_dir):
    # 2d plots
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 2}}
    # static
    plot_test('fig74', fig_dir, data, bounding_box=True, reduce=pca)

    # animated
    plot_test('fig75', fig_dir, data, bounding_box=True, reduce=pca, animate=True)

    # 3d plots
    # static
    plot_test('fig76', fig_dir, data, bounding_box=True)

    # animated
    plot_test('fig77', fig_dir, data, bounding_box=True, animate='chemtrails')


def test_backend_management():
    # not implemented
    pass
