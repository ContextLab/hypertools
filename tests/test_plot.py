# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import dill
import os

import hypertools as hyp


data = hyp.load('weights')[:5]


def compare_figs(fig_dir, f, name):
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
    # save_fig(name, fig)
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
    gaussian_mixture = {'model': 'GaussianMixture', 'args': [], 'kwargs': {'n_components': 3,
                                                                           'mode': 'fit_predict_proba'}}
    plot_test(f'fig26', data[0], cluster=gaussian_mixture, cmap='husl')
    plot_test(f'fig27', data, cluster=gaussian_mixture, cmap='hls')


def test_animated_plot2d():
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 2}}

    # basic animations of each style
    styles = ['window', 'chemtrails', 'precog', 'bullettime', 'grow', 'shrink', 'spin']
    fig_num = 28
    for s in styles:
        plot_test(f'fig{fig_num}', data, reduce=pca, animate=s)
        fig_num += 1

    # zscore + resampling + smoothing + hyperalign, with each style
    manip = ['ZScore', {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 500}}, 'Smooth']
    hyperalign = {'model': 'HyperAlign', 'args': [], 'kwargs': {'n_iter': 3}}

    for s in styles:
        plot_test(f'fig{fig_num}', data, reduce=pca, animate=s, manip=manip, align=hyperalign)
        fig_num += 1

    # also test each combination of lines, markers, and lines + markers
    # use different line styles and marker shapes
    styles = ['.', ':o']
    for s in styles:
        plot_test(f'fig{fig_num}', data, s, reduce=pca, animate='window')
        fig_num += 1

    # verify that resampling and smoothing change animations correctly
    for s in styles:
        plot_test(f'fig{fig_num}', data, s, reduce=pca, manip=manip, animate='chemtrails')
        fig_num += 1

    # verify that single-line animations work
    plot_test(f'fig{fig_num}', data[0], reduce=pca, animate='shrink', cmap='husl')
    fig_num += 1

    # test timing: total duration, window length, tail length (noting for 3d: also test number of rotations)
    # (for 3d: test zoom)
    plot_test(f'fig{fig_num}', data, reduce=pca, animate='precog', cmap='light:seagreen', duration=20,
              focused=1, unfocused=5)


def test_animated_plot3d():
    # basic animations of each style
    styles = ['window', 'chemtrails', 'precog', 'bullettime', 'grow', 'shrink', 'spin']
    fig_num = 48
    for s in styles:
        plot_test(f'fig{fig_num}', data, animate=s)
        fig_num += 1

    # zscore + resampling + smoothing + hyperalign, with each style
    manip = ['ZScore', {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 500}}, 'Smooth']
    hyperalign = {'model': 'HyperAlign', 'args': [], 'kwargs': {'n_iter': 3}}

    for s in styles:
        plot_test(f'fig{fig_num}', data, animate=s, manip=manip, align=hyperalign)
        fig_num += 1

    # also test each combination of lines, markers, and lines + markers
    # use different line styles and marker shapes
    styles = ['o', '-:.']
    for s in styles:
        plot_test(f'fig{fig_num}', data, s, animate='grow')
        fig_num += 1

    # verify that resampling and smoothing change animations correctly
    for s in styles:
        plot_test(f'fig{fig_num}', data, s, manip=manip, animate='bullettime')
        fig_num += 1

    # verify that single-line animations work
    plot_test(f'fig{fig_num}', data[0], animate='spin', cmap='cubehelix')
    fig_num += 1

    # test timing: total duration, window length, tail length (noting for 3d: also test number of rotations)
    # (for 3d: test zoom)
    plot_test(f'fig{fig_num}', data, animate='chemtrails', cmap='dark:blue', duration=20,
              focused=1, unfocused=5)


def test_advanced_plot_color_manipulations():
    # 2d plots with "spin animation"
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 2}}
    kmeans = {'model': 'KMeans', 'args': [], 'kwargs': {'n_clusters': 4}}
    gaussian_mixture = {'model': 'GaussianMixture', 'args': [],
                        'kwargs': {'n_components': 4, 'mode': 'fit_predict_proba'}}

    # color by weights
    plot_test('fig68', data, 'bowtie-open-dashdot', animate='spin', reduce=pca, cmap='mako', color=data)

    # color by k-means cluster
    plot_test('fig69', data, '8-dash', animate='spin', reduce=pca, cmap='rocket', cluster=kmeans)

    # color by mixture component
    plot_test('fig70', data, 'H-dot', animate='spin', reduce=pca, cmap='crest', cluster=gaussian_mixture)

    # 3d plots with spin animations
    # color by weights
    plot_test('fig71', data, 'dashdot', animate='spin', cmap='dark:salmon_r', color=data)

    # color by k-means cluster
    plot_test('fig72', data, '.-dash', animate='spin', cmap='YlOrBr', cluster=kmeans)

    # color by mixture component
    plot_test('fig73', data, ',-dot', animate='spin', cmap='icefire', cluster=gaussian_mixture)


def test_bounding_box():
    # 2d plots
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 2}}
    # static
    plot_test('fig74', data, bounding_box=True, reduce=pca)

    # animated
    plot_test('fig75', data, bounding_box=True, reduce=pca, animate=True)

    # 3d plots
    # static
    plot_test('fig76', data, bounding_box=True)

    # animated
    plot_test('fig77', data, bounding_box=True, animate='chemtrails')


def test_backend_management():
    # not implemented
    pass
