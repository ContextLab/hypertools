import numpy as np
import pandas as pd

import hypertools as hyp


weights = hyp.load('weights')


def test_static_plot2d():
    # test lines, markers
    # test different strategies for managing color
    # test various manipulations (align, cluster, manip, reduce)
    pca = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 2}}

    # lines
    # fig1a = hyp.plot(weights, reduce=pca)
    # fig1b = hyp.plot(weights, '-', reduce=pca)
    # fig2 = hyp.plot(weights, reduce=pca, color=weights)

    # markers
    # fig3 = hyp.plot(weights, ',', reduce=pca)
    # fig4 = hyp.plot(weights, '.', reduce=pca)
    # fig5 = hyp.plot(weights, 'o', reduce=pca, color=weights)

    # lines + markers
    # fig6 = hyp.plot(weights, '-.', reduce=pca)
    # fig7 = hyp.plot(weights, ':o', reduce=pca, color=weights)

    # zscore, align, cluster
    kmeans = {'model': 'KMeans', 'args': [], 'kwargs': {'n_clusters': 5}}
    # fig8 = hyp.plot(weights, '*-', reduce=pca, cluster=kmeans, align='HyperAlign', manip='ZScore')
    # fig9 = hyp.plot(weights, 'D--', reduce=pca, align='SharedResponseModel', manip=['ZScore', 'Resample', 'Smooth'])

    # pipeline
    pca_10d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 10}}
    pca_5d = {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 5}}
    fig10 = hyp.plot(weights, 'star-triangle-down-open-dot--', reduce=pca, pipeline=[pca_10d, pca_5d])

    # TODO: need to create reference versions of all figures and compare with produced versions...
    pass


test_static_plot2d()


def test_animated_plot2d():
    pass


def test_static_plot3d():
    pass


def test_animated_plot3d():
    # test each type of animation
    # also test saving in various formats
    pass


def test_backend_management():
    # not implemented
    pass
