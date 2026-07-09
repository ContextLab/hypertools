#!/usr/bin/env python

import warnings
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from .reduce import reduce as reducer
from ..tools.format_data import format_data as formatter


def describe(x, reduce='IncrementalPCA', max_dims=None, show=True,
             format_data=True, backend='auto'):
    """
    Create plot describing covariance with as a function of number of dimensions

    This function correlates the raw data with reduced data to get a sense
    for how well the data can be summarized with n dimensions.  Useful for
    evaluating quality of dimensionality reduced plots.

    Parameters
    ----------

    x : Numpy array, DataFrame or list of arrays/dfs
        A list of Numpy arrays or Pandas Dataframes

    reduce : str, dict, class, instance, or fitted Reducer
        Decomposition/manifold learning model to use (default:
        'IncrementalPCA'). Models supported: PCA, IncrementalPCA, SparsePCA,
        MiniBatchSparsePCA, KernelPCA, FastICA, FactorAnalysis, TruncatedSVD,
        DictionaryLearning, MiniBatchDictionaryLearning, TSNE, Isomap,
        SpectralEmbedding, LocallyLinearEmbedding, MDS, and UMAP; the mixture
        models GaussianMixture, BayesianGaussianMixture,
        LatentDirichletAllocation and NMF (GH #174); and the torch autoencoders
        Autoencoder, DeepAutoencoder, SparseAutoencoder,
        ConvolutionalAutoencoder, SequenceAutoencoder and
        VariationalAutoencoder (GH #162, `pip install "hypertools[torch]"`).
        Can be passed as a string, or for finer control as a dictionary, e.g.
        reduce={'model': 'PCA', 'kwargs': {'whiten': True}}. See scikit-learn
        model docs for details on parameters supported for each model.

    max_dims : int
        Maximum number of dimensions to consider

    show : bool
        Plot the result (default : true)

    format_data : bool
        Whether or not to first call the format_data function (default: True).

    backend : {'auto', 'matplotlib', 'plotly'}
        Which plotting backend to draw the correlation-vs-dimensions figure
        with when `show=True` (default: 'auto', resolved the same way
        `hyp.plot` resolves it -- plotly on Colab/Kaggle when available, else
        matplotlib). The matplotlib figure is a seaborn line plot with the top
        and right spines removed; the plotly figure is an interactive
        `go.Figure` (which has no top/right spines by default). The return
        value (a dict) is identical either way.

    Returns
    ----------

    result : dict
        A dictionary with the analysis results. 'average' is the correlation
        by number of components for all data. 'individual' is a list of lists,
        where each list is a correlation by number of components vector (for each
        input list).

    """

    warnings.warn('When input data is large, this computation can take a long time.')

    def summary(x, max_dims=None):
        """Correlation between full-dimensional and reduced-dimensional pairwise distances, per component count.

        For each number of components from 2 up to `max_dims - 1`,
        reduces `x` to that many dimensions and correlates its pairwise
        distance matrix against the full-dimensional pairwise distance
        matrix (via `get_cdist`/`get_corr`).

        Parameters
        ----------
        x : numpy.ndarray or list of numpy.ndarray
            Data to summarize (stacked via `numpy.vstack` if a list).
        max_dims : int or None, optional
            Maximum number of components to consider. Defaults to
            `min(x.shape)`.

        Returns
        -------
        list of float
            Correlation coefficient for each component count in
            `range(2, max_dims)`.
        """
        # if data is a list, stack it
        if isinstance(x, list):
            x = np.vstack(x)

        # if max dims is not set, make it the length of the minimum number of columns
        if max_dims is None:
            if x.shape[1]>x.shape[0]:
                max_dims = x.shape[0]
            else:
                max_dims = x.shape[1]

        # correlation matrix for all dimensions
        alldims = get_cdist(x)

        corrs=[]
        for dims in range(2, max_dims):
            reduced = get_cdist(reducer(x, ndims=dims, reduce=reduce))
            corrs.append(get_corr(alldims, reduced))
            del reduced
        return corrs

    # common format
    if format_data:
        x = formatter(x, ppca=True)

    # a dictionary to store results
    result = {}
    result['average'] = summary(x, max_dims)
    result['individual'] = [summary(x_i, max_dims) for x_i in x]

    if max_dims is None:
        max_dims = len(result['average'])

    # if show, plot it on the resolved backend
    if show:
        from ..plot.plotly_backend import resolve_backend
        resolved_backend = resolve_backend(backend)
        title = 'Correlation with raw data by number of components'
        if resolved_backend == 'plotly':
            import plotly.graph_objects as go
            pfig = go.Figure()
            for i, trace in enumerate(result['individual']):
                pfig.add_trace(go.Scatter(
                    x=list(range(2, 2 + len(trace))), y=list(trace),
                    mode='lines', opacity=0.7, name=f'dataset {i}',
                    showlegend=len(result['individual']) > 1))
            pfig.update_layout(title=title, xaxis_title='Number of components',
                               yaxis_title='Correlation')
            # plotly axes have no top/right spines by default (Jeremy's despine
            # request is inherent here); return the dict, show the figure
            pfig.show()
        else:
            import pandas as pd
            import seaborn as sns
            df = pd.DataFrame([
                {'components': j + 2, 'correlation': value, 'trace': i}
                for i, trace in enumerate(result['individual'])
                for j, value in enumerate(trace)
            ])
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x='components', y='correlation', units='trace',
                         estimator=None, alpha=0.7, ax=ax)
            ax.set_title(title)
            ax.set_ylabel('Correlation')
            ax.set_xlabel('Number of components')
            # drop the top and right spines (Jeremy's despine request)
            sns.despine(ax=ax, top=True, right=True)
            plt.show()
    return result


def get_corr(reduced, alldims):
    """Pearson correlation coefficient between two flattened distance matrices.

    Parameters
    ----------
    reduced : numpy.ndarray
        First pairwise-distance matrix (e.g. from `get_cdist` on
        reduced-dimensional data).
    alldims : numpy.ndarray
        Second pairwise-distance matrix (e.g. from `get_cdist` on
        full-dimensional data), same shape as `reduced`.

    Returns
    -------
    float
        The Pearson correlation coefficient between the two matrices'
        flattened entries.
    """
    return pearsonr(alldims.ravel(), reduced.ravel())[0]


def get_cdist(x):
    """Pairwise Euclidean distance matrix for the rows of `x`.

    Parameters
    ----------
    x : array-like
        2D array of observations.

    Returns
    -------
    numpy.ndarray
        Square pairwise-distance matrix, `scipy.spatial.distance.cdist(x, x)`.
    """
    return cdist(x, x)
