#!/usr/bin/env python

import inspect
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
    Describe how well reduced data preserves the raw data's
    pairwise-distance structure, as a function of the number of dimensions

    For each candidate number of dimensions, this function reduces the data
    and Pearson-correlates the flattened pairwise Euclidean distance matrix
    of the reduced data with that of the raw (full-dimensional) data, to
    give a sense for how well the data can be summarized with n dimensions.
    Useful for evaluating quality of dimensionality reduced plots.

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
        Dimensionalities 2 through `max_dims - 1` are evaluated (the bound
        is EXCLUSIVE, matching Python's `range`). Defaults to
        `min(n_observations, n_features)` of the stacked data. Note: with
        `reduce='TSNE'` (default `barnes_hut` method), only dimensionalities
        2-3 can be evaluated -- larger `max_dims` values are clamped with a
        `UserWarning`; pass `reduce={'model': 'TSNE', 'kwargs': {'method':
        'exact'}}` to evaluate more.

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
        `go.Figure` (which has no top/right spines by default). Multi-dataset
        inputs get one distinguishable color per dataset, a legend, and the
        'average' curve overlaid, on both backends. The analysis results in
        the returned dict are identical either way (only 'fig' differs: it
        holds the backend's own figure object).

    Returns
    ----------

    result : dict
        A dictionary with the analysis results. 'average' is the correlation
        by number of components for all data. 'individual' is a list of lists,
        where each list is a correlation by number of components vector (for each
        input list). 'fig' is the rendered figure handle (a
        `matplotlib.figure.Figure` or `plotly.graph_objects.Figure`,
        depending on the backend) when `show=True` and a figure was drawn;
        otherwise None.

    """

    # sklearn TSNE's default 'barnes_hut' method only supports
    # n_components <= 3, so the summary loop crashed at dims=4 for the
    # documented reduce='TSNE' at any realistic max_dims
    # (F11-reduce-describe-001). Detect that combination up front so the
    # loop can clamp its dimension range (with a warning) instead.
    def _tsne_barnes_hut(spec):
        model = spec.get('model') if isinstance(spec, dict) else spec
        if isinstance(model, str):
            name = model
        elif inspect.isclass(model):
            name = model.__name__
        else:
            name = type(model).__name__
        if name != 'TSNE':
            return False
        method = 'barnes_hut'
        if isinstance(spec, dict):
            spec_kwargs = spec.get('kwargs', spec.get('params')) or {}
            method = spec_kwargs.get('method', method)
        elif not isinstance(model, str):
            method = getattr(model, 'method', method)
        return method == 'barnes_hut'

    tsne_dims_cap = 4 if _tsne_barnes_hut(reduce) else None

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

        # TSNE's default barnes_hut method cannot fit n_components >= 4:
        # clamp the sweep (with a warning) instead of crashing at dims=4
        # (F11-reduce-describe-001)
        if tsne_dims_cap is not None and max_dims > tsne_dims_cap:
            warnings.warn(
                "TSNE's default 'barnes_hut' method only supports "
                "n_components <= 3, so describe() will evaluate "
                f"dimensionalities 2-3 instead of 2-{max_dims - 1}; pass "
                "reduce={'model': 'TSNE', 'kwargs': {'method': 'exact'}} "
                "to evaluate more dimensions.", UserWarning)
            max_dims = tsne_dims_cap

        # correlation matrix for all dimensions
        alldims = get_cdist(x)

        corrs=[]
        for dims in range(2, max_dims):
            reduced = get_cdist(reducer(x, ndims=dims, reduce=reduce))
            corrs.append(get_corr(reduced, alldims))
            del reduced
        return corrs

    # common format
    if format_data:
        x = formatter(x, ppca=True)

    # only warn about runtime when the input is actually large: the summary
    # loop's pairwise-distance matrices grow with the square of the number
    # of observations (F11-reduce-describe-011 -- this used to warn
    # unconditionally, even for tiny inputs)
    datasets = x if isinstance(x, list) else [x]
    n_rows = sum(np.asarray(xi).shape[0] for xi in datasets)
    n_elements = sum(np.asarray(xi).size for xi in datasets)
    if n_rows > 1000 or n_elements > 1_000_000:
        warnings.warn(
            f'input data is large ({n_rows} total observations); this '
            'computation can take a long time.')

    # a dictionary to store results
    result = {}
    result['average'] = summary(x, max_dims)
    result['individual'] = [summary(x_i, max_dims) for x_i in x]

    if max_dims is None:
        max_dims = len(result['average'])

    # if show, plot it on the resolved backend. With max_dims < 3 there is no
    # component range to correlate over (range(2, max_dims) is empty), so the
    # result is empty and there is nothing to plot -- warn and skip rather than
    # crash inside seaborn/plotly (QC 2026-07).
    fig = None
    if show and not any(result['individual']):
        warnings.warn('describe() has no components to plot (need max_dims >= 3 '
                      'and at least 3 features); skipping the figure.')
        show = False
    if show:
        from ..plot.plotly_backend import resolve_backend
        resolved_backend = resolve_backend(backend)
        title = 'Correlation with raw data by number of components'
        # only multi-dataset inputs need a legend and the 'average' overlay
        # (for a single dataset, average == individual[0])
        multi = len(result['individual']) > 1
        if resolved_backend == 'plotly':
            import plotly.graph_objects as go
            fig = go.Figure()
            for i, trace in enumerate(result['individual']):
                fig.add_trace(go.Scatter(
                    x=list(range(2, 2 + len(trace))), y=list(trace),
                    mode='lines', opacity=0.7, name=f'dataset {i}',
                    showlegend=multi))
            if multi:
                # the documented headline 'average' curve, overlaid on the
                # per-dataset traces (F11-reduce-describe-012)
                fig.add_trace(go.Scatter(
                    x=list(range(2, 2 + len(result['average']))),
                    y=list(result['average']), mode='lines', name='average',
                    line=dict(color='black', width=3, dash='dash')))
            fig.update_layout(title=title, xaxis_title='Number of components',
                              yaxis_title='Correlation')
            # plotly axes have no top/right spines by default (Jeremy's despine
            # request is inherent here); return the dict, show the figure
            fig.show()
        else:
            import pandas as pd
            import seaborn as sns
            df = pd.DataFrame([
                {'components': j + 2, 'correlation': value,
                 'dataset': f'dataset {i}'}
                for i, trace in enumerate(result['individual'])
                for j, value in enumerate(trace)
            ])
            fig, ax = plt.subplots()
            # one distinguishable color per dataset via hue= -- the old
            # units=/estimator=None draw put every trace in the same color
            # with no legend, so multi-dataset figures were unreadable
            # (F11-reduce-describe-012)
            sns.lineplot(data=df, x='components', y='correlation',
                         hue='dataset', alpha=0.7, ax=ax,
                         legend='auto' if multi else False)
            if multi:
                # the documented 'average' curve (F11-reduce-describe-012)
                ax.plot(range(2, 2 + len(result['average'])),
                        result['average'], color='black', linestyle='--',
                        linewidth=2, label='average')
                ax.legend()
            ax.set_title(title)
            ax.set_ylabel('Correlation')
            ax.set_xlabel('Number of components')
            # drop the top and right spines (Jeremy's despine request)
            sns.despine(ax=ax, top=True, right=True)
            plt.show()
    # hand the figure back so it can be saved/styled/embedded -- describe()
    # used to be the one plotting entry point with no figure handle
    # (F11-reduce-describe-015)
    result['fig'] = fig
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
