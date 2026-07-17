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

    x : Numpy array, DataFrame or list/tuple of arrays/dfs
        A list of Numpy arrays or Pandas Dataframes (a tuple of datasets
        is treated exactly like a list). Datasets in a list are stacked
        into one shared feature space, so they must all have the same
        number of columns (ragged lists raise a `ValueError`); `None`
        raises a `TypeError`.

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
        is EXCLUSIVE, matching Python's `range`), so `max_dims` must be an
        integer >= 3 (or None); anything else raises a `ValueError` naming
        the kwarg. Defaults to
        `min(n_observations, n_features)` of the stacked data (floored at
        3, so 2-feature data still evaluates its one meaningful
        dimensionality; 1-feature or single-observation data raises a
        `ValueError` -- there is no dimensionality structure to
        describe). Values
        beyond the data's own dimensionality are clamped with a
        `UserWarning` -- past `min(n_observations, n_features)` the
        correlations just flatline at 1.0, which is not evidence for more
        meaningful components. Note: with `reduce='TSNE'` (default
        `barnes_hut` method), only dimensionalities 2-3 can be evaluated
        -- larger `max_dims` values are clamped with a `UserWarning`; pass
        `reduce={'model': 'TSNE', 'kwargs': {'method': 'exact'}}` to
        evaluate more.

    show : bool
        Plot the result (default: True). The figure is displayed only
        when the resolved backend can show one (plotly, or an interactive
        matplotlib backend); under a non-interactive matplotlib backend
        (e.g. Agg) the figure is still drawn and returned in the result
        dict's 'fig' key, without calling `plt.show()`.

    format_data : bool
        Whether or not to first call the format_data function (default: True).

    backend : {'auto', 'matplotlib', 'plotly'}
        Which plotting backend to draw the correlation-vs-dimensions figure
        with when `show=True`. Validated eagerly (even with `show=False`,
        an unknown backend raises the same "backend must be one of ..."
        `ValueError` as `hyp.plot`). Default: 'auto', resolved the same way
        `hyp.plot` resolves it -- plotly on Colab/Kaggle when available, else
        matplotlib. The matplotlib figure is a seaborn line plot with the top
        and right spines removed; the plotly figure is an interactive
        `go.Figure` (which has no top/right spines by default). Multi-dataset
        inputs get one distinguishable color per dataset, a legend, and the
        'average' curve overlaid, on both backends. The analysis results in
        the returned dict are identical either way (only 'fig' differs: it
        holds the backend's own figure object).

    Returns
    -------

    result : dict
        A dictionary with the analysis results. 'average' is the correlation
        by number of components for all data. 'individual' is a list of lists,
        where each list is a correlation by number of components vector (for each
        input list). 'fig' is the rendered figure handle (a
        `matplotlib.figure.Figure` or `plotly.graph_objects.Figure`,
        depending on the backend) when `show=True` and a figure was drawn;
        otherwise None.

    Examples
    --------
    >>> import numpy as np
    >>> import hypertools as hyp
    >>> x = np.cumsum(np.random.default_rng(0).standard_normal((40, 5)),
    ...               axis=0)
    >>> result = hyp.describe(x, reduce='PCA', max_dims=4, show=False)
    >>> sorted(result.keys())
    ['average', 'fig', 'individual']

    """
    from ..core.shared import require_data
    from ..core.model import external_stacklevel
    # None always raises the unified dispatcher TypeError, and a tuple of
    # datasets is accepted exactly like a list (2026-07 release audit,
    # final wave items 9/15)
    require_data(x, 'describe')
    if isinstance(x, tuple):
        x = list(x)

    # validate max_dims up front (release-1.0 audit, X2-error-quality-017):
    # max_dims=0/-3 silently returned empty results, and a float hit a bare
    # "'float' object cannot be interpreted as an integer" from range()
    # that never named the kwarg. The bound is EXCLUSIVE (range(2,
    # max_dims)), so max_dims must be at least 3 for any component count
    # to be evaluated.
    if max_dims is not None:
        if (isinstance(max_dims, bool)
                or not isinstance(max_dims, (int, np.integer))
                or max_dims < 3):
            raise ValueError(
                f'max_dims must be an integer >= 3 (dimensionalities 2 '
                f'through max_dims - 1 are evaluated; the bound is '
                f'exclusive) or None; got {max_dims!r}.')
        max_dims = int(max_dims)

    # validate backend= eagerly even when show=False (release-1.0 audit,
    # X2-error-quality-017: a bogus backend was silently accepted unless a
    # figure was actually drawn) -- same error hyp.plot raises.
    from ..plot.plotly_backend import resolve_backend as _resolve_backend
    _resolve_backend(backend)

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
    _dim_cap_warned = []  # warn about a too-large max_dims only once

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
            if max_dims < 3:
                # (n, 2) data: the historical default (min(shape)) made the
                # sweep empty -- range(2, 2) -- so describe() silently
                # returned empty results (release-1.0 audit,
                # X2-error-quality-017). Evaluate the one meaningful
                # dimensionality (2) instead.
                max_dims = 3

        # cap the sweep at the data's true dimensionality (release-1.0
        # audit, D14-docs-drift-015): with 8-feature data, max_dims=14
        # used to silently evaluate components 8-13, whose correlations
        # flatline at 1.0 past the true dimensionality -- easily misread
        # as evidence that the data support more meaningful components
        # than they have.
        dim_cap = min(x.shape[0], x.shape[1]) + 1
        if max_dims > dim_cap:
            if not _dim_cap_warned:
                _dim_cap_warned.append(True)
                warnings.warn(
                    f'max_dims={max_dims} exceeds the data dimensionality: '
                    f'with {x.shape[0]} observations x {x.shape[1]} '
                    f'features, at most {dim_cap - 1} components are '
                    f'meaningful. Evaluating dimensionalities '
                    f'2-{dim_cap - 1} instead.', UserWarning,
                    stacklevel=external_stacklevel())
            max_dims = dim_cap

        # TSNE's default barnes_hut method cannot fit n_components >= 4:
        # clamp the sweep (with a warning) instead of crashing at dims=4
        # (F11-reduce-describe-001)
        if tsne_dims_cap is not None and max_dims > tsne_dims_cap:
            warnings.warn(
                "TSNE's default 'barnes_hut' method only supports "
                "n_components <= 3, so describe() will evaluate "
                f"dimensionalities 2-3 instead of 2-{max_dims - 1}; pass "
                "reduce={'model': 'TSNE', 'kwargs': {'method': 'exact'}} "
                "to evaluate more dimensions.", UserWarning,
                stacklevel=external_stacklevel())
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

    # give ragged lists a real error instead of numpy's raw vstack message
    # (2026-07 release audit, final wave item 6, matching reduce/cluster):
    # the summary loop stacks every dataset into one shared feature space
    if isinstance(x, list) and len(x) > 1:
        widths = [np.atleast_2d(np.asarray(xi)).shape[1] for xi in x]
        if len(set(widths)) > 1:
            raise ValueError(
                f"cannot describe a list of datasets with different numbers "
                f"of columns (got column counts {widths}): the datasets are "
                f"stacked and reduced in one SHARED space, which requires "
                f"the same columns in every dataset. Bring them to a common "
                f"set of columns first (e.g. hyp.align, or pad/trim the "
                f"features).")

    # 1-column (or single-observation) data has no dimensionality structure
    # to describe: the sweep would be empty and the result silently useless
    # (release-1.0 audit, X2-error-quality-017)
    _dsets = x if isinstance(x, list) else [x]
    _shapes = [np.atleast_2d(np.asarray(xi)).shape for xi in _dsets]
    if any(s[1] < 2 for s in _shapes):
        raise ValueError(
            f'describe() needs at least 2 features (columns) per dataset to '
            f'evaluate how reduced dimensionalities preserve the data\'s '
            f'structure; got dataset shape(s) {_shapes}. 1-column data has '
            'no dimensionality to reduce.')
    if any(s[0] < 2 for s in _shapes):
        raise ValueError(
            f'describe() needs at least 2 observations (rows) per dataset '
            f'to compute pairwise distances; got dataset shape(s) '
            f'{_shapes}.')

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
            'computation can take a long time.',
            stacklevel=external_stacklevel())

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
                      'and at least 3 features); skipping the figure.',
                      stacklevel=external_stacklevel())
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
            # only call plt.show() when the backend can actually display a
            # window: under Agg/pdf/svg/ps (scripts, CI, doc builds) it
            # just emitted matplotlib's 'FigureCanvasAgg is non-interactive'
            # UserWarning on every default describe() call (release-1.0
            # audit, X4-warnings-006). The figure is returned either way.
            import matplotlib
            if matplotlib.get_backend().lower() not in ('agg', 'pdf',
                                                        'svg', 'ps',
                                                        'template'):
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
