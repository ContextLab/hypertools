#!/usr/bin/env python

from ..reduce.reduce import reduce as reducer
from .align import align as aligner
from .normalize import normalize as normalizer


def analyze(data, normalize=None, reduce=None, ndims=None, align=None, internal=False, impute=None):
    """
    Wrapper function for normalize -> reduce -> align transformations.

    Parameters
    ----------
    data : numpy array, pandas df, or list of arrays/dfs
        The data to analyze

    normalize : str or False or None
        If set to 'across', the columns of the input data will be z-scored
        across lists (default). That is, the z-scores will be computed with
        with respect to column n across all arrays passed in the list. If set
        to 'within', the columns will be z-scored within each list that is
        passed. If set to 'row', each row of the input data will be z-scored.
        If set to False, the input data will be returned with no z-scoring.

    reduce : str or dict
        Decomposition/manifold learning model to use.  Models supported: PCA,
        IncrementalPCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, FastICA,
        FactorAnalysis, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
        TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, and MDS. Can be
        passed as a string, but for finer control of the model parameters, pass
        as a dictionary, e.g. reduce={'model' : 'PCA', 'params' : {'whiten' : True}}.
        See scikit-learn specific model docs for details on parameters supported
        for each model.

    ndims : int
        Number of dimensions to reduce

    align : str or dict
        If str, either 'hyper' or 'SRM'.  If 'hyper', alignment algorithm will be
        hyperalignment. If 'SRM', alignment algorithm will be shared response
        model.  You can also pass a dictionary for finer control, where the 'model'
        key is a string that specifies the model and the params key is a dictionary
        of parameter values (default : 'hyper').

    impute : str, dict, class, class instance or None
        Overrides the default PPCA missing-data fill (applied at the
        `format_data` stage, when normalization triggers it) with a
        different `hypertools.impute` model, e.g. 'Kalman', 'KNNImputer'
        (default: None, i.e. PPCA -- byte-compatible with pre-1.0 behavior).

    Returns
    ----------
    analyzed_data : list of numpy arrays
        The processed data

    """

    # return processed data
    return aligner(reducer(normalizer(data, normalize=normalize, internal=internal,
                                      impute=impute),
                   reduce=reduce, ndims=ndims, internal=internal), align=align)
