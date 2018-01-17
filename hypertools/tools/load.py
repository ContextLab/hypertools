import requests
import pickle
import pandas as pd
import deepdish as dd
import sys
from warnings import warn
from sklearn.externals import joblib
from .analyze import analyze
from .._shared.helpers import format_data
from ..datageometry import DataGeometry

BASE_URL = 'https://docs.google.com/uc?export=download&id='

def load(dataset, reduce=None, ndims=None, align=None, normalize=None):
    """
    Load a .geo file or example data

    Parameters
    ----------
    dataset : string
        The name of the example dataset.  Can be a `.geo` file, or one of a
        number of example datasets listed below. `weights` is list of 2 numpy arrays, each containing average brain activity (fMRI) from 18 subjects listening to the same story, fit using Hierarchical Topographic Factor Analysis (HTFA) with 100 nodes. The rows are fMRI
        measurements and the columns are parameters of the model. `weights_sample` is a sample of 3 subjects from that dataset.  `weights_avg` is the dataset split in half and averaged into two groups. `spiral` is numpy array containing data for a 3D spiral, used to
        highlight the `procrustes` function.  `mushrooms` is a numopy array
        comprised of features (columns) of a collection of 8,124 mushroomm samples (rows).

    normalize : str or False or None
        If set to 'across', the columns of the input data will be z-scored
        across lists (default). That is, the z-scores will be computed with
        with repect to column n across all arrays passed in the list. If set
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

    Returns
    ----------
    data : Numpy Array
        Example data

    """
    data = None

    if dataset[-4:] == '.geo':
        geo = dd.io.load(dataset)
        data = DataGeometry(fig=None, ax=None, data=geo['data'],
                            xform_data=geo['xform_data'], line_ani=None,
                            reduce=geo['reduce'], align=geo['align'],
                            normalize=geo['normalize'], kwargs=geo['kwargs'],
                            version=geo['version'])
    elif dataset is 'weights':
        data = _load_stream('0B7Ycm4aSYdPPREJrZ2stdHBFdjg')
    elif dataset is 'weights_avg':
        data = _load_stream('0B7Ycm4aSYdPPRmtPRnBJc3pieDg')
    elif dataset is 'weights_sample':
        data = _load_stream('0B7Ycm4aSYdPPTl9IUUVlamJ2VjQ')
    elif dataset is 'spiral':
        data = _load_stream('0B7Ycm4aSYdPPQS0xN3FmQ1FZSzg')
    elif dataset is 'mushrooms':
        data = pd.read_csv(BASE_URL + '0B7Ycm4aSYdPPY3J0U2tRNFB4T3c')

    if data is not None:
        return analyze(data, reduce=reduce, ndims=ndims, align=align, normalize=normalize)
    else:
        raise RuntimeError('No data loaded. Please specify a .geo file or '
                           'one of the following sample files: weights, '
                           'weights_avg, weights_sample, spiral, mushrooms')


def _load_stream(fileid):
    url = BASE_URL + fileid

    pickle_options = {'encoding': 'latin1'} if sys.version_info[0] == 3 else {}

    return pickle.loads(requests.get(url, stream=True).content, **pickle_options)
