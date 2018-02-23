import requests
import dill as pickle
import pandas as pd
import deepdish as dd
import sys
import os
from warnings import warn
from sklearn.externals import joblib
from .analyze import analyze
from .format_data import format_data
from ..datageometry import DataGeometry

BASE_URL = 'https://docs.google.com/uc?export=download'
homedir = os.path.expanduser('~/')
datadir = os.path.join(homedir, 'hypertools_data')

datadict = {
    'weights' : '1VktreKU1SgYMpjsE2yzg8EmfSr-5SbeL',
    'weights_avg' : '1gPOoK6oEeFWY3zAql9MuDxP-oPosMsn0',
    'weights_sample' : '1WDAKRmK4O-NSWP8CiBJ80M-6o_N8LjqQ',
    'spiral' : '15Twh8AjncPN0YIymJ6wvP02quXnI8Pv9',
    'mushrooms' : '1ZoItEheTTwXpKJlqKA2yxfy6gDAhsJWc',
    'wiki_model' : '1IOtLJf5ZnpmPvf2MRP7xAMcNwZruL23M',
    'wiki' : '17mZ8rs_r1KwT9vxPEym6wLCZejJVevlo',
    'sotus' : '1UBjWRLaVR3R_T8IBUf7BN4uAFPTlYv-J',
    'nips' : '1QSP5esFknjaxa_3_XCC0tX_yPh-Dmxcx'
}

def load(dataset, reduce=None, ndims=None, align=None, normalize=None,
         download=True):
    """
    Load a .geo file or example data

    Parameters
    ----------
    dataset : string
        The name of the example dataset.  Can be a `.geo` file, or one of a
        number of example datasets listed below.

        `weights` is list of 2 numpy arrays, each containing average brain
        activity (fMRI) from 18 subjects listening to the same story, fit using
        Hierarchical Topographic Factor Analysis (HTFA) with 100 nodes. The rows
        are fMRI measurements and the columns are parameters of the model.

        `weights_sample` is a sample of 3 subjects from that dataset.

        `weights_avg` is the dataset split in half and averaged into two groups.

        `spiral` is numpy array containing data for a 3D spiral, used to
        highlight the `procrustes` function.

        `mushrooms` is a numpy array comprised of features (columns) of a
        collection of 8,124 mushroomm samples (rows).

        `sotus` is a collection of State of the Union speeches from 1989-2018.

        `wiki` is a collection of wikipedia pages used to fit wiki-model.

        `wiki-model` is a sklearn Pipeline (CountVectorizer->LatentDirichletAllocation)
        trained on a sample of wikipedia articles. It can be used to transform
        text to topic vectors.

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

    if dataset[-4:] == '.geo':
        data = DataGeometry(**dd.io.load(dataset))
    elif dataset in datadict.keys():
        data = _load_data(dataset, datadict[dataset])
    else:
        raise RuntimeError('No data loaded. Please specify a .geo file or '
                       'one of the following sample files: weights, '
                       'weights_avg, weights_sample, spiral, mushrooms or '
                       'wiki.')


    if data is not None:
        if dataset in ('wiki_model'):
            return data
    if isinstance(data, DataGeometry):
        opts = {}
        if reduce:
            opts.update(dict(reduce=reduce))
        if ndims:
            opts.update(dict(ndims=ndims))
        if align:
            opts.update(dict(align=align))
        if normalize:
            opts.update(dict(normalize=normalize))
        if opts:
            return data.plot(data=data.get_data(), show=False, **opts)
        else:
            return data
    else:
        return analyze(data, reduce=reduce, ndims=ndims, align=align, normalize=normalize)


def _load_data(dataset, fileid):
    fullpath = os.path.join(homedir, 'hypertools_data', dataset)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    if not os.path.exists(fullpath):
        data = _load_stream(fileid)
        _download(dataset, data)
    else:
        data = _load_from_disk(dataset)
    return data

def _load_stream(fileid):
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    url = BASE_URL + fileid
    session = requests.Session()
    response = session.get(BASE_URL, params = { 'id' : fileid }, stream = True)
    token = _get_confirm_token(response)
    if token:
        params = { 'id' : fileid, 'confirm' : token }
        response = session.get(BASE_URL, params = params, stream = True)
    # pickle_options = {'encoding': 'latin1'} if sys.version_info[0] == 3 else {}
    return pickle.loads(response.content)

def _download(dataset, data):
    fullpath = os.path.join(homedir, 'hypertools_data', dataset)
    with open(fullpath, 'wb') as f:
        pickle.dump(data, f, protocol=2)

def _load_from_disk(dataset):
    fullpath = os.path.join(homedir, 'hypertools_data', dataset)
    with open(fullpath, 'rb') as f:
        pickle_options = {'encoding': 'latin1'} if sys.version_info[0] == 3 else {}
        return pickle.load(f, **pickle_options)
