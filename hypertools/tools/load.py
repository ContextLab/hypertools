import requests
import pandas as pd
import deepdish as dd
import sys
import os
from warnings import warn
from .analyze import analyze
from .format_data import format_data
from ..datageometry import DataGeometry
from .._shared.helpers import check_geo
import pickle

BASE_URL = 'https://docs.google.com/uc?export=download'
homedir = os.path.expanduser('~/')
datadir = os.path.join(homedir, 'hypertools_data')

datadict = {
    'weights' : '1-zzaUMHuXHSzFcGT4vNlqcV8tMY4q7jS',
    'weights_avg' : '1v_IrU6n72nTOHwD3AnT2LKgtKfHINyXt',
    'weights_sample' : '1CiVSP-8sjdQN_cdn3uCrBH5lNOkvgJp1',
    'spiral' : '1JB4RIgNfzGaTFWRBCzi8CQ2syTE-BnWg',
    'mushrooms' : '1wRXObmwLjSHPAUWC8QvUl37iY2qRObg8',
    'wiki' : '1e5lCi17bLbOXuRjiGO2eqkEWVpeCuRvM',
    'sotus' : '1D2dsrLAXkC3eUUaw2VV_mldzxX5ufmkm',
    'nips' : '1Vva4Xcc5kUX78R0BKkLtdCWQx9GI-FG2',
    'wiki_model' : '1OrN1F39GkMPjrB2bOTgNRT1pNBmsCQsN',
    'nips_model' : '1orgxWJdWYzBlU3EF2u7EDsZrp3jTNNLG',
    'sotus_model' : '1g2F18WLxfFosIqhiLs79G0MpiG72mWQr'
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
        geo = dd.io.load(dataset)
        if 'dtype' in geo:
            if 'list' in geo['dtype']:
                geo['data'] = list(geo['data'])
            elif 'df' in geo['dtype']:
                geo['data'] = pd.DataFrame(geo['data'])
        geo['xform_data'] = list(geo['xform_data'])
        data = DataGeometry(**geo)
    elif dataset in datadict.keys():
        data = _load_data(dataset, datadict[dataset])
    else:
        raise RuntimeError('No data loaded. Please specify a .geo file or '
                       'one of the following sample files: weights, '
                       'weights_avg, weights_sample, spiral, mushrooms or '
                       'wiki.')


    if data is not None:
        if dataset in ('wiki_model', 'nips_model', 'sotus_model',):
            return data
    if isinstance(data, DataGeometry):
        # data = check_geo(data)
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
        try:
            _download(dataset, _load_stream(fileid))
            data = _load_from_disk(dataset)
        except:
            raise ValueError('Download failed.')
    else:
        try:
            data = _load_from_disk(dataset)
        except:
            try:
                _download(dataset, _load_stream(fileid))
                data = _load_from_disk(dataset)
            except:
                raise ValueError('Download failed. Try deleting cache data in'
                                 ' /Users/homedir/hypertools_data.')
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
    return response

def _download(dataset, data):
    fullpath = os.path.join(homedir, 'hypertools_data', dataset)
    with open(fullpath, 'wb') as f:
        f.write(data.content)

def _load_from_disk(dataset):
    fullpath = os.path.join(homedir, 'hypertools_data', dataset)
    if dataset in ('wiki_model', 'nips_model', 'sotus_model',):
        try:
            with open(fullpath, 'rb') as f:
                return pickle.load(f)
        except ValueError as e:
            print(e)
    else:
        geo = dd.io.load(fullpath)
        if 'dtype' in geo:
            if 'list' in geo['dtype']:
                geo['data'] = list(geo['data'])
            elif 'df' in geo['dtype']:
                geo['data'] = pd.DataFrame(geo['data'])
        geo['xform_data'] = list(geo['xform_data'])
        return DataGeometry(**geo)
