import requests
import pandas as pd
import deepdish as dd
import os
import pickle
import warnings
from .analyze import analyze
from ..datageometry import DataGeometry


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


def load(dataset, reduce=None, ndims=None, align=None, normalize=None):
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

    Returns
    ----------
    data : Numpy Array
        Example data

    """
    if dataset in EXAMPLE_DATA.keys():
        geo_data = _load_example_data(dataset)
        if dataset.endswith('_model'):
            # geo_data is a sklearn.pipeline.Pipeline, not a DataGeometry
            return geo_data
        elif dataset == 'mushrooms':
            # format mushrooms dataset as a pandas DataFrame
            geo_data.data = pd.DataFrame(geo_data.data)
    else:
        dataset_path = Path(expanduser(expandvars(dataset))).resolve()
        if not dataset_path.is_file():
            raise HypertoolsIOError(
                f"[Errno 2] No such file or directory: {dataset_path}. "
                "Please specify a .geo file or one of the following sample "
                "files: 'weights', 'weights_avg', 'weights_sample', 'spiral', "
                "'mushrooms', 'wiki', 'nips', 'sotus', 'wiki_model', "
                "'nips_model', 'sotus_model'"
            )
        elif legacy:
            geo_data = _load_legacy(dataset_path)
        else:
            try:
                geo_data = pickle.loads(dataset_path.read_bytes())
            except pickle.UnpicklingError as e:
                raise HypertoolsIOError(
                    "Failed to load DataGeometry object from "
                    f"{dataset_path}. If {dataset_path.name} was created "
                    "with hypertools<0.8.0, pass legacy=True to load it."
                ) from e
            if isinstance(geo_data.data, dict):
                geo_data.data = pd.DataFrame(geo_data.data)

    if isinstance(geo_data.data, dict):
        geo_data.data = pd.DataFrame(geo_data.data)

    if any({reduce, ndims, align, normalize}):
        reduce = reduce or 'IncrementalPCA'
        d = analyze(geo_data.get_data(),
                    reduce=reduce,
                    ndims=ndims,
                    align=align,
                    normalize=normalize)
        return plot(d, show=False)
    return geo_data

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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            geo = dd.io.load(fullpath)
        if 'dtype' in geo:
            if 'list' in geo['dtype']:
                geo['data'] = list(geo['data'])
            elif 'df' in geo['dtype']:
                geo['data'] = pd.DataFrame(geo['data'])
        geo['xform_data'] = list(geo['xform_data'])
        return DataGeometry(**geo)
