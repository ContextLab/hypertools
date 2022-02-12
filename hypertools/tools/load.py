import pickle
from os.path import expanduser, expandvars
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..datageometry import DataGeometry
from .._shared.exceptions import HypertoolsIOError
from .analyze import analyze


BASE_URL = 'https://docs.google.com/uc?export=download'
DATA_DIR = Path.home().joinpath('hypertools_data')

EXAMPLE_DATA = {
    'weights': '1ZXLao5Rxkr45KUMkv08Y1eAedTkpivsd',
    'weights_avg': '1gfI1WB7QqogdYgdclqznhUfxsrhobueO',
    'weights_sample': '1ub-xlYW1D_ASzbLcALcPJuhHUxRwHdIs',
    'spiral': '1nHAusn2VsQinJk35xvJSd7CtWPC1uOwK',
    'mushrooms': '12hmCIZp1tyUoPRHwpiAsm1GDBxiJS8ji',
    'wiki': '1NUqm3svfu2rrFH04xmLbOh0u5WyTe9mh',
    'sotus': '1J0MBhpRwdT2WChfWJ4HXYq6jU4XpyJPm',
    'nips': '1FV7xT2hVgZ1sXfMvAdP1jRsK_dWhp49I',
    'wiki_model': '1T-UAU-6KVGUBcUWqz7yG59vXnThu9T0H',
    'nips_model': '1J0MBhpRwdT2WChfWJ4HXYq6jU4XpyJPm',
    'sotus_model': '16_n9r82pwxzZh-0qdS4a6l0z3v__Q91C'
}


def load(
        dataset,
        reduce=None,
        ndims=None,
        align=None,
        normalize=None,
        *,
        legacy=False
):
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

    legacy : bool
        Pass legacy=True to load DataGeometry objects created with hypertools<0.8.0

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
    else:
        dataset_path = Path(expanduser(expandvars(dataset))).resolve()
        if not dataset_path.is_file():
            raise HypertoolsIOError(
                f"Dataset not found at {dataset_path}. Please specify a .geo "
                "file or one of the following sample files: 'weights', "
                "'weights_avg', 'weights_sample', 'spiral', 'mushrooms', "
                "'wiki', 'nips', 'sotus', 'wiki_model', 'nips_model', "
                "'sotus_model'"
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

    if any({reduce, ndims, align, normalize}):
        from ..plot.plot import plot
        reduce = reduce or 'IncrementalPCA'
        d = analyze(geo_data.get_data(),
                    reduce=reduce,
                    ndims=ndims,
                    align=align,
                    normalize=normalize)
        return plot(d, show=False)
    return geo_data


def _load_legacy(dataset_path):
    try:
        import deepdish as dd
    except ImportError as e:
        # catches ModuleNotFoundError since it's a subclass
        raise HypertoolsIOError(
            "To load legacy-format datasets, install the 'deepdish' module"
        ) from e
    data_dict = dd.io.load(dataset_path)

    if isinstance(data_dict['data'], dict):
        data_dict['data'] = pd.DataFrame(data_dict['data'])
    elif isinstance(data_dict['data'], np.ndarray):
        data_dict['data'] = list(data_dict['data'])
    data_dict['xform_data'] = list(data_dict['xform_data'])
    return DataGeometry(**data_dict)


def _load_example_data(dataset):
    dataset_path = DATA_DIR.joinpath(dataset)
    if not dataset_path.is_file():
        if not DATA_DIR.is_dir():
            DATA_DIR.mkdir()
        _download_example_data(dataset_path)

    try:
        geo_data = pickle.loads(dataset_path.read_bytes())
    except Exception as e:
        raise HypertoolsIOError(
            f"Failed to load '{dataset}' data. Try deleting cached file at"
            f"{dataset_path} and reloading."
        ) from e

    if dataset == 'mushrooms':
        # format mushrooms dataset as a pandas DataFrame
        geo_data.data = pd.DataFrame(geo_data.data)
    return geo_data


def _download_example_data(dataset_path):
    file_id = EXAMPLE_DATA[dataset_path.name]
    session = requests.Session()
    params = {'id': file_id}
    try:
        response = session.get(BASE_URL, params=params, stream=True)
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                # Google Drive requires confirmation for large files
                params['confirm'] = value
                response = session.get(BASE_URL, params=params, stream=True)
                break

        with dataset_path.open('wb') as f:
            # write stream in chunks to avoid loading whole file into memory
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        # clean up partial file in case of error while writing stream
        dataset_path.unlink(missing_ok=True)
        raise HypertoolsIOError(
            f"Failed to download '{dataset_path.name}' dataset"
        ) from e
