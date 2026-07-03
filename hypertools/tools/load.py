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
    'sotus_model': '16_n9r82pwxzZh-0qdS4a6l0z3v__Q91C',
    # "shapes zoo" 3D point clouds + the datasaurus dozen (hosted on
    # Dropbox; full-URL entries are fetched directly). NOTE: the
    # 'egyption_mask' source file is an empty (0, 3) array upstream, so it
    # is intentionally not registered.
    'bunny': 'https://www.dropbox.com/s/7d9vo9idqk1hn31/bunny.pkl?dl=1',
    'cube': 'https://www.dropbox.com/s/tkrwe2m4maxl83j/cube.pkl?dl=1',
    'dragon': 'https://www.dropbox.com/s/6w84icbvzh5oilr/dragon.pkl?dl=1',
    'sphere': 'https://www.dropbox.com/s/wp8suye6oh4ze3u/sphere.pkl?dl=1',
    'teapot': 'https://www.dropbox.com/s/f3jj18h3ge2gns6/teapot.pkl?dl=1',
    'vase': 'https://www.dropbox.com/s/prquc7ov18zguuu/vase.pkl?dl=1',
    'biplane': 'https://www.dropbox.com/s/4b9y9ouvjpjbj6x/biplane.pkl?dl=1',
    'datasaurus':
        'https://www.dropbox.com/s/6wxjyw8p052a5t9/datasaurus.pkl?dl=1',
}


def load(
        dataset,
        reduce=None,
        ndims=None,
        align=None,
        normalize=None,
        *,
        legacy=False,
        split=None,
        streaming=False
):
    """
    Load data from a built-in example dataset, a local file, a Hugging Face
    dataset, Google Drive, Dropbox, or any URL

    A string is interpreted by trying, in order:

    1. a built-in example dataset name (listed below)
    2. a path to a local file (.geo/pickle, .npy/.npz, .csv/.tsv/.txt,
       .json, .parquet, .mat)
    3. a Hugging Face dataset id such as ``'scikit-learn/iris'``
       (pass ``streaming=True`` for a streaming dataset, which can be
       passed straight to :func:`hypertools.plot`)
    4. a Google Drive URL or bare file id
    5. a Dropbox URL or shared-link path
    6. any other URL, with or without an ``https://`` scheme

    A **list of strings** resolves element-wise and returns a list of
    datasets that can be passed to any hypertools function.

    .. warning::
        Pickled payloads (``.pkl``/``.geo``) can execute arbitrary code
        when loaded -- only load pickled data from sources you trust.

    Parameters
    ----------
    dataset : string or list of strings
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

        The "shapes zoo" datasets -- `bunny`, `cube`, `dragon`, `sphere`,
        `teapot`, `vase`, and `biplane` -- are 3D point clouds of the
        corresponding objects (numpy arrays / DataFrames of x, y, z
        coordinates), useful for demonstrating alignment and plotting.

        `datasaurus` is the "Datasaurus Dozen": a list of 2D datasets with
        near-identical summary statistics but wildly different shapes.

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

    split : string or None
        Hugging Face datasets only: which split to load (default: the
        'train' split if present, otherwise the first available split).

    streaming : bool
        Hugging Face datasets only: if True, return a streaming
        ``IterableDataset`` instead of materializing the data (see
        https://huggingface.co/docs/datasets/en/stream). The result can be
        passed directly to :func:`hypertools.plot`.

    Returns
    ----------
    data : DataGeometry, numpy array, DataFrame, list, or IterableDataset
        The loaded data (a list of datasets when a list of strings was
        passed).

    """
    # lists of strings resolve element-wise to a list of datasets
    if isinstance(dataset, (list, tuple)):
        return [load(d, reduce=reduce, ndims=ndims, align=align,
                     normalize=normalize, legacy=legacy, split=split,
                     streaming=streaming)
                for d in dataset]

    if dataset in EXAMPLE_DATA.keys():
        geo_data = _load_example_data(dataset)
        if dataset.endswith('_model'):
            # geo_data is a sklearn.pipeline.Pipeline, not a DataGeometry
            return geo_data
    else:
        dataset_path = Path(expanduser(expandvars(dataset))).resolve()
        if dataset_path.is_file():
            if legacy:
                geo_data = _load_legacy(dataset_path)
            else:
                geo_data = _load_local(dataset_path)
        else:
            # resolution chain: Hugging Face -> Google Drive -> Dropbox ->
            # generic URL (see tools.sources)
            from .sources import load_source
            geo_data = load_source(dataset, split=split,
                                   streaming=streaming)

    from .streaming import is_stream
    if is_stream(geo_data):
        if any({reduce, ndims, align, normalize}):
            raise ValueError(
                'reduce/ndims/align/normalize cannot be applied to a '
                'streaming dataset at load time; pass the stream to '
                'hypertools.plot(), which fits models on the first '
                'stream_init samples')
        return geo_data

    if any({reduce, ndims, align, normalize}):
        from ..plot.plot import plot
        reduce = reduce or 'IncrementalPCA'
        # shapes-zoo/datasaurus entries are plain arrays/DataFrames/lists
        # rather than DataGeometry objects
        raw = geo_data.get_data() if hasattr(geo_data, 'get_data') \
            else geo_data
        d = analyze(raw,
                    reduce=reduce,
                    ndims=ndims,
                    align=align,
                    normalize=normalize)
        return plot(d, show=False)
    return geo_data


def _load_local(dataset_path):
    """Load a local file: pickled DataGeometry objects keep their historical
    behavior (including the legacy=True hint); everything else goes through
    the extension/sniff-based parser (npy/npz/csv/tsv/txt/json/parquet/mat).
    """
    raw = dataset_path.read_bytes()
    looks_pickled = dataset_path.suffix.lower() in (
        '.geo', '.pkl', '.pickle', '.p') or raw[:1] == b'\x80'
    if looks_pickled:
        try:
            geo_data = pickle.loads(raw)
        except pickle.UnpicklingError as e:
            raise HypertoolsIOError(
                "Failed to load DataGeometry object from "
                f"{dataset_path}. If {dataset_path.name} was created "
                "with hypertools<0.8.0, pass legacy=True to load it."
            ) from e
        if isinstance(geo_data, DataGeometry) and \
                isinstance(geo_data.data, dict):
            geo_data.data = pd.DataFrame(geo_data.data)
        return geo_data
    from .sources import load_local_file
    return load_local_file(dataset_path)


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
        geo_data = _unpickle_example(dataset_path)
    except Exception:
        # a corrupt cache (e.g. an HTML error page saved by an interrupted
        # or rate-limited download) poisons every subsequent load -- delete
        # it and retry the download once before giving up
        dataset_path.unlink(missing_ok=True)
        _download_example_data(dataset_path)
        try:
            geo_data = _unpickle_example(dataset_path)
        except Exception as e:
            dataset_path.unlink(missing_ok=True)
            raise HypertoolsIOError(
                f"Failed to load '{dataset}' data after re-downloading. "
                "The download source may be temporarily unavailable or "
                "rate-limited; please try again later."
            ) from e

    if dataset == 'mushrooms':
        # format mushrooms dataset as a pandas DataFrame
        geo_data.data = pd.DataFrame(geo_data.data)
    return geo_data


def _download_example_data(dataset_path, max_attempts=4):
    """Download an example dataset, retrying with backoff when the host
    rate-limits (Google Drive answers rate-limited requests with an HTML
    error page and a 200 status)."""
    import time

    last_error = None
    for attempt in range(max_attempts):
        if attempt > 0:
            # 2s, 6s, 18s -- long enough for transient Drive rate limits
            time.sleep(2 * 3 ** (attempt - 1))
        try:
            _download_example_data_once(dataset_path)
            return
        except HypertoolsIOError as e:
            last_error = e
    raise last_error


def _download_example_data_once(dataset_path):
    source = EXAMPLE_DATA[dataset_path.name]
    session = requests.Session()
    try:
        if source.startswith('http'):
            # full-URL entries (e.g. Dropbox ?dl=1 direct downloads)
            response = session.get(source, stream=True)
        else:
            # legacy entries are Google Drive file ids
            params = {'id': source}
            response = session.get(BASE_URL, params=params, stream=True)
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    # Google Drive requires confirmation for large files
                    params['confirm'] = value
                    response = session.get(BASE_URL, params=params,
                                           stream=True)
                    break

        response.raise_for_status()
        with dataset_path.open('wb') as f:
            # write stream in chunks to avoid loading whole file into memory
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

        # Google Drive answers rate-limited/oversized requests with an HTML
        # page and a 200 status; caching it would poison every later load.
        # All hypertools example datasets are pickles, which never start
        # with '<'.
        with dataset_path.open('rb') as f:
            if f.read(1) == b'<':
                dataset_path.unlink(missing_ok=True)
                raise HypertoolsIOError(
                    f"Download of '{dataset_path.name}' returned an error "
                    "page instead of the dataset (the host may be "
                    "rate-limiting requests). Please try again later."
                )
    except HypertoolsIOError:
        raise
    except Exception as e:
        # clean up partial file in case of error while writing stream
        dataset_path.unlink(missing_ok=True)
        raise HypertoolsIOError(
            f"Failed to download '{dataset_path.name}' dataset"
        ) from e


def _unpickle_example(dataset_path):
    """Load a cached example dataset, tolerating the formats used across
    hypertools' history: standard pickles (DataGeometry objects), pickles
    created with old pandas versions (pd.read_pickle applies compatibility
    shims), and dill-serialized arrays (the shapes-zoo datasets)."""
    raw = dataset_path.read_bytes()
    try:
        return pickle.loads(raw)
    except Exception:
        pass
    try:
        return pd.read_pickle(dataset_path)
    except Exception:
        pass
    import dill
    return dill.loads(raw)
