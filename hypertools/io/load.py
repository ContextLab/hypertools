import os
import pickle
import warnings
from os.path import expanduser, expandvars
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..datageometry import DataGeometry
from ..core.exceptions import HypertoolsIOError
from ..tools.analyze import analyze


BASE_URL = 'https://docs.google.com/uc?export=download'
DATA_DIR = Path.home().joinpath('hypertools_data')

EXAMPLE_DATA = {
    'weights': '1ZXLao5Rxkr45KUMkv08Y1eAedTkpivsd',
    'weights_avg': '1gfI1WB7QqogdYgdclqznhUfxsrhobueO',
    'weights_sample': '1ub-xlYW1D_ASzbLcALcPJuhHUxRwHdIs',
    'spiral': '1nHAusn2VsQinJk35xvJSd7CtWPC1uOwK',
    'mushrooms': '12hmCIZp1tyUoPRHwpiAsm1GDBxiJS8ji',
    'wiki': '1NUqm3svfu2rrFH04xmLbOh0u5WyTe9mh',
    # 'sotus' is loaded via datawrangler's text zoo (see
    # _load_sotus_corpus), NOT downloaded from this registry: the
    # historical Drive id here had been duplicated with 'nips_model' (it
    # served a pickled topic-model Pipeline instead of the documented
    # speeches -- QC 2026-07, F18-load-hosted-001), and every older
    # 'sotus' Drive id is dead.
    'sotus': 'datawrangler-zoo:sotus',
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

# SHA-256 of each hosted built-in file, pinned so a built-in is verified
# against a hard-coded cryptographic hash BEFORE it is deserialized (2026-07
# release review, blocker #1). A mismatch (a corrupted/rate-limited download,
# a poisoned cache, or a tampered/changed upstream file) is a HARD error --
# never a silent redownload-and-reparse. Every cache hit is validated too.
#
# These pin the current pickle files. When the datasets are re-hosted in
# non-executable formats (.npz/.parquet/.json.gz -- see the verified
# conversion bundle handed off for Dropbox hosting), swap each EXAMPLE_DATA
# entry to its new URL and replace the hash here with the converted file's
# SHA-256; the .npz/.parquet path then never unpickles at all.
_EXAMPLE_DATA_SHA256 = {
    'weights': '695f50f48328f7b9f5741c89854b07f0c4989c4275f929caa76e95af2c92a7ff',
    'weights_avg': '52be2d02d2c5754adbb58e68f86d2c2da2b7a339162f1d2e0c7e3b987ffde06f',
    'weights_sample': 'eaf67c631e9cc8207c70ad1c93c6c022298a6e57f946ef39e24299c9c1bf3f8d',
    'spiral': '7ca728d2972cb0271b3c68693aa7ec744962f8499043120eeefc6b755591f94c',
    'mushrooms': 'b3abdaf8ae1597eeb95c1f1bc6cff6c38d02c9dff99a66ebafed6dc168d2c8cf',
    'bunny': '7a43745c17834d54bb9dc10b7c286b4f23a4a1c437f8419d53dbe2eaf6ece663',
    'cube': 'ca43191a3c77ce90d449a9cd327a53aaa7bd55032c7de06567c175d6524a02c1',
    'dragon': 'dbfdbbc077f3884251a7140ee030eaf29cff915448d68e3afd96780e5cf79434',
    'sphere': '8dae53277e2f15a57b3ca00299b6e7b980dcde6524c17350ad3b0cc3b3e0688f',
    'teapot': 'c195e6221ad369b274d5f531b98a763c8fe03efadfc5d582011b3148fbf35973',
    'vase': 'b1ef3da871ae93f1a661cc432cc70a2b662cc98748173b44457f838aee493e0f',
    'biplane': 'f5e5661c2eea7a03f30229d6df5546bdd1a9df9e578c865dcefee983801fc814',
    'datasaurus': '7ce78b634ef299098c75445bfc8f28f3edf122b415cdcc179ffda11b2e0bd126',
    'wiki': '722d20a286edfad607904123d7756b95fb49e72e037af5d091422c994c4893be',
    'nips': 'e240532dab310652bb489b4f0880af9f681652708dfe60ac3d6ff4e4ee4aaffc',
    'wiki_model': '5ec3c34e2524e105a90ae498cca809d61ddfa90813a4621de65b37275fd515c9',
    'nips_model': '4f93308a48002730866659bda7ef393f5451dc8360b9e3c91c9cf5d77f73a762',
    'sotus_model': 'a7b085f7f6d94dbed6d961a1950de18a07b56456c77c2495a2868a9fefb07aa4',
}


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _integrity_ok(dataset_path, name):
    """True when the file matches its pinned SHA-256 (or the dataset is not
    integrity-pinned, e.g. 'sotus' which loads via datawrangler)."""
    pinned = _EXAMPLE_DATA_SHA256.get(name)
    if pinned is None:
        return True
    return _sha256_file(dataset_path) == pinned


def load(
        dataset,
        reduce=None,
        ndims=None,
        align=None,
        normalize=None,
        *,
        legacy=False,
        split=None,
        streaming=False,
        trust=False
):
    """
    Load data from a built-in example dataset, a scikit-learn or seaborn
    named dataset, a local file, a Hugging Face dataset, Google Drive,
    Dropbox, or any URL

    A string is interpreted by trying, in order:

    1. a built-in example dataset name (listed below)
    2. a scikit-learn bundled dataset name -- ``'iris'``, ``'digits'``,
       ``'wine'``, ``'breast_cancer'``, ``'diabetes'``, or ``'linnerud'``
       (the small datasets shipped with ``sklearn.datasets``; the
       network-fetched ``fetch_*`` datasets are not included). Loaded via
       the corresponding ``sklearn.datasets.load_*`` function and returned
       as a DataFrame of the features with the target appended as a
       ``'target'`` column (for multi-output targets, e.g. ``'linnerud'``,
       one column per target name instead)
    3. a seaborn dataset name -- any name returned by
       ``seaborn.get_dataset_names()`` (e.g. ``'penguins'``, ``'tips'``,
       ``'titanic'``), loaded via ``seaborn.load_dataset()`` and returned
       unchanged. This is a network lookup (cached per-process); if it
       can't reach the seaborn-data repo, this step is skipped
    4. a FiveThirtyEight dataset, explicit prefix
       ``'fivethirtyeight/<slug>'`` (e.g. ``'fivethirtyeight/bechdel'``),
       where ``<slug>`` is the dataset's folder in
       https://github.com/fivethirtyeight/data. The folder's CSV file(s)
       are downloaded from raw.githubusercontent.com: a single CSV
       becomes a DataFrame, multiple CSVs become a dict of
       ``{filename: DataFrame}``
    5. a Kaggle dataset, explicit prefix ``'kaggle/<owner>/<dataset>'``
       (e.g. ``'kaggle/uciml/iris'``), downloaded anonymously via
       ``kagglehub.dataset_download`` (requires the optional
       ``kagglehub`` dependency -- ``pip install hypertools[kaggle]``).
       Every CSV/TSV file in the dataset is loaded the same way as step 4
    6. a path to a local file (.geo/pickle, .npy/.npz, .csv/.tsv/.txt,
       .json, .parquet, .mat, .xlsx/.xls; gzip-compressed variants (.gz)
       are decompressed transparently). Files with no extension are
       parsed by content sniffing; files with any other extension raise
       an error unless their content matches a recognized binary format
       (pickle/npy/zip)
    7. a Hugging Face dataset id such as ``'scikit-learn/iris'``
       (pass ``streaming=True`` for a streaming dataset, which can be
       passed straight to :func:`hypertools.plot`)
    8. a Google Sheets URL (``docs.google.com/spreadsheets/d/<id>``),
       loaded via its CSV export
    9. a Google Drive URL or bare file id (large files behind Drive's
       "can't scan this file for viruses" interstitial are followed
       automatically)
    10. a Dropbox URL or shared-link path
    11. any other URL, with or without an ``https://`` scheme

    .. note::
        Precedence: a built-in example dataset name (step 1) always wins,
        even over a same-named scikit-learn/seaborn dataset. Between
        scikit-learn and seaborn, scikit-learn wins -- e.g. ``'iris'``
        resolves to scikit-learn's ``load_iris`` (columns like
        ``'sepal length (cm)'``), not seaborn's ``'iris'`` dataset
        (columns like ``'sepal_length'``), since both define an ``'iris'``
        name. Because these resolvers run before local-file resolution, a
        local file whose name (without an extension) matches a
        scikit-learn or seaborn dataset name is shadowed -- pass a path
        with an extension, or an absolute/relative path containing a
        ``/``, to force local-file resolution.

        The ``'fivethirtyeight/'`` and ``'kaggle/'`` prefixes (steps 4-5)
        are explicit: a name starting with one of them is always treated
        as that source, so a same-named relative local path (e.g. a local
        file ``fivethirtyeight/bechdel``) is shadowed -- prepend ``'./'``
        to force local-file resolution. For the same reason, a prefixed
        name that then fails (unknown slug/dataset id, no CSV/TSV files
        found, malformed id) raises immediately instead of falling
        through to the remaining steps.

    Examples
    --------
    >>> import hypertools
    >>> hypertools.load('iris').columns.tolist()  # doctest: +NORMALIZE_WHITESPACE
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
     'petal width (cm)', 'target']
    >>> hypertools.load('penguins').columns.tolist()  # doctest: +NORMALIZE_WHITESPACE
    ['species', 'island', 'bill_length_mm', 'bill_depth_mm',
     'flipper_length_mm', 'body_mass_g', 'sex']
    >>> hypertools.load('fivethirtyeight/bechdel').shape  # 538's bechdel data
    (1794, 15)
    >>> hypertools.load('kaggle/uciml/iris').shape  # a Kaggle dataset
    (150, 6)
    >>> weights = hypertools.load('weights')  # built-in name always wins
    >>> type(weights).__name__, len(weights)
    ('list', 36)

    A **list of strings** resolves element-wise and returns a list of
    datasets that can be passed to any hypertools function.

    .. warning::
        Pickled payloads (``.pkl``/``.geo``) can execute arbitrary code
        when loaded -- only load pickled data from sources you trust.

    Parameters
    ----------
    dataset : string, path-like, or list of strings
        The name of a built-in example dataset (listed below), a dataset
        name resolvable per the steps above, or a file path / URL.

        `weights` is a list of numpy arrays, one PER SUBJECT (36 arrays,
        each 300 timepoints x 100 parameters, float32), containing brain
        activity (fMRI) from subjects listening to the same story, fit
        using Hierarchical Topographic Factor Analysis (HTFA) with 100
        nodes; each array's rows are timepoints and its columns are model
        parameters.

        `weights_sample` is a sample of 3 subjects from that dataset.

        `weights_avg` is a 2-array group-averaged variant of the same
        experiment: a list of two (100, 100) arrays, one per group.

        `spiral` is a list of two (1000, 3) numpy arrays containing 3D
        spiral data, used to highlight the `procrustes` function.

        `mushrooms` is a pandas DataFrame of categorical features
        (columns) describing 8,124 mushroom samples (rows).

        `sotus` is a list of 29 State of the Union addresses (1989-2018),
        as strings (loaded via the ``datawrangler`` text zoo).

        `wiki` is a list holding one (3136, 1) numpy object array of
        wikipedia page texts, used to fit `wiki_model`.

        `nips` is a list holding one (7241, 1) numpy object array of NIPS
        conference paper texts (~181 MB download), used to fit
        `nips_model`.

        `wiki_model`, `nips_model`, and `sotus_model` are sklearn
        Pipelines (CountVectorizer -> LatentDirichletAllocation, 50
        topics) trained on the wiki, nips, and sotus corpora,
        respectively; each transforms text into 50-dimensional topic
        vectors. (The hosted files were pickled under an older
        scikit-learn; hypertools backfills newer estimator attributes on
        load so ``repr()``/``get_params()``/``transform()`` work under
        the installed version.)

        The "shapes zoo" datasets -- `bunny`, `cube`, `dragon`, `sphere`,
        `teapot`, `vase`, and `biplane` -- are 3D point clouds of the
        corresponding objects (numpy arrays / DataFrames of x, y, z
        coordinates), useful for demonstrating alignment and plotting.

        `datasaurus` is the "Datasaurus Dozen": a list of 2D datasets with
        near-identical summary statistics but wildly different shapes.

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

    ndims : int
        Number of dimensions to reduce

    align : str, dict, False, or None
        Alignment model to bring a list of datasets into a shared space. If
        str, 'hyper' (hyperalignment) or 'SRM' (shared response model). You
        can also pass a dictionary for finer control, where 'model' specifies
        the model and 'kwargs' holds its parameters, e.g.
        align={'model': 'HyperAlign', 'kwargs': {'n_iter': 10}}. If False or
        None, no alignment is applied (default: None).

    normalize : str or False or None
        If set to 'across', the columns of the input data will be z-scored
        across lists. That is, the z-scores will be computed with
        respect to column n across all arrays passed in the list. If set
        to 'within', the columns will be z-scored within each list that is
        passed. If set to 'row', each row of the input data will be z-scored.
        If set to False or None (default), the input data will be returned
        with no z-scoring.

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

    trust : bool
        Remote (non-built-in) sources only. Unpickling a payload fetched
        from a URL/Drive/Dropbox/Sheets executes arbitrary code embedded in
        it, so by default hypertools **refuses** to unpickle remote data
        (raising :class:`~hypertools.core.exceptions.HypertoolsIOError` --
        a warning is not a security boundary). Remote ``.npy``/``.npz``
        payloads are likewise loaded with ``allow_pickle=False`` (raising
        if the array actually needs pickle support, e.g. an object array).
        Pass ``trust=True`` -- only once you have verified the source -- to
        allow unpickling remote data and pickle-backed remote arrays. This
        covers every remote-pickle path (extension-based, content-sniffed,
        and extensionless). Non-executable remote formats
        (``.csv``/``.npz`` numeric/``.parquet``) never require ``trust``.
        Built-in example datasets (listed below) are downloaded from a
        fixed, integrity-checked set of hosts and do not require this flag.
        Local files are never subject to this policy.

    Returns
    -------
    data : numpy array, DataFrame, list, or IterableDataset
        The loaded raw data (a list of datasets when a list of strings was
        passed). If reduce/ndims/align/normalize are supplied, the analyzed
        data is returned directly.

    """
    # lists of strings resolve element-wise to a list of datasets
    if isinstance(dataset, (list, tuple)):
        return [load(d, reduce=reduce, ndims=ndims, align=align,
                     normalize=normalize, legacy=legacy, split=split,
                     streaming=streaming, trust=trust)
                for d in dataset]

    if not isinstance(dataset, (str, os.PathLike)):
        raise TypeError(
            'hypertools.load: dataset must be a string (a dataset name, '
            'file path, or URL), a path-like object, or a list/tuple of '
            f'strings; got {type(dataset).__name__}')
    dataset = os.fspath(dataset)

    if dataset in EXAMPLE_DATA.keys():
        geo_data = _load_example_data(dataset)
        if dataset.endswith('_model'):
            # geo_data is a sklearn.pipeline.Pipeline, not a DataGeometry
            return geo_data
    else:
        # resolution chain, right after built-in names: scikit-learn's
        # small bundled datasets, then seaborn's named datasets (see
        # io.sources; scikit-learn wins over seaborn for names both
        # define, e.g. 'iris'), before falling back to local file ->
        # Hugging Face -> Google Sheets -> Google Drive -> Dropbox ->
        # generic URL.
        from .sources import sklearn_dataset, seaborn_dataset, \
            fivethirtyeight_dataset, kaggle_dataset, SKLEARN_DATASETS
        extra_attempts = [
            'built-in example dataset: not one of '
            f'{sorted(EXAMPLE_DATA)}']
        geo_data = sklearn_dataset(dataset)
        if geo_data is None:
            extra_attempts.append(
                'scikit-learn bundled dataset: not one of '
                f'{sorted(SKLEARN_DATASETS)}')
            geo_data = seaborn_dataset(dataset)
            if geo_data is None:
                extra_attempts.append(
                    'seaborn dataset: not found via '
                    'seaborn.get_dataset_names() (or that lookup failed, '
                    'e.g. no network access)')
                # explicit prefixes -- 'fivethirtyeight/<slug>' and
                # 'kaggle/<owner>/<dataset>' are unambiguous, so a
                # matching-but-failing name raises directly instead of
                # falling through to the attempts digest below
                geo_data = fivethirtyeight_dataset(dataset)
                if geo_data is None:
                    geo_data = kaggle_dataset(dataset)

        if geo_data is None:
            dataset_path = Path(expanduser(expandvars(dataset))).resolve()
            if dataset_path.is_file():
                if legacy:
                    geo_data = _load_legacy(dataset_path)
                else:
                    geo_data = _load_local(dataset_path)
            else:
                from .sources import load_source
                geo_data = load_source(dataset, split=split,
                                       streaming=streaming, trust=trust,
                                       extra_attempts=extra_attempts)

    from .streaming import is_stream
    if is_stream(geo_data):
        if any(v is not None and v is not False
              for v in (reduce, ndims, align, normalize)):
            raise ValueError(
                'reduce/ndims/align/normalize cannot be applied to a '
                'streaming dataset at load time; pass the stream to '
                'hypertools.plot(), which fits models on the first '
                'stream_init samples')
        return geo_data

    if any(v is not None and v is not False
          for v in (reduce, ndims, align, normalize)):
        reduce = reduce or 'IncrementalPCA'
        # shapes-zoo/datasaurus entries are plain arrays/DataFrames/lists
        # rather than DataGeometry objects
        raw = geo_data.get_data() if isinstance(geo_data, DataGeometry) \
            else geo_data
        return analyze(raw,
                       reduce=reduce,
                       ndims=ndims,
                       align=align,
                       normalize=normalize)

    # hypertools 1.0 users never receive a geo: extract the raw data (list
    # of arrays / DataFrame) from any DataGeometry unpickled from a hosted
    # or legacy file. Everything else passes through unchanged -- checked
    # by type, NOT by duck-typed hasattr('get_data'), so unrelated objects
    # that happen to expose a get_data() method (e.g. a pickled matplotlib
    # Line2D) round-trip intact (QC 2026-07, F20-save-004).
    return geo_data.get_data() if isinstance(geo_data, DataGeometry) \
        else geo_data


def _load_local(dataset_path):
    """Load a local file: pickle-format files (by extension or magic byte)
    are unpickled here, keeping the historical DataGeometry handling;
    everything else goes through the extension-based parser
    (npy/npz/csv/tsv/txt/json/parquet/mat/xlsx, gzip variants, and content
    sniffing for extensionless files).
    """
    raw = dataset_path.read_bytes()
    if not raw:
        raise HypertoolsIOError(
            f'{dataset_path} is empty (0 bytes) -- nothing to load. If a '
            'save writing this file failed midway, re-run it.')
    looks_pickled = dataset_path.suffix.lower() in (
        '.geo', '.pkl', '.pickle', '.p') or raw[:1] == b'\x80'
    if looks_pickled:
        try:
            geo_data = pickle.loads(raw)
        except Exception as e:
            # covers pickle.UnpicklingError AND the bare EOFError a
            # truncated/half-written pickle raises (QC 2026-07,
            # F20-save-007)
            raise HypertoolsIOError(
                f'failed to unpickle {dataset_path} '
                f'({type(e).__name__}: {e}). The file may be truncated '
                'or corrupted (e.g. an interrupted download or save). '
                f'If {dataset_path.name} was created with '
                'hypertools<0.8.0, pass legacy=True to load it.'
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
    except Exception as e:
        # Broad except (not just ImportError): `deepdish` is unmaintained and
        # references numpy internals removed in numpy 2 (e.g. np.ComplexWarning),
        # so on this package's required numpy>=2 it fails to IMPORT with an
        # AttributeError, not an ImportError. Either way the user needs the same
        # remedy, so surface one friendly message.
        raise HypertoolsIOError(
            "This looks like a legacy (<1.0) deepdish/HDF5-format dataset. "
            "Reading it needs the `deepdish` package, which is unmaintained and "
            "only works under numpy<2 (incompatible with hypertools' numpy>=2 "
            "requirement). Read the file in a separate environment with "
            "`numpy<2` and `pip install deepdish`, then re-save it in a modern "
            "format (e.g. .npz/.csv)."
        ) from e
    data_dict = dd.io.load(dataset_path)

    if isinstance(data_dict['data'], dict):
        data_dict['data'] = pd.DataFrame(data_dict['data'])
    elif isinstance(data_dict['data'], np.ndarray):
        data_dict['data'] = list(data_dict['data'])
    data_dict['xform_data'] = list(data_dict['xform_data'])
    return DataGeometry(**data_dict)


def _load_sotus_corpus():
    """The 'sotus' example dataset: 29 State of the Union addresses
    (1989-2018), as a list of strings.

    Loaded via datawrangler's text zoo (datawrangler is a core
    dependency, and it caches the download locally) rather than the
    legacy Google Drive registry: the historical Drive id for 'sotus'
    had been duplicated with 'nips_model', so it served a pickled
    topic-model Pipeline instead of the documented speeches, and every
    older 'sotus' Drive id is dead (QC 2026-07, F18-load-hosted-001).
    """
    import contextlib
    import io as _io

    import datawrangler as dw

    # get_corpus prints "loading corpus: sotus...done!" chatter; keep
    # hypertools' own output clean
    with contextlib.redirect_stdout(_io.StringIO()):
        corpus = dw.zoo.text.get_corpus('sotus')
    return [str(doc) for doc in np.asarray(corpus).ravel()]


def _load_example_data(dataset):
    if dataset == 'sotus':
        return _load_sotus_corpus()

    dataset_path = DATA_DIR.joinpath(dataset)
    if not dataset_path.is_file():
        if not DATA_DIR.is_dir():
            if DATA_DIR.exists():
                raise HypertoolsIOError(
                    f'{DATA_DIR} exists but is not a directory, so '
                    'hypertools cannot cache example datasets there. '
                    'Delete or rename it, then retry.')
            try:
                DATA_DIR.mkdir(parents=True)
            except OSError as e:
                raise HypertoolsIOError(
                    f'could not create the example-dataset cache '
                    f'directory {DATA_DIR} ({type(e).__name__}: {e}). '
                    'Check the path and its permissions, then retry.'
                ) from e
        _download_example_data(dataset_path)
    elif not _integrity_ok(dataset_path, dataset):
        # validate every CACHE HIT: a cached file whose hash no longer
        # matches the pin (corruption, or a poisoned/edited cache) is
        # re-downloaded ONCE from the authoritative host, then re-checked
        # below -- it is never deserialized on the strength of a stale,
        # unverified cache (2026-07 release review, blocker #1).
        dataset_path.unlink(missing_ok=True)
        _download_example_data(dataset_path)

    # hard integrity gate: the file is verified against its pinned SHA-256
    # BEFORE any deserialization. A mismatch here is a hard error (the
    # download loop already retried transient/rate-limited responses), never
    # a silent redownload-and-reparse.
    if not _integrity_ok(dataset_path, dataset):
        actual = _sha256_file(dataset_path)
        dataset_path.unlink(missing_ok=True)
        raise HypertoolsIOError(
            f"integrity check failed for the built-in dataset '{dataset}': "
            f"its SHA-256 ({actual[:16]}...) does not match the expected, "
            "pinned value. The download may have been corrupted or tampered "
            "with, or the hosted file may have changed. The unverified file "
            "was removed; re-run to download it again. If this persists, "
            "please report it at "
            "https://github.com/ContextLab/hypertools/issues.")

    geo_data = _unpickle_example(dataset_path)

    if dataset == 'mushrooms':
        # format mushrooms dataset as a pandas DataFrame
        geo_data.data = pd.DataFrame(geo_data.data)
    if dataset.endswith('_model'):
        # hosted pipelines were pickled under an older scikit-learn (QC
        # 2026-07, F18-load-hosted-002); restore the standard estimator
        # surface (repr/get_params/clone) under the installed version
        geo_data = _repair_unpickled_model(geo_data)
    return geo_data


def _repair_unpickled_model(model):
    """Backfill constructor attributes added by scikit-learn versions
    newer than the one a hosted model was pickled under.

    The hosted *_model pipelines were pickled with scikit-learn 1.0.2;
    unpickling them under a newer scikit-learn leaves init attributes
    introduced since then (e.g. ``Pipeline.transform_input``, added in
    1.6) missing, which crashes ``repr()``/``get_params()``/``clone()``
    with AttributeError even though ``transform()`` works (QC 2026-07,
    F18-load-hosted-002). Filling each missing constructor parameter with
    its current default restores the standard estimator surface without
    touching the fitted state, so model outputs are unchanged.
    """
    import inspect

    try:
        from sklearn.base import BaseEstimator
    except ImportError:  # pragma: no cover -- sklearn is a core dependency
        return model

    def _fill_defaults(est):
        if not isinstance(est, BaseEstimator):
            return
        try:
            sig = inspect.signature(type(est).__init__)
        except (TypeError, ValueError):  # pragma: no cover
            return
        for pname, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty:
                continue
            if not hasattr(est, pname):
                setattr(est, pname, param.default)

    _fill_defaults(model)
    for _, step in getattr(model, 'steps', None) or []:
        _fill_defaults(step)
    return model


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
        except HypertoolsIOError as e:
            last_error = e
            continue
        # a download only counts as SUCCESS when its bytes match the pinned
        # checksum: Google Drive serves rate-limit/error HTML with a 200
        # status, which would otherwise be cached as the "dataset". Retry
        # those exactly like a transport failure (2026-07 release review).
        if _integrity_ok(dataset_path, dataset_path.name):
            return
        last_error = HypertoolsIOError(
            f"the downloaded '{dataset_path.name}' did not match its "
            "expected checksum -- often a transient rate-limit response "
            "served in place of the file; retrying")
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
            # Google Drive serves a "can't scan this file for viruses"
            # HTML interstitial (not a cookie) for large files; peek at
            # the Content-Type header (available before the streamed body
            # is read) and follow its confirm form when present.
            if 'html' in response.headers.get('Content-Type', ''):
                from .sources import parse_drive_interstitial
                html = response.content.decode('utf-8', errors='replace')
                parsed = parse_drive_interstitial(html)
                if parsed is not None:
                    action_url, form_params = parsed
                    response = session.get(action_url, params=form_params,
                                           stream=True)

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
    shims), and dill-serialized arrays (the shapes-zoo datasets).

    scikit-learn's InconsistentVersionWarning is suppressed here: the
    hosted files are hypertools' own artifacts (pickled under an older
    scikit-learn), the version skew is known, and the estimator surface is
    repaired after unpickling (see _repair_unpickled_model), so the
    warning is pure noise for users (QC 2026-07, F18-load-hosted-002)."""
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except ImportError:  # pragma: no cover -- sklearn is a core dependency
        InconsistentVersionWarning = None

    raw = dataset_path.read_bytes()
    with warnings.catch_warnings():
        if InconsistentVersionWarning is not None:
            warnings.simplefilter('ignore', InconsistentVersionWarning)
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
