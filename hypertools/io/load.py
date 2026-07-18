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

# Built-in datasets. The DATA datasets are hosted in non-executable formats
# (.npz / .parquet / .json.gz) on Dropbox and loaded WITHOUT unpickling
# (2026-07 release review, blocker #1). The old Google-Drive pickle ids for
# them stay live so hypertools <1.0 keeps loading; 1.0 uses these URLs. The
# fitted sklearn *_model Pipelines are inherently pickle, still hosted on
# Drive and hash-verified before load (skops re-hosting is a follow-up).
# 'sotus' loads via datawrangler's text zoo (see _load_sotus_corpus).
EXAMPLE_DATA = {
    'weights': 'https://www.dropbox.com/scl/fi/9byhw72c36grhf2toj3to/weights.npz?rlkey=tbr21gqpflljjttl15nuwij8z&dl=1',
    'weights_avg': 'https://www.dropbox.com/scl/fi/qaj00kxx4c6j3309pryll/weights_avg.npz?rlkey=y8nx2n1j6fx7e8zrz8ko80mrs&dl=1',
    'weights_sample': 'https://www.dropbox.com/scl/fi/96u1kjkvcb8449hxh6b0p/weights_sample.npz?rlkey=4f6vl26re8qpm626wtrfud6ox&dl=1',
    'spiral': 'https://www.dropbox.com/scl/fi/mcjpsfkihjered7kyfcg4/spiral.npz?rlkey=4x0va41duh4txjr9y8jq4963c&dl=1',
    'mushrooms': 'https://www.dropbox.com/scl/fi/4sz0zv1pypoko9nh9adh9/mushrooms.parquet?rlkey=1ndugdy10c4iznze65rvwie3j&dl=1',
    'wiki': 'https://www.dropbox.com/scl/fi/50genuwwocsad93ciwyrj/wiki.json.gz?rlkey=kq921wznkx6iiuelvq2yk2mlp&dl=1',
    'nips': 'https://www.dropbox.com/scl/fi/feszsl8vl6fn5u4iok1at/nips.json.gz?rlkey=zqcsarcqb336h7j23b9mwgbf3&dl=1',
    'bunny': 'https://www.dropbox.com/scl/fi/teteaybwrtiqd671p1dlp/bunny.npz?rlkey=cvg855t17z1vq1e1ypy1kxzum&dl=1',
    'cube': 'https://www.dropbox.com/scl/fi/hg00kbk331h64wlgnub04/cube.npz?rlkey=elo0ms55n8uz0e4qq5sclku1y&dl=1',
    'dragon': 'https://www.dropbox.com/scl/fi/u9dc5oiirsb0vxuwwpwxw/dragon.npz?rlkey=oex83erlrdcskwhh6v8obiv38&dl=1',
    'sphere': 'https://www.dropbox.com/scl/fi/cjvehkj3js7humoa0r1pa/sphere.npz?rlkey=n1ncxsx5550b8cormhe46rn3v&dl=1',
    'teapot': 'https://www.dropbox.com/scl/fi/qgz9j9696lqwtea4lzkb7/teapot.npz?rlkey=vdee69b4wd6l9499kq8on02vl&dl=1',
    'vase': 'https://www.dropbox.com/scl/fi/9m207u7ta0gu04hxbcnjr/vase.npz?rlkey=bitxamldgkyg2ybkaka6qdpo2&dl=1',
    'biplane': 'https://www.dropbox.com/scl/fi/s2f4g8652dm5xdc313ogm/biplane.parquet?rlkey=x7jn1al8i92my6llkexilsi58&dl=1',
    'datasaurus': 'https://www.dropbox.com/scl/fi/5hk73y5qehe2o31eflvfd/datasaurus.npz?rlkey=pj8y6so417g6t4nbpihx0s0tl&dl=1',
    'sotus': 'datawrangler-zoo:sotus',
    # fitted sklearn Pipelines (pickle; hash-verified before load)
    'wiki_model': '1T-UAU-6KVGUBcUWqz7yG59vXnThu9T0H',
    'nips_model': '1J0MBhpRwdT2WChfWJ4HXYq6jU4XpyJPm',
    'sotus_model': '16_n9r82pwxzZh-0qdS4a6l0z3v__Q91C',
}

# How each non-executable built-in reconstructs to the exact value hyp.load
# returned from its former pickle (verified equal, incl. dtype/columns/index):
#   npz_list      -> list of arrays (arr_0..arr_{n-1}, in order)
#   npz_array     -> a single array (arr_0)
#   npz_df_xy     -> list of DataFrames with columns ['x', 'y'] and each
#                    frame's original integer index restored (datasaurus; see
#                    _DATASAURUS_INDEX_STARTS)
#   parquet       -> DataFrame (columns + index preserved by parquet)
#   jsongz_text   -> [ (n, 1) object array of document strings ] (text corpus)
_REHOSTED = {
    'weights': 'npz_list', 'weights_avg': 'npz_list',
    'weights_sample': 'npz_list', 'spiral': 'npz_list',
    'datasaurus': 'npz_df_xy',
    'bunny': 'npz_array', 'cube': 'npz_array', 'dragon': 'npz_array',
    'sphere': 'npz_array', 'teapot': 'npz_array', 'vase': 'npz_array',
    'mushrooms': 'parquet', 'biplane': 'parquet',
    'wiki': 'jsongz_text', 'nips': 'jsongz_text',
}

# The "Datasaurus Dozen" is 13 shuffled 142-row blocks of one 1846-row table;
# `hyp.load('datasaurus')` returns them as 13 DataFrames, and each frame's
# ORIGINAL pandas index is the contiguous global row range of its block (e.g.
# frame 0 spans rows 142-283). `hyp.load` returns raw data, so those indexes
# are part of the public result -- 1.0 preserves them exactly (2026-07 release
# review, finding #3). The hosted .npz stores only the x/y values (verified
# bit-identical to the pre-1.0 pickle); these per-frame start offsets -- an
# immutable compatibility constant, in frame order -- restore the indexes on
# load. A regression fixture (tests/data/rehosted_compat_baseline.json) checks
# the fully-reconstructed result against the pre-1.0 original.
_DATASAURUS_INDEX_STARTS = (142, 1278, 1136, 0, 994, 284, 852, 1562, 1420,
                            710, 426, 1704, 568)

# SHA-256 of each hosted built-in file, pinned so a built-in is verified
# against a hard-coded cryptographic hash BEFORE it is read (2026-07 release
# review, blocker #1). A mismatch (corrupt/rate-limited download, poisoned
# cache, or a tampered/changed upstream file) is a HARD error -- never a
# silent redownload-and-reparse. Every cache hit is validated too. The DATA
# hashes are of the non-executable Dropbox files; the *_model hashes are of
# the Drive pickle files (still hash-verified before unpickling).
_EXAMPLE_DATA_SHA256 = {
    'weights': 'ab24402f6d998eea0550044f264d79593cd1adc97903d54322118119dbe8ed55',
    'weights_avg': 'f8b38023867157f44fc7b22723f0e539eb7519d52cb1eff7be8b71f73f71b9f7',
    'weights_sample': 'e5876ba8599b6819bfbf2e44dc58d9cfa0a9d1afbb8c6ad92687604bf8393933',
    'spiral': '5c713739be0843c407cceebe659a0838fe18cf69f84291896bccb0c282f9d622',
    'mushrooms': 'e9edd15fa603ba8ea49ca1726ccc889a1378b68286467b1e91a1ff0f0dc49de2',
    'wiki': '9c1cfcb4552841d1f192de1a2cd4eeb33bbd3d0697c408f50a559666561d220a',
    'nips': '10409cd39c62eea8325d98afdcf09b2a84e119e6ff4f977cb8e52eb144b624c4',
    'bunny': '8c010711a9f7ca779c7a9b804f21028d15c099271a1c9c54ed221c10f8754f22',
    'cube': 'fd87423f357a0bf909c74c56da1d6cd2a50a373d777dba442ab609aad0bb33bb',
    'dragon': 'be1ae5262850539d9bfe37ec946a9a0e0fd745e58c5b69fa3b39284beca8cfa8',
    'sphere': '503f1820e20f2e1daf0e9dc63bf5035ed5ad0b9299b1b7dbc560eb11e585b57a',
    'teapot': '804e851abee7037d7bb9f135d5c3344b790af64aa4b04f96e789eb8473e4b284',
    'vase': 'b3ee4ffd68b0b1e4e05ca54ef1193507c625915f5e8f7c6c50f576669bb80bc7',
    'biplane': '8ffb74e24af0b84e20c151c6f0601fd677cb1f1f029e4d86eb7523c3a9a4268b',
    'datasaurus': '8e8c2e1bc4ac33402f9448ab78860e7d79bef97ee5b413e4b2081b8b4d3f5f52',
    'wiki_model': '5ec3c34e2524e105a90ae498cca809d61ddfa90813a4621de65b37275fd515c9',
    'nips_model': '4f93308a48002730866659bda7ef393f5451dc8360b9e3c91c9cf5d77f73a762',
    'sotus_model': 'a7b085f7f6d94dbed6d961a1950de18a07b56456c77c2495a2868a9fefb07aa4',
}


def _parse_rehosted(path, name):
    """Reconstruct a re-hosted (non-executable) built-in dataset from its
    cached .npz/.parquet/.json.gz file -- no unpickling. Returns exactly what
    hyp.load(name) returned from the former pickle (see _REHOSTED)."""
    fmt = _REHOSTED[name]
    if fmt == 'parquet':
        return pd.read_parquet(path)
    if fmt == 'jsongz_text':
        import gzip
        import json
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            docs = json.load(f)
        return [np.array(docs, dtype=object).reshape(-1, 1)]
    # npz variants (content-sniffed by np.load regardless of the cache
    # filename having no extension); allow_pickle=False -> no code execution
    with np.load(path, allow_pickle=False) as z:
        arrs = [z[f'arr_{i}'] for i in range(len(z.files))]
    if fmt == 'npz_array':
        return arrs[0]
    if fmt == 'npz_df_xy':
        # restore each frame's original global-row-range index (finding #3)
        assert len(arrs) == len(_DATASAURUS_INDEX_STARTS), (
            f'{name}: {len(arrs)} frames but '
            f'{len(_DATASAURUS_INDEX_STARTS)} pinned index offsets')
        return [pd.DataFrame(a, columns=['x', 'y'],
                             index=pd.Index(np.arange(s, s + len(a))))
                for a, s in zip(arrs, _DATASAURUS_INDEX_STARTS)]
    return arrs  # npz_list


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

    # the DATA datasets are non-executable (.npz/.parquet/.json.gz) -> read
    # them without ever unpickling (2026-07 release review, blocker #1)
    if dataset in _REHOSTED:
        return _parse_rehosted(dataset_path, dataset)

    # remaining built-ins are the fitted sklearn *_model Pipelines (pickle)
    geo_data = _unpickle_example(dataset_path)
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


def _dataset_download_lock(dataset_path):
    """A best-effort, per-dataset advisory lock so that concurrent processes
    do not each re-download the same (potentially large) file.

    Correctness NEVER depends on this lock -- the atomic temp-file +
    ``os.replace`` in :func:`_download_example_data` is what guarantees a
    reader never sees a partial or unverified file. The lock only avoids
    wasted duplicate downloads, so if ``filelock`` is unavailable (it is a
    transitive dependency of some extras, not the base install) or the lock
    cannot be acquired promptly, we simply proceed without it.
    """
    import contextlib

    try:
        from filelock import FileLock, Timeout
    except Exception:
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _guarded():
        lock = FileLock(str(dataset_path) + '.lock', timeout=600)
        acquired = False
        try:
            try:
                lock.acquire()
                acquired = True
            except Timeout:
                pass  # heavy contention -> proceed; os.replace stays atomic
            yield
        finally:
            if acquired:
                lock.release()

    return _guarded()


def _download_example_data(dataset_path, max_attempts=4):
    """Download an example dataset ATOMICALLY and hash-verify it before it is
    ever visible at its final cache path.

    Each attempt streams into a private temp file in the SAME directory,
    verifies its SHA-256, and only then ``os.replace``s it into place (an
    atomic rename on one filesystem). Consequences: a concurrent reader
    never observes a partial or unverified cache entry; an interrupted or
    killed download leaves at most an orphan ``.part`` temp file, never a
    corrupt cache file; and two processes racing to fetch the same dataset
    cannot delete or truncate each other's in-progress writes. A best-effort
    per-dataset lock (:func:`_dataset_download_lock`) additionally avoids
    duplicate large downloads.

    Retries with backoff when the host rate-limits (Google Drive answers
    rate-limited requests with an HTML error page and a 200 status).
    """
    import tempfile
    import time

    name = dataset_path.name
    parent = dataset_path.parent

    with _dataset_download_lock(dataset_path):
        # another process may have finished the download while we waited on
        # the lock -- don't redo it
        if dataset_path.is_file() and _integrity_ok(dataset_path, name):
            return

        last_error = None
        for attempt in range(max_attempts):
            if attempt > 0:
                # 2s, 6s, 18s -- long enough for transient Drive rate limits
                time.sleep(2 * 3 ** (attempt - 1))

            fd, tmp_name = tempfile.mkstemp(dir=parent, prefix=f'.{name}.',
                                            suffix='.part')
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                _download_example_data_once(tmp_path, name)
            except HypertoolsIOError as e:
                last_error = e
                tmp_path.unlink(missing_ok=True)
                continue

            # a download only counts as SUCCESS when its bytes match the
            # pinned checksum: Google Drive serves rate-limit/error HTML with
            # a 200 status, which would otherwise be cached as the "dataset".
            # Retry those exactly like a transport failure.
            if _integrity_ok(tmp_path, name):
                # atomic publish: a reader sees either the previous state or
                # the whole verified file, never a half-written one
                os.replace(tmp_path, dataset_path)
                return

            tmp_path.unlink(missing_ok=True)
            last_error = HypertoolsIOError(
                f"the downloaded '{name}' did not match its expected "
                "checksum -- often a transient rate-limit response served "
                "in place of the file; retrying")
        raise last_error


def _download_example_data_once(dest_path, name):
    """Fetch ``EXAMPLE_DATA[name]`` into ``dest_path`` (a private temp file;
    never the final cache path). Raises HypertoolsIOError on a transport
    failure or an obvious error page."""
    source = EXAMPLE_DATA[name]
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
        with dest_path.open('wb') as f:
            # write stream in chunks to avoid loading whole file into memory
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

        # Google Drive answers rate-limited/oversized requests with an HTML
        # page and a 200 status; caching it would poison every later load.
        # Every hypertools example file is binary (.npz/.parquet/.json.gz or
        # a pickle) and never starts with '<'.
        with dest_path.open('rb') as f:
            if f.read(1) == b'<':
                dest_path.unlink(missing_ok=True)
                raise HypertoolsIOError(
                    f"Download of '{name}' returned an error page instead "
                    "of the dataset (the host may be rate-limiting "
                    "requests). Please try again later."
                )
    except HypertoolsIOError:
        raise
    except Exception as e:
        # clean up partial file in case of error while writing stream
        dest_path.unlink(missing_ok=True)
        raise HypertoolsIOError(
            f"Failed to download '{name}' dataset"
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
