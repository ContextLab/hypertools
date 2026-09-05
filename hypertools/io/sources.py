#!/usr/bin/env python
"""Universal data-source resolution for hyp.load / DataGeometry.

A string may name (tried in this order):

1. a built-in example dataset (``EXAMPLE_DATA`` in ``hypertools.io.load``)
2. a scikit-learn bundled dataset name (``sklearn_dataset``)
3. a seaborn dataset name (``seaborn_dataset``)
4. a fivethirtyeight/data dataset, explicit prefix ``'fivethirtyeight/<slug>'``
   (``fivethirtyeight_dataset``)
5. a Kaggle dataset, explicit prefix ``'kaggle/<owner>/<dataset>'``
   (``kaggle_dataset``)
6. a built-in SYNTHETIC dataset name -- ``'random_walk'``, ``'helix'``,
   ``'lorenz'``, ``'blobs'``, ``'moons'``, ``'swiss_roll'``, ``'s_curve'``
   (``synthetic_dataset``; generated on the spot, never fetched)
7. a web source, explicit prefix ``'wikipedia:'``, ``'yahoo:'`` or
   ``'sec:'`` (``web_source``)
8. a path to a local file
9. a Hugging Face dataset (including streaming datasets)
10. a Google Sheets URL
11. a Google Drive URL or bare file ID
12. a Dropbox URL or shared-link path
13. any other URL (with or without an ``https://`` scheme)

Steps 4, 5 and 7 are explicit, unambiguous prefixes: unlike the scikit-learn/
seaborn/synthetic name lookups (which silently fall through to the next
resolver when they don't recognize a name), a string that starts with
``'fivethirtyeight/'``, ``'kaggle/'``, ``'wikipedia:'``, ``'yahoo:'`` or
``'sec:'`` unambiguously names that source, so a failure there raises
immediately instead of falling through the rest of the chain.

Any URL download (steps 10-13) can be cached on disk with ``cache=True``
and replayed from that cache without touching the network with
``offline=True`` (see :func:`url_cache_dir`).

Lists of strings resolve element-wise to a list of datasets.

.. warning::
    Pickle-format payloads (``.pkl``/``.geo``/pickled arrays) can execute
    arbitrary code when loaded, exactly like ``pandas.read_pickle``. Only
    load pickled data from sources you trust. Non-pickle formats
    (csv/tsv/json/npy without object arrays/parquet/mat) do not have this
    risk.
"""

import io
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..core.exceptions import HypertoolsIOError

_DRIVE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{25,}$')
_DOMAIN_RE = re.compile(r'^[\w-]+(\.[\w-]+)+(/\S*)?$')
_HOST_PORT_RE = re.compile(r'^[\w.-]+:\d+(/\S*)?$')
_HF_ID_RE = re.compile(r'^[\w.-]+/[\w.-]+$')
_UA = {'User-Agent': 'hypertools'}

# every filename extension load_local_file/_parse_payload knows how to
# parse; anything else is rejected (QC 2026-07, X2-error-quality-001)
# unless the CONTENT unambiguously matches a known binary format
_SUPPORTED_EXTENSIONS = (
    '.pkl', '.pickle', '.p', '.geo', '.npy', '.npz', '.csv', '.tsv',
    '.txt', '.json', '.parquet', '.mat', '.xlsx', '.xls', '.gz')


def _github_api_headers():
    """Headers for a github.com API request, authenticated when possible.

    GitHub's REST API allows only 60 requests/hour for unauthenticated
    clients but 5000/hour for authenticated ones. If a token is available
    in the environment (``GITHUB_TOKEN`` -- provided automatically inside
    GitHub Actions -- or ``GH_TOKEN``), it is sent as a bearer token so the
    listing calls don't exhaust the shared unauthenticated quota (e.g. when
    many CI jobs run concurrently from the same runner IP pool). Without a
    token the request is simply unauthenticated, exactly as before.

    Returns
    -------
    dict
        Request headers (always includes the ``hypertools`` User-Agent;
        includes an ``Authorization`` bearer header when a token is found).
    """
    headers = dict(_UA)
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return headers


def _env_token_hint(status_code):
    """Extra error-message sentence when an ambient GITHUB_TOKEN/GH_TOKEN
    was sent and GitHub answered 401: the stale/invalid env token -- not
    hypertools or the dataset -- is the likely cause, and unsetting it
    (anonymous access) would work (QC 2026-07, F19-load-external-011)."""
    if status_code != 401:
        return ''
    if not (os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')):
        return ''
    return (
        ' A GITHUB_TOKEN/GH_TOKEN environment variable was found and sent '
        'with the request, and GitHub rejected it -- the token is likely '
        'expired or invalid. Unset it (anonymous access works for these '
        'requests) or refresh it.')


# GitHub's REST API and raw host intermittently return transient gateway
# errors under load -- e.g. a CI matrix's shared runner-IP pool hammering the
# API concurrently regularly sees 502 Bad Gateway. These are server-side, not
# a client/code error, and almost always clear within a second or two.
_TRANSIENT_STATUS = frozenset({502, 503, 504})


def _github_get_with_retry(url, headers, timeout, attempts=4, backoff=1.0):
    """``requests.get`` that retries transient failures with exponential
    backoff.

    Retries on a transient HTTP status (502/503/504 gateway errors) or a
    connection-level ``requests.RequestException`` (dropped connection, DNS
    blip, read timeout), sleeping ``backoff``, ``2*backoff``, ``4*backoff``,
    ... between tries. Any non-transient response (2xx, 404, 403 rate-limit,
    ...) is returned immediately on the first try, so healthy calls pay no
    penalty and every caller's own status handling is unchanged. After the
    final attempt the last response is returned (letting the caller's status
    handling run) or, for a persistent connection error, the last exception
    is re-raised.

    ``attempts`` and ``backoff`` are tunable so tests can exercise the retry
    loop quickly against a real local server; production callers use the
    defaults (four tries over ~7s).
    """
    delay = backoff
    last_exc = None
    for attempt in range(attempts):
        is_last = attempt == attempts - 1
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last_exc = e
            if is_last:
                raise
        else:
            if resp.status_code not in _TRANSIENT_STATUS or is_last:
                return resp
        time.sleep(delay)
        delay *= 2
    raise last_exc  # unreachable: the loop always returns or raises

# Google Sheets URL -> CSV export (must be checked before generic Drive id
# extraction, since a Sheets URL also matches the '/d/<id>' Drive pattern).
_SHEETS_ID_RE = re.compile(
    r'docs\.google\.com/spreadsheets/d/([A-Za-z0-9_-]+)')

# Google Drive's large-file "can't scan for viruses" interstitial: an HTML
# page (200 status) with a confirm form instead of the file. The real
# download lives at the form's action URL with its hidden inputs as params.
_DRIVE_INTERSTITIAL_FORM_RE = re.compile(
    r'<form[^>]*action="(https://drive\.usercontent\.google\.com/download)"'
    r'[^>]*>(.*?)</form>', re.DOTALL)
_HIDDEN_INPUT_RE = re.compile(
    r'<input\s+type="hidden"\s+name="([^"]+)"\s+value="([^"]*)"')

_PICKLE_TRUST_REFUSAL = (
    'refusing to unpickle data fetched from a remote source: unpickling '
    'executes arbitrary code embedded in the payload, so hypertools does '
    'NOT do it for remote data by default. If you downloaded this from a '
    'source you trust and have verified it, pass trust=True to '
    'hypertools.load() to allow it. When you control the data, prefer a '
    'non-executable format (.npz / .csv / .parquet), which never needs '
    'trust=True.')


class HypertoolsTrustError(ValueError):
    """Raised when a remote payload can only be loaded by relaxing a
    security policy (currently: ``allow_pickle`` for remote .npy/.npz
    object arrays) and ``trust=True`` was not passed.

    Subclasses ``ValueError`` so any existing code that catches
    ``ValueError`` still catches this. ``load_source()``'s per-branch
    guards key on this specific type (not bare ``ValueError``) so that
    unrelated ``ValueError`` subclasses -- e.g. pandas' ``ParserError``,
    ``EmptyDataError``, or ``json.JSONDecodeError`` from a genuinely
    malformed payload -- fall through to the "Tried, in order" digest
    instead of escaping raw.
    """


class HypertoolsOfflineError(HypertoolsIOError):
    """Raised when ``offline=True`` was passed and the URL has no cached
    copy to read (GH #285).

    Subclasses :class:`~hypertools.core.exceptions.HypertoolsIOError`, so
    existing handlers still catch it; ``load_source`` keys on this
    specific type to let it ESCAPE the per-branch "tried, in order"
    digest -- an offline miss is a deliberate, self-explanatory refusal
    (the message names the cache path to populate), not one failed guess
    among several.
    """


SKLEARN_DATASETS = {
    'iris': 'load_iris',
    'digits': 'load_digits',
    'wine': 'load_wine',
    'breast_cancer': 'load_breast_cancer',
    'diabetes': 'load_diabetes',
    'linnerud': 'load_linnerud',
}

# seaborn.get_dataset_names() is a network call (fetches the seaborn-data
# GitHub repo's file listing); cache it per-process so repeated hyp.load()
# calls don't re-hit the network for every unresolved name.
_seaborn_names_cache = None

# fivethirtyeight/data folder listings, keyed by slug (e.g. 'bechdel'):
# GitHub's unauthenticated REST API rate limit is 60 requests/hour, so the
# per-slug CSV-filename listing is cached per-process. Value is a (possibly
# empty) list of CSV filenames.
_538_listing_cache = {}
_538_API = 'https://api.github.com/repos/fivethirtyeight/data/contents'
_538_RAW = 'https://raw.githubusercontent.com/fivethirtyeight/data/master'


def sklearn_dataset(name):
    """Load one of scikit-learn's small, bundled example datasets by name.

    Only the ``sklearn.datasets.load_*`` loaders that ship their data with
    the package are considered (``'iris'``, ``'digits'``, ``'wine'``,
    ``'breast_cancer'``, ``'diabetes'``, ``'linnerud'``); the network-fetched
    ``fetch_*`` datasets are intentionally excluded so this never triggers a
    download.

    Returns
    -------
    data : pandas.DataFrame or None
        The dataset's features as columns, with the target appended as a
        ``'target'`` column (or, for multi-output targets such as
        ``'linnerud'``, one column per target name). None if ``name`` isn't
        one of the bundled dataset names above -- callers should treat this
        as "not mine" and continue down the resolution chain.
    """
    fn_name = SKLEARN_DATASETS.get(name)
    if fn_name is None:
        return None
    from sklearn import datasets as sk_datasets
    bunch = getattr(sk_datasets, fn_name)(as_frame=True)
    df = bunch.data.copy()
    target = bunch.target
    if isinstance(target, pd.DataFrame):
        # multi-output datasets (e.g. linnerud): one column per target name
        for col in target.columns:
            df[col] = target[col]
    else:
        df[getattr(target, 'name', None) or 'target'] = target
    return df


def seaborn_dataset(name):
    """Load a seaborn example dataset by name.

    ``name`` is matched against ``seaborn.get_dataset_names()`` (a network
    call to the seaborn-data GitHub repo; the result is cached per-process).

    Returns
    -------
    data : pandas.DataFrame or None
        The result of ``seaborn.load_dataset(name)``, unchanged. None if
        ``name`` isn't a known seaborn dataset, or if the dataset name list
        can't be fetched (network failure) -- either way, callers should
        treat this as "not mine" and continue down the resolution chain.
    """
    global _seaborn_names_cache
    import seaborn as sns
    if _seaborn_names_cache is None:
        try:
            _seaborn_names_cache = set(sns.get_dataset_names())
        except Exception:
            return None
    if name not in _seaborn_names_cache:
        return None
    return sns.load_dataset(name)


def fivethirtyeight_dataset(name):
    """Load a dataset from FiveThirtyEight's public data repository
    (https://github.com/fivethirtyeight/data) by explicit prefix.

    ``name`` must look like ``'fivethirtyeight/<slug>'``, where ``<slug>``
    is the dataset's top-level folder in that repo (e.g.
    ``'fivethirtyeight/bechdel'``, whose folder is
    https://github.com/fivethirtyeight/data/tree/master/bechdel). The
    folder is listed via the GitHub contents API (cached per-process, since
    the unauthenticated API is rate-limited to 60 requests/hour) and every
    CSV file it contains is downloaded from raw.githubusercontent.com.

    This is an explicit, unambiguous prefix -- unlike :func:`sklearn_dataset`
    and :func:`seaborn_dataset`, an unrecognized slug or a folder with no
    CSV files raises :class:`~hypertools.core.exceptions.HypertoolsIOError`
    directly rather than returning None to fall through to the next
    resolver, since the user has unambiguously asked for a 538 dataset.

    Returns
    -------
    data : pandas.DataFrame, dict of {str: pandas.DataFrame}, or None
        A single DataFrame when the folder contains exactly one CSV file;
        a dict mapping each CSV's filename (without extension) to its
        DataFrame when it contains more than one. None if ``name`` doesn't
        start with the ``'fivethirtyeight/'`` prefix -- callers should
        treat this (and only this) case as "not mine".
    """
    if not isinstance(name, str) or not name.startswith('fivethirtyeight/'):
        return None
    slug = name[len('fivethirtyeight/'):].strip('/')
    if not slug:
        raise HypertoolsIOError(
            f"{name!r} is missing a dataset slug -- expected "
            "'fivethirtyeight/<slug>', e.g. 'fivethirtyeight/bechdel'. "
            "Browse available slugs at "
            "https://github.com/fivethirtyeight/data.")

    if slug in _538_listing_cache:
        csv_names = _538_listing_cache[slug]
    else:
        try:
            resp = _github_get_with_retry(f'{_538_API}/{slug}',
                                          _github_api_headers(),
                                          timeout=30)
        except requests.RequestException as e:
            raise HypertoolsIOError(
                f"could not reach the GitHub API to list "
                f"fivethirtyeight/data/{slug}: {type(e).__name__}: {e}"
            ) from e
        if resp.status_code == 404:
            csv_names = []
        elif resp.status_code >= 400:
            # Wrapped in HypertoolsIOError (not left as a raw requests
            # HTTPError) so callers get a consistent exception type and,
            # for the common unauthenticated-rate-limit case, an
            # actionable message. The 403/rate-limited branch can't be
            # exercised in tests on demand without burning GitHub's real
            # 60-requests/hour unauthenticated quota -- reviewed by
            # inspection instead of a live test; the 404 path above (and
            # its HypertoolsIOError) is covered by
            # test_load_538_bad_slug_raises_immediately.
            message = (
                f"GitHub API request for fivethirtyeight/data/{slug} "
                f"failed with HTTP {resp.status_code}"
                f"{' ' + resp.reason if resp.reason else ''}.")
            if resp.status_code == 403 and \
                    resp.headers.get('X-RateLimit-Remaining') == '0':
                message += (
                    " This looks like GitHub's unauthenticated API rate "
                    "limit (60 requests/hour) has been exhausted -- wait "
                    "for it to reset, or authenticate your requests to "
                    "raise the limit.")
            message += _env_token_hint(resp.status_code)
            raise HypertoolsIOError(message)
        else:
            entries = resp.json()
            csv_names = sorted(
                e['name'] for e in entries
                if e.get('type') == 'file'
                and e['name'].lower().endswith('.csv'))
        _538_listing_cache[slug] = csv_names

    if not csv_names:
        raise HypertoolsIOError(
            f"{name!r} does not look like a fivethirtyeight/data dataset "
            f"-- no CSV files found in "
            f"https://github.com/fivethirtyeight/data/tree/master/{slug} "
            "(the slug may not exist, or the folder may hold no CSVs). "
            "Browse available datasets at "
            "https://github.com/fivethirtyeight/data.")

    frames = {}
    for csv_name in csv_names:
        raw = _fetch_538_csv(slug, csv_name)
        frames[Path(csv_name).stem] = pd.read_csv(io.BytesIO(raw))
    if len(frames) == 1:
        return next(iter(frames.values()))
    return frames


def _fetch_538_csv(slug, csv_name):
    """Download one fivethirtyeight CSV, authenticated when possible.

    When a GitHub token is available (``GITHUB_TOKEN``/``GH_TOKEN``), the
    file is fetched through the authenticated GitHub *contents* API
    (``Accept: application/vnd.github.raw``), which counts against the
    5000-requests/hour authenticated quota. Without a token it falls back
    to ``raw.githubusercontent.com`` -- convenient for interactive use, but
    subject to an anonymous per-IP rate limit that many concurrent clients
    (e.g. a CI matrix sharing a runner IP pool) can exhaust with HTTP 429.

    Parameters
    ----------
    slug : str
        The dataset folder in the fivethirtyeight/data repo.
    csv_name : str
        The CSV file name within that folder.

    Returns
    -------
    bytes
        The raw CSV bytes.
    """
    headers = _github_api_headers()
    if 'Authorization' in headers:
        url = (f'https://api.github.com/repos/fivethirtyeight/data/'
               f'contents/{slug}/{csv_name}?ref=master')
        headers = {**headers, 'Accept': 'application/vnd.github.raw'}
        resp = _github_get_with_retry(url, headers, timeout=60)
        if resp.status_code == 401:
            raise HypertoolsIOError(
                f'GitHub API request for fivethirtyeight/data/{slug}/'
                f'{csv_name} failed with HTTP 401 Unauthorized.'
                + _env_token_hint(resp.status_code))
        resp.raise_for_status()
        return resp.content
    # the anonymous raw fetch goes through the same transient-5xx retry
    # helper as the listing call (QC 2026-07, F19-load-external-003: a
    # single transient 502 here used to fail the whole load for every
    # user without a token)
    url = f'{_538_RAW}/{slug}/{csv_name}'
    resp = _github_get_with_retry(url, dict(_UA), timeout=60)
    resp.raise_for_status()
    if not resp.content:
        raise HypertoolsIOError(f'empty response from {url}')
    return resp.content


def kaggle_dataset(name):
    """Load a dataset from Kaggle (https://www.kaggle.com/datasets) by
    explicit prefix, using ``kagglehub.dataset_download`` (anonymous
    downloads work for public datasets).

    ``name`` must look like ``'kaggle/<owner>/<dataset>'``, matching the
    ``<owner>/<dataset>`` id in the dataset's Kaggle URL (e.g.
    ``'kaggle/uciml/iris'`` for https://www.kaggle.com/datasets/uciml/iris).
    Every CSV/TSV file in the downloaded dataset is loaded; other files
    (e.g. a bundled sqlite database) are ignored.

    This is an explicit, unambiguous prefix -- like
    :func:`fivethirtyeight_dataset`, a malformed id or a dataset with no
    tabular files raises
    :class:`~hypertools.core.exceptions.HypertoolsIOError` directly
    rather than returning None to fall through to the next resolver.

    Returns
    -------
    data : pandas.DataFrame, dict of {str: pandas.DataFrame}, or None
        A single DataFrame when the dataset contains exactly one CSV/TSV
        file; a dict mapping each file's name (without extension) to its
        DataFrame when it contains more than one. None if ``name`` doesn't
        start with the ``'kaggle/'`` prefix -- callers should treat this
        (and only this) case as "not mine".
    """
    if not isinstance(name, str) or not name.startswith('kaggle/'):
        return None
    rest = name[len('kaggle/'):]
    parts = [p for p in rest.split('/') if p != '']
    if len(parts) != 2:
        raise HypertoolsIOError(
            f"{name!r} does not look like a valid Kaggle dataset id -- "
            "expected 'kaggle/<owner>/<dataset>', e.g. 'kaggle/uciml/iris' "
            "(from https://www.kaggle.com/datasets/uciml/iris).")
    owner, dataset_slug = parts

    from .._shared.lazy_import import lazy_import
    kagglehub = lazy_import('kagglehub', purpose=f'loading the Kaggle dataset {name!r}')

    try:
        download_path = Path(
            kagglehub.dataset_download(f'{owner}/{dataset_slug}'))
    except Exception as e:
        raise HypertoolsIOError(
            f"could not download Kaggle dataset '{owner}/{dataset_slug}' "
            f"via kagglehub: {type(e).__name__}: {e}") from e

    table_files = sorted(
        p for p in download_path.rglob('*')
        if p.is_file() and p.suffix.lower() in ('.csv', '.tsv'))
    if not table_files:
        raise HypertoolsIOError(
            f"Kaggle dataset '{owner}/{dataset_slug}' was downloaded to "
            f"{download_path}, but it doesn't contain any .csv/.tsv "
            "files.")

    keys = _table_file_keys(download_path, table_files)
    frames = {}
    for f in table_files:
        sep = '\t' if f.suffix.lower() == '.tsv' else ','
        frames[keys[f]] = pd.read_csv(f, sep=sep)
    if len(frames) == 1:
        return next(iter(frames.values()))
    return frames


def _table_file_keys(root, files):
    """Build the dict keys :func:`kaggle_dataset` uses for a list of
    downloaded CSV/TSV files under ``root``.

    Keying by bare filename stem (``f.stem``) is nicer UX for the common
    case (a flat download with unique filenames), but silently overwrites
    entries when two files share a stem in different subdirectories (e.g.
    ``train/data.csv`` and ``test/data.csv``). To avoid that collision:
    use the plain stem when stems are unique across ``files``; otherwise
    fall back to each file's path relative to ``root``, with '/'
    separators and no extension (e.g. ``'train/data'``, ``'test/data'``).
    """
    stems = [f.stem for f in files]
    if len(set(stems)) == len(stems):
        return dict(zip(files, stems))
    return {f: f.relative_to(root).with_suffix('').as_posix()
            for f in files}


# ---------------------------------------------------------------------------
# Synthetic (generated) datasets -- GH #285
#
# Every other resolver in this module fetches or reads data; these MAKE it.
# They are registered by name in SYNTHETIC_DATASETS below, resolved by
# :func:`synthetic_dataset` (which returns None for an unknown name, the same
# "not mine" convention :func:`sklearn_dataset` and :func:`seaborn_dataset`
# use), and every one of them is deterministic given ``random_state``.
# ---------------------------------------------------------------------------

def _synthetic_rng(random_state):
    """``numpy.random.Generator`` for a hypertools ``random_state``.

    Accepts None (fresh entropy), an int seed, a ``SeedSequence``, an
    existing ``Generator``, or a legacy ``RandomState`` (whose own bit
    generator is reused, so a caller threading a ``RandomState`` through
    still gets a reproducible stream)."""
    if isinstance(random_state, np.random.RandomState):
        return np.random.default_rng(random_state.bit_generator)
    return np.random.default_rng(random_state)


def _synthetic_frame(x, target=None, target_name='target'):
    """(n_samples, n_features) array -> DataFrame with ``dim_0 ... dim_k``
    columns, plus ``target_name`` appended when scikit-learn gave us labels
    or manifold positions (mirroring :func:`sklearn_dataset`, which appends
    the bundled datasets' target the same way)."""
    df = pd.DataFrame(
        np.asarray(x, dtype=float),
        columns=[f'dim_{i}' for i in range(np.shape(x)[1])])
    if target is not None:
        df[target_name] = target
    return df


def random_walk(n_samples=300, n_features=10, step=1.0, drift=0.0,
                random_state=None):
    """A Gaussian random walk: the cumulative sum of ``n_samples`` i.i.d.
    normal steps in ``n_features`` dimensions.

    Parameters
    ----------
    n_samples : int
        Number of timepoints (rows).
    n_features : int
        Number of dimensions (columns).
    step : float
        Standard deviation of each step.
    drift : float
        Mean of each step (0 for an unbiased walk).
    random_state : int, SeedSequence, Generator, RandomState, or None
        Seed. The same seed always produces the same walk.

    Returns
    -------
    numpy.ndarray
        ``(n_samples, n_features)`` float array, rows ordered in time.
    """
    rng = _synthetic_rng(random_state)
    steps = rng.normal(loc=drift, scale=step, size=(int(n_samples),
                                                    int(n_features)))
    return np.cumsum(steps, axis=0)


def helix(n_samples=300, turns=3.0, noise=0.0, radius=1.0, pitch=1.0,
          random_state=None):
    """A 3D helix (spiral) sampled at ``n_samples`` evenly spaced angles.

    Parameters
    ----------
    n_samples : int
        Number of points (rows).
    turns : float
        Number of complete revolutions.
    noise : float
        Standard deviation of optional Gaussian noise added to every
        coordinate (0 for a noiseless helix).
    radius : float
        Radius of the circle traced in the x/y plane.
    pitch : float
        Rise along z per full revolution.
    random_state : int, SeedSequence, Generator, RandomState, or None
        Seed for the noise (ignored when ``noise`` is 0, in which case the
        result is fully deterministic).

    Returns
    -------
    numpy.ndarray
        ``(n_samples, 3)`` float array of x, y, z coordinates.
    """
    t = np.linspace(0.0, 2.0 * np.pi * float(turns), int(n_samples))
    xyz = np.column_stack([radius * np.cos(t),
                           radius * np.sin(t),
                           pitch * t / (2.0 * np.pi)])
    if noise:
        xyz = xyz + _synthetic_rng(random_state).normal(
            scale=float(noise), size=xyz.shape)
    return xyz


def lorenz(n_samples=2000, sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.01,
           x0=None, random_state=None):
    """A trajectory of the Lorenz system, integrated with fixed-step RK4.

    ``dx/dt = sigma * (y - x)``, ``dy/dt = x * (rho - z) - y``,
    ``dz/dt = x * y - beta * z``.

    Parameters
    ----------
    n_samples : int
        Number of integration steps to return (rows), spaced ``dt`` apart.
    sigma, rho, beta : float
        The system's parameters (the defaults are Lorenz's classic
        butterfly-attractor values).
    dt : float
        Integration step size.
    x0 : sequence of 3 floats or None
        Initial condition. When None (the default), it is ``(1, 1, 1)``
        plus a small random perturbation drawn from ``random_state``, so
        ``n_datasets > 1`` yields nearby trajectories that diverge -- the
        butterfly effect itself.
    random_state : int, SeedSequence, Generator, RandomState, or None
        Seed for that perturbation (unused when ``x0`` is given, which
        makes the result fully deterministic).

    Returns
    -------
    numpy.ndarray
        ``(n_samples, 3)`` float array of x, y, z coordinates in time
        order.
    """
    n_samples = int(n_samples)
    if x0 is None:
        state = np.array([1.0, 1.0, 1.0]) + _synthetic_rng(
            random_state).normal(scale=1e-3, size=3)
    else:
        state = np.asarray(x0, dtype=float).ravel()
        if state.size != 3:
            raise HypertoolsIOError(
                f'lorenz: x0 must have 3 elements (x, y, z); got '
                f'{state.size}')

    def _deriv(s):
        x, y, z = s
        return np.array([sigma * (y - x),
                         x * (rho - z) - y,
                         x * y - beta * z])

    out = np.empty((n_samples, 3), dtype=float)
    dt = float(dt)
    for i in range(n_samples):
        out[i] = state
        k1 = _deriv(state)
        k2 = _deriv(state + 0.5 * dt * k1)
        k3 = _deriv(state + 0.5 * dt * k2)
        k4 = _deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def _sklearn_synthetic(maker, target_name):
    """Wrap a ``sklearn.datasets.make_*`` generator as a hypertools
    synthetic dataset: every keyword argument is passed straight through
    to scikit-learn (so its own defaults and validation apply), and the
    ``(X, y)`` pair it returns becomes one DataFrame with the labels /
    manifold positions appended as ``target_name``."""
    def _make(random_state=None, **kwargs):
        from sklearn import datasets as sk_datasets
        x, y = getattr(sk_datasets, maker)(random_state=random_state,
                                           **kwargs)
        return _synthetic_frame(x, y, target_name)
    return _make


blobs = _sklearn_synthetic('make_blobs', 'target')
moons = _sklearn_synthetic('make_moons', 'target')
swiss_roll = _sklearn_synthetic('make_swiss_roll', 't')
s_curve = _sklearn_synthetic('make_s_curve', 't')

# name -> (generator, one-line description). The descriptions are the
# single source of truth for the synthetic half of hyp.load's docstring
# and docs/io.rst -- see synthetic_dataset_docs().
SYNTHETIC_DATASETS = {
    'random_walk': (
        random_walk,
        'Gaussian random walk: (n_samples, n_features) array, cumulative '
        'sum of normal steps. kwargs: n_samples=300, n_features=10, '
        'step=1.0, drift=0.0'),
    'helix': (
        helix,
        '3D helix: (n_samples, 3) array. kwargs: n_samples=300, '
        'turns=3.0, noise=0.0, radius=1.0, pitch=1.0'),
    'lorenz': (
        lorenz,
        "Lorenz attractor trajectory (fixed-step RK4): (n_samples, 3) "
        'array. kwargs: n_samples=2000, sigma=10.0, rho=28.0, beta=8/3, '
        'dt=0.01, x0=None'),
    'blobs': (
        blobs,
        'isotropic Gaussian blobs via sklearn.datasets.make_blobs: '
        "DataFrame of dim_* columns + a 'target' cluster label. Its "
        'kwargs (n_samples, n_features, centers, cluster_std, ...) pass '
        'straight through'),
    'moons': (
        moons,
        'two interleaving half-circles via sklearn.datasets.make_moons: '
        "DataFrame of dim_0/dim_1 + a 'target' label. Its kwargs "
        '(n_samples, noise, ...) pass straight through'),
    'swiss_roll': (
        swiss_roll,
        'Swiss-roll manifold via sklearn.datasets.make_swiss_roll: '
        "DataFrame of dim_0/1/2 + 't', the position along the roll. Its "
        'kwargs (n_samples, noise, hole, ...) pass straight through'),
    's_curve': (
        s_curve,
        'S-curve manifold via sklearn.datasets.make_s_curve: DataFrame of '
        "dim_0/1/2 + 't', the position along the curve. Its kwargs "
        '(n_samples, noise, ...) pass straight through'),
}


def synthetic_dataset_docs(indent='    '):
    """The registered synthetic datasets as a reStructuredText bullet
    list, so ``hyp.load``'s docstring and ``docs/io.rst`` can list them
    without duplicating :data:`SYNTHETIC_DATASETS`."""
    return '\n'.join(f'{indent}* ``{name}`` -- {doc}'
                     for name, (_, doc) in SYNTHETIC_DATASETS.items())


def synthetic_dataset(name, n_datasets=1, random_state=None, seed=None,
                      **kwargs):
    """Generate one of the built-in synthetic datasets by name (GH #285).

    Unlike every other resolver in this module, nothing is fetched or read
    from disk: the data is generated on the spot and is fully reproducible
    given ``random_state``.

    Parameters
    ----------
    name : str
        One of :data:`SYNTHETIC_DATASETS` (``'random_walk'``, ``'helix'``,
        ``'lorenz'``, ``'blobs'``, ``'moons'``, ``'swiss_roll'``,
        ``'s_curve'``).
    n_datasets : int
        How many independent datasets to generate (default 1). When
        greater than 1 the result is a *list* of that many datasets -- one
        hypertools multi-dataset, exactly the shape
        ``hypertools.load('weights')`` returns -- each generated from its
        own seed derived deterministically from ``random_state``.
    random_state : int, SeedSequence, Generator, RandomState, or None
        Seed. ``seed=`` is accepted as an alias (hypertools uses
        ``random_state`` throughout its sklearn-facing API, and ``seed``
        in its plotting/animation API; passing both raises unless they
        are equal).
    **kwargs
        Passed to the named generator (see :data:`SYNTHETIC_DATASETS` and
        the per-generator docstrings). An unknown keyword raises
        ``TypeError`` from the generator itself.

    Returns
    -------
    data : numpy.ndarray, pandas.DataFrame, list, or None
        The generated dataset (a list when ``n_datasets > 1``). None when
        ``name`` isn't a registered synthetic dataset -- callers should
        treat this as "not mine" and continue down the resolution chain,
        exactly like :func:`sklearn_dataset`.
    """
    if not isinstance(name, str):
        return None
    entry = SYNTHETIC_DATASETS.get(name)
    if entry is None:
        return None
    maker = entry[0]

    if seed is not None:
        if random_state is not None and random_state is not seed \
                and random_state != seed:
            raise HypertoolsIOError(
                f'{name!r}: pass either random_state= or its alias seed=, '
                f'not both with different values (got '
                f'random_state={random_state!r}, seed={seed!r})')
        random_state = seed

    try:
        n_datasets = int(n_datasets)
    except (TypeError, ValueError) as e:
        raise HypertoolsIOError(
            f'{name!r}: n_datasets must be a positive integer; got '
            f'{n_datasets!r}') from e
    if n_datasets < 1:
        raise HypertoolsIOError(
            f'{name!r}: n_datasets must be a positive integer; got '
            f'{n_datasets}')

    if n_datasets == 1:
        return maker(random_state=random_state, **kwargs)
    # independent-but-reproducible seeds for the list case: one
    # SeedSequence spawn per dataset when a seed was given, plain None
    # (fresh entropy) when it wasn't
    if random_state is None:
        seeds = [None] * n_datasets
    else:
        base = random_state if isinstance(random_state, np.random.SeedSequence) \
            else np.random.SeedSequence(
                _synthetic_rng(random_state).integers(2 ** 63))
        seeds = [int(child.generate_state(1)[0])
                 for child in base.spawn(n_datasets)]
    return [maker(random_state=s, **kwargs) for s in seeds]


# ---------------------------------------------------------------------------
# Web sources -- explicit 'wikipedia:' / 'yahoo:' / 'sec:' prefixes (GH #285)
#
# Like 'fivethirtyeight/' and 'kaggle/', these prefixes are unambiguous: a
# matching-but-failing name raises instead of falling through the rest of the
# chain. Wikipedia's and the SEC's access policies both ask for a
# User-Agent naming the client and a contact address, so every request here
# goes out with _contact_user_agent() rather than the bare 'hypertools' UA.
# ---------------------------------------------------------------------------

WEB_SOURCE_PREFIXES = ('wikipedia:', 'yahoo:', 'sec:')

# Matches an explicit web-source prefix followed by something non-blank, so
# ordinary prose ('wikipedia: the free encyclopedia') is not mistaken for a
# source name in is_loadable_string().
_WEB_PREFIX_RE = re.compile(r'^(wikipedia|yahoo|sec):\S')

_PROJECT_CONTACT_FALLBACK = 'contextualdynamics@gmail.com'
_contact_ua_cache = None

_SEC_TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'
_SEC_CONCEPT_URL = ('https://data.sec.gov/api/xbrl/companyconcept/'
                    'CIK{cik:010d}/{taxonomy}/{concept}.json')
_SEC_FACTS_URL = ('https://data.sec.gov/api/xbrl/companyfacts/'
                  'CIK{cik:010d}.json')
# XBRL taxonomies tried, in order, when the caller doesn't name one: the
# default concept (shares outstanding) is a 'dei' cover-page fact, while
# financial-statement concepts (Revenues, Assets, ...) live in 'us-gaap'.
_SEC_TAXONOMIES = ('dei', 'us-gaap')
_sec_cik_cache = None

_YAHOO_CHART_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'


def _contact_user_agent():
    """Request headers whose User-Agent names hypertools, its version, and
    the project's contact address.

    Wikipedia's User-Agent policy and the SEC's fair-access policy both ask
    automated clients to identify themselves with a contact, and the SEC
    returns 403 without one. The address is read from the *installed
    package metadata* (``Author-email``), which pip generates from
    ``pyproject.toml``'s ``authors`` -- so ``pyproject.toml`` stays the
    single declaration, and this still works from a wheel, where the file
    itself isn't shipped. Falls back to the project address if the metadata
    can't be read (e.g. hypertools imported from a source tree that was
    never installed).

    The string is deliberately ``hypertools/<version> (<contact>)`` with
    NO project URL in it: measured 2026-09-05, sec.gov's WAF answers HTTP
    403 to an otherwise identical User-Agent that contains a ``https://``
    URL, while both Wikipedia and the SEC accept the contact-only form.
    """
    global _contact_ua_cache
    if _contact_ua_cache is None:
        contact, ver = _PROJECT_CONTACT_FALLBACK, 'unknown'
        try:
            from importlib.metadata import metadata as _metadata
            from importlib.metadata import version as _version
            md = _metadata('hypertools')
            raw = md.get('Author-email') or md.get('Maintainer-email') or ''
            match = re.search(r'<([^>]+@[^>]+)>', raw)
            if match:
                contact = match.group(1)
            elif '@' in raw:
                contact = raw.strip()
            ver = _version('hypertools')
        except Exception:
            pass
        _contact_ua_cache = {'User-Agent': f'hypertools/{ver} ({contact})'}
    return dict(_contact_ua_cache)


def web_source(name, **kwargs):
    """Dispatch an explicit web-source prefix (GH #285).

    ``'wikipedia:<Title>'`` -> :func:`wikipedia_source`,
    ``'yahoo:<TICKER>'`` -> :func:`yahoo_source`,
    ``'sec:<TICKER>'`` -> :func:`sec_source`. Returns None (only) when
    ``name`` carries none of those prefixes, so callers can continue down
    the resolution chain; a prefixed name that fails raises
    :class:`~hypertools.core.exceptions.HypertoolsIOError`.
    """
    if not isinstance(name, str):
        return None
    if name.startswith('wikipedia:'):
        return wikipedia_source(name, **kwargs)
    if name.startswith('yahoo:'):
        return yahoo_source(name, **kwargs)
    if name.startswith('sec:'):
        return sec_source(name, **kwargs)
    return None


def _web_get(url, params=None, timeout=30, label=''):
    """GET a web-source URL with the contact User-Agent, wrapping transport
    failures in HypertoolsIOError and retrying transient 5xx gateway errors
    the same way the fivethirtyeight loader does."""
    headers = _contact_user_agent()
    delay, last_exc = 1.0, None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers,
                                timeout=timeout)
        except requests.RequestException as e:
            last_exc = e
            if attempt == 2:
                raise HypertoolsIOError(
                    f'could not reach {label or url}: '
                    f'{type(e).__name__}: {e}') from e
        else:
            if resp.status_code not in _TRANSIENT_STATUS or attempt == 2:
                return resp
        time.sleep(delay)
        delay *= 2
    raise last_exc  # unreachable: the loop always returns or raises


def wikipedia_source(name, lang='en', intro=False, timeout=30):
    """Plain-text extract of one or more Wikipedia articles (GH #285).

    ``name`` is ``'wikipedia:<Title>'``. Several articles can be requested
    at once by separating their titles with ``'|'``
    (``'wikipedia:Physics|Chemistry'``), which returns a list of strings in
    the order asked for; passing a *list* of ``'wikipedia:...'`` names to
    :func:`hypertools.load` does the same thing.

    Titles are fetched from the MediaWiki action API
    (``action=query&prop=extracts&explaintext=1``) with redirects followed,
    so ``'wikipedia:NYC'`` resolves to "New York City". Note that MediaWiki
    caps a full-text extract request at ONE title per request, so a
    multi-title name issues one request per title.

    Parameters
    ----------
    name : str
        ``'wikipedia:<Title>'``, or ``'wikipedia:<A>|<B>|<C>'``.
    lang : str
        Wikipedia language edition (default ``'en'`` ->
        en.wikipedia.org).
    intro : bool
        If True, return only the lead section (``exintro``) instead of the
        whole article.
    timeout : float
        Per-request timeout, in seconds.

    Returns
    -------
    text : str or list of str
        The article's plain text; a list when several titles were asked
        for.
    """
    if not isinstance(name, str) or not name.startswith('wikipedia:'):
        return None
    raw = name[len('wikipedia:'):].strip()
    titles = [t.strip() for t in raw.split('|') if t.strip()]
    if not titles:
        raise HypertoolsIOError(
            f"{name!r} is missing an article title -- expected "
            "'wikipedia:<Title>', e.g. 'wikipedia:Dartmouth College' "
            "(several titles can be separated by '|').")

    api = f'https://{lang}.wikipedia.org/w/api.php'
    extracts = []
    for title in titles:
        params = {'action': 'query', 'prop': 'extracts', 'explaintext': 1,
                  'redirects': 1, 'format': 'json', 'formatversion': 2,
                  'titles': title}
        if intro:
            params['exintro'] = 1
        resp = _web_get(api, params=params, timeout=timeout,
                        label=f'the MediaWiki API at {api}')
        if resp.status_code >= 400:
            raise HypertoolsIOError(
                f'the MediaWiki API at {api} returned HTTP '
                f'{resp.status_code} for {title!r}.')
        try:
            pages = resp.json()['query']['pages']
        except (ValueError, KeyError) as e:
            raise HypertoolsIOError(
                f'unexpected response from the MediaWiki API at {api} for '
                f'{title!r}: {type(e).__name__}: {e}') from e
        if not pages:
            raise HypertoolsIOError(
                f'no Wikipedia article matched {title!r} on '
                f'{lang}.wikipedia.org.')
        page = pages[0]
        if page.get('missing'):
            raise HypertoolsIOError(
                f'{title!r} is not a {lang}.wikipedia.org article title '
                '(the page does not exist). Titles are case-sensitive '
                'after the first letter; underscores and spaces are '
                'equivalent.')
        text = (page.get('extract') or '').strip()
        if not text:
            raise HypertoolsIOError(
                f'the Wikipedia article {page.get("title", title)!r} '
                'returned an empty plain-text extract (it may be a '
                'disambiguation or redirect-only page).')
        extracts.append(text)
    return extracts[0] if len(extracts) == 1 else extracts


def _to_epoch(value, default):
    """Epoch seconds from a date-like value (str, date, datetime, pandas
    Timestamp) or a number already in epoch seconds; ``default`` when
    None."""
    if value is None:
        return int(default)
    if isinstance(value, bool):
        raise HypertoolsIOError(f'invalid date bound: {value!r}')
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    try:
        return int(pd.Timestamp(value).timestamp())
    except (ValueError, TypeError) as e:
        raise HypertoolsIOError(
            f'could not interpret {value!r} as a date; pass a string '
            "('2020-01-01'), a datetime/date, or epoch seconds."
        ) from e


def yahoo_source(name, start=None, end=None, interval='1d', timeout=30):
    """Daily (or other-interval) price history for one ticker from Yahoo
    Finance's v8 chart endpoint (GH #285).

    ``name`` is ``'yahoo:<TICKER>'``, e.g. ``'yahoo:AAPL'``.

    The request always carries EXPLICIT ``period1``/``period2`` epoch
    bounds. ``range=max&interval=1d`` silently degrades to 3-month bars
    (measured 2026-09-03: AAPL came back with 169 rows since 1984), so the
    range form is deliberately not used.

    Parameters
    ----------
    name : str
        ``'yahoo:<TICKER>'``.
    start, end : str, date, datetime, epoch seconds, or None
        Inclusive window bounds. Defaults: the epoch (1970-01-01) and
        "now", i.e. the ticker's whole history.
    interval : str
        Yahoo's bar size -- ``'1d'`` (default), ``'1wk'``, ``'1mo'``, or an
        intraday size such as ``'1h'`` (intraday data is only served for
        recent windows).
    timeout : float
        Request timeout, in seconds.

    Returns
    -------
    pandas.DataFrame
        Indexed by a ``DatetimeIndex`` named ``'date'`` (bar timestamps
        normalized to midnight), with float columns ``open``, ``high``,
        ``low``, ``close``, ``volume`` and, when Yahoo provides it,
        ``adj_close`` (split/dividend-adjusted). Rows are in time order;
        gaps Yahoo reports as nulls stay NaN rather than being dropped.
    """
    if not isinstance(name, str) or not name.startswith('yahoo:'):
        return None
    ticker = name[len('yahoo:'):].strip()
    if not ticker:
        raise HypertoolsIOError(
            f"{name!r} is missing a ticker symbol -- expected "
            "'yahoo:<TICKER>', e.g. 'yahoo:AAPL'.")

    params = {'period1': _to_epoch(start, 0),
              'period2': _to_epoch(end, time.time() + 86400),
              'interval': interval}
    url = _YAHOO_CHART_URL.format(ticker=ticker)
    resp = _web_get(url, params=params, timeout=timeout,
                    label=f"Yahoo Finance's chart API for {ticker!r}")
    try:
        payload = resp.json()
    except ValueError as e:
        raise HypertoolsIOError(
            f'Yahoo Finance returned a non-JSON response (HTTP '
            f'{resp.status_code}) for {ticker!r}: {type(e).__name__}: {e}'
        ) from e
    chart = payload.get('chart') or {}
    error = chart.get('error')
    if error:
        raise HypertoolsIOError(
            f'Yahoo Finance rejected {ticker!r}: '
            f'{error.get("description", error)} (HTTP {resp.status_code}). '
            'Check the symbol on https://finance.yahoo.com.')
    results = chart.get('result') or []
    if not results:
        raise HypertoolsIOError(
            f'Yahoo Finance returned no result for {ticker!r} (HTTP '
            f'{resp.status_code}).')
    result = results[0]
    stamps = result.get('timestamp')
    if not stamps:
        raise HypertoolsIOError(
            f'Yahoo Finance returned no {interval} bars for {ticker!r} in '
            f'the requested window ({params["period1"]}..'
            f'{params["period2"]}, epoch seconds). Widen start=/end=, or '
            'note that intraday intervals are only served for recent '
            'windows.')
    quote = (result.get('indicators') or {}).get('quote') or [{}]
    quote = quote[0]
    index = pd.to_datetime(stamps, unit='s').normalize()
    index.name = 'date'
    frame = {}
    for col in ('open', 'high', 'low', 'close', 'volume'):
        if quote.get(col) is not None:
            frame[col] = pd.Series(quote[col], index=index, dtype=float)
    adj = (result.get('indicators') or {}).get('adjclose')
    if adj and adj[0].get('adjclose') is not None:
        frame['adj_close'] = pd.Series(adj[0]['adjclose'], index=index,
                                       dtype=float)
    if not frame:
        raise HypertoolsIOError(
            f'Yahoo Finance returned bars for {ticker!r} with no price '
            'columns.')
    order = [c for c in ('open', 'high', 'low', 'close', 'adj_close',
                         'volume') if c in frame]
    return pd.DataFrame(frame, columns=order).sort_index()


def _sec_cik_map(timeout=30):
    """``{TICKER: CIK}`` from the SEC's public ticker file, cached
    per-process (it is ~800 KB and changes rarely)."""
    global _sec_cik_cache
    if _sec_cik_cache is None:
        resp = _web_get(_SEC_TICKERS_URL, timeout=timeout,
                        label="the SEC's company_tickers.json")
        if resp.status_code >= 400:
            raise HypertoolsIOError(
                f'the SEC ticker file ({_SEC_TICKERS_URL}) returned HTTP '
                f'{resp.status_code}. The SEC requires a User-Agent with a '
                'contact address; hypertools sends one, so this is most '
                'likely a rate limit (10 requests/second) or an outage.')
        try:
            rows = resp.json()
        except ValueError as e:
            raise HypertoolsIOError(
                f'could not parse the SEC ticker file: '
                f'{type(e).__name__}: {e}') from e
        _sec_cik_cache = {str(row['ticker']).upper(): int(row['cik_str'])
                          for row in rows.values()}
    return _sec_cik_cache


def _sec_facts_frame(facts, unit, dedupe):
    """XBRL fact records ({unit: [fact, ...]}) -> DataFrame indexed by
    period end."""
    rows = []
    for unit_name, records in facts.items():
        if unit is not None and unit_name != unit:
            continue
        for record in records:
            rows.append({**record, 'unit': unit_name})
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    frame = frame.rename(columns={'val': 'value'})
    for col in ('end', 'start', 'filed'):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col], errors='coerce')
    frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
    sort_cols = [c for c in ('end', 'filed') if c in frame]
    frame = frame.sort_values(sort_cols)
    if dedupe:
        # one value per period end: the LATEST filing wins over amendments
        frame = frame.drop_duplicates('end', keep='last')
    frame = frame.set_index('end')
    order = [c for c in ('value', 'unit', 'start', 'filed', 'form', 'fy',
                         'fp', 'frame', 'accn') if c in frame.columns]
    return frame[order + [c for c in frame.columns if c not in order]]


def sec_source(name, concept='EntityCommonStockSharesOutstanding',
               taxonomy=None, unit=None, dedupe=True, timeout=30):
    """One XBRL concept's reported values for a filer, from the SEC's
    public company-facts API (GH #285).

    ``name`` is ``'sec:<TICKER>'``, e.g. ``'sec:AAPL'``. The ticker is
    mapped to its CIK through the SEC's ``company_tickers.json`` (cached
    per-process).

    The per-concept ``companyconcept`` endpoint is tried first and the
    complete ``companyfacts`` file is used as a fallback, because
    ``companyconcept`` comes back EMPTY for some filers whose
    ``companyfacts`` carries the same concept (measured 2026-09-03: ABT and
    KO for the default shares-outstanding concept).

    Every request carries a User-Agent with the project's contact address,
    as the SEC's fair-access policy requires (it answers 403 without one).

    Parameters
    ----------
    name : str
        ``'sec:<TICKER>'``.
    concept : str
        XBRL concept/tag name (default
        ``'EntityCommonStockSharesOutstanding'``, the cover-page share
        count). Financial-statement concepts (``'Revenues'``,
        ``'Assets'``, ...) work too.
    taxonomy : str or None
        XBRL taxonomy holding the concept (``'dei'``, ``'us-gaap'``, ...).
        None (the default) tries ``'dei'`` then ``'us-gaap'``, and lets the
        companyfacts fallback search every taxonomy the filer reports.
    unit : str or None
        Keep only facts reported in this unit (e.g. ``'shares'``,
        ``'USD'``). None keeps every unit and labels each row in a
        ``'unit'`` column.
    dedupe : bool
        When True (default), keep one row per period end -- the most
        recently filed value, so an amended figure supersedes the
        original. False returns every filed fact, amendments included.
    timeout : float
        Per-request timeout, in seconds.

    Returns
    -------
    pandas.DataFrame
        Indexed by period end (``DatetimeIndex`` named ``'end'``), with a
        numeric ``'value'`` column plus the fact's metadata (``unit``,
        ``start``, ``filed``, ``form``, ``fy``, ``fp``, ``frame``,
        ``accn``) where the SEC reports it.
    """
    if not isinstance(name, str) or not name.startswith('sec:'):
        return None
    ticker = name[len('sec:'):].strip().upper()
    if not ticker:
        raise HypertoolsIOError(
            f"{name!r} is missing a ticker symbol -- expected "
            "'sec:<TICKER>', e.g. 'sec:AAPL'.")
    ciks = _sec_cik_map(timeout=timeout)
    if ticker not in ciks:
        raise HypertoolsIOError(
            f'{ticker!r} is not in the SEC ticker file '
            f'({_SEC_TICKERS_URL}), so hypertools cannot map it to a CIK. '
            'Only SEC registrants (US-listed filers) are there; check the '
            'symbol at https://www.sec.gov/cgi-bin/browse-edgar?action='
            'getcompany.')
    cik = ciks[ticker]
    taxonomies = (taxonomy,) if taxonomy else _SEC_TAXONOMIES

    for tax in taxonomies:
        url = _SEC_CONCEPT_URL.format(cik=cik, taxonomy=tax, concept=concept)
        resp = _web_get(url, timeout=timeout,
                        label=f"the SEC's companyconcept API for {ticker}")
        if resp.status_code == 404:
            continue
        if resp.status_code >= 400:
            raise HypertoolsIOError(
                f'the SEC companyconcept API returned HTTP '
                f'{resp.status_code} for {ticker} ({tax}/{concept}): {url}')
        try:
            units = resp.json().get('units') or {}
        except ValueError:
            units = {}
        frame = _sec_facts_frame(units, unit, dedupe)
        if frame is not None and len(frame):
            return frame

    # companyconcept was missing or EMPTY -- fall back to the filer's
    # complete facts file, which carries the same concept for filers whose
    # per-concept endpoint returns nothing
    url = _SEC_FACTS_URL.format(cik=cik)
    resp = _web_get(url, timeout=timeout,
                    label=f"the SEC's companyfacts API for {ticker}")
    if resp.status_code >= 400:
        raise HypertoolsIOError(
            f'the SEC companyfacts API returned HTTP {resp.status_code} '
            f'for {ticker} (CIK {cik:010d}): {url}')
    try:
        all_facts = resp.json().get('facts') or {}
    except ValueError as e:
        raise HypertoolsIOError(
            f'could not parse the SEC companyfacts response for {ticker}: '
            f'{type(e).__name__}: {e}') from e
    search = taxonomies if taxonomy else tuple(all_facts)
    for tax in search:
        entry = (all_facts.get(tax) or {}).get(concept)
        if not entry:
            continue
        frame = _sec_facts_frame(entry.get('units') or {}, unit, dedupe)
        if frame is not None and len(frame):
            return frame
    raise HypertoolsIOError(
        f'the SEC reports no {concept!r} facts for {ticker} (CIK '
        f'{cik:010d})'
        + (f' in the {taxonomy!r} taxonomy' if taxonomy else '')
        + (f' in unit {unit!r}' if unit else '')
        + '. Browse what the filer does report at '
        f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json.')


# ---------------------------------------------------------------------------
# On-disk cache for arbitrary URL downloads (GH #285)
# ---------------------------------------------------------------------------

def url_cache_dir():
    """Directory holding cached URL downloads.

    Defaults to ``urls/`` inside the same ``~/hypertools_data`` directory
    the built-in example datasets are cached in (``hypertools.io.load.
    DATA_DIR``); set ``HYPERTOOLS_URL_CACHE`` to put it somewhere else
    (useful for tests and for read-only home directories)."""
    override = os.environ.get('HYPERTOOLS_URL_CACHE')
    if override:
        return Path(override).expanduser()
    from .load import DATA_DIR
    return Path(DATA_DIR) / 'urls'


def cached_url_path(url):
    """Path this URL's cached copy lives at: the SHA-256 of the full URL
    (so query strings and rlkey tokens can't collide), keeping the URL's
    filename extension when it has a recognized one so the cached file is
    still identifiable by hand."""
    import hashlib
    digest = hashlib.sha256(url.encode('utf-8')).hexdigest()
    suffix = Path(url.split('?')[0].rstrip('/')).suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        suffix = ''
    return url_cache_dir() / f'{digest}{suffix}'


def _write_cached(path, raw, name_hint):
    """Write ``raw`` into the cache atomically: a per-process ``.part``
    file, then ``os.replace``, so an interrupted download can never leave a
    truncated file that later runs would trust. The download's filename
    hint is stored beside it so a cache hit parses the payload exactly the
    way the live download did."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f'{path.name}.{os.getpid()}.part')
    part.write_bytes(raw)
    os.replace(part, path)
    if name_hint:
        meta = path.with_name(f'{path.name}.meta.json')
        meta_part = meta.with_name(f'{meta.name}.{os.getpid()}.part')
        meta_part.write_text(json.dumps({'url_name_hint': name_hint}),
                             encoding='utf-8')
        os.replace(meta_part, meta)


def _read_cached(path):
    """``(bytes, name_hint)`` from a cached download, or None when it isn't
    cached."""
    import json
    if not path.is_file():
        return None
    meta = path.with_name(f'{path.name}.meta.json')
    name_hint = None
    if meta.is_file():
        try:
            name_hint = json.loads(meta.read_text(encoding='utf-8')).get(
                'url_name_hint')
        except (ValueError, OSError):
            name_hint = None
    return path.read_bytes(), name_hint


def is_loadable_string(s):
    """Cheap (no-network) check: could this string plausibly name a data
    source? Used to decide whether to route strings through load() rather
    than treating them as raw text to embed."""
    from .load import EXAMPLE_DATA
    if not isinstance(s, str):
        return False
    # explicit web-source prefixes are checked BEFORE the whitespace guard,
    # since article titles contain spaces ('wikipedia:Dartmouth College');
    # the pattern requires a non-blank character right after the colon, so
    # ordinary prose ('wikipedia: the free encyclopedia') is not matched
    if _WEB_PREFIX_RE.match(s.strip()):
        return True
    if not s.strip() or any(c.isspace() for c in s.strip()):
        return False
    if s in EXAMPLE_DATA or s in SYNTHETIC_DATASETS:
        return True
    if s.startswith('fivethirtyeight/') or s.startswith('kaggle/'):
        return True
    try:
        if Path(s).expanduser().is_file():
            return True
    except OSError:
        return False
    if s.startswith(('http://', 'https://')):
        return True
    if 'drive.google.com' in s or 'docs.google.com' in s \
            or 'dropbox.com' in s:
        return True
    if _DRIVE_ID_RE.match(s):
        return True
    if _DOMAIN_RE.match(s):
        return True
    if _HF_ID_RE.match(s):
        return True
    return False


def load_source(source, split=None, streaming=False, trust=False,
                extra_attempts=None, cache=False, offline=False,
                decode_labels=True, **source_kwargs):
    """Resolve one non-builtin string source (steps 6-13 of the chain --
    built-in/scikit-learn/seaborn names and the explicit
    fivethirtyeight/kaggle prefixes, steps 1-5, are tried by
    :func:`hypertools.load` before this function is called).

    Returns the loaded dataset (DataFrame, array, list, dict, or a
    Hugging Face [Iterable]Dataset). Raises HypertoolsIOError listing
    every attempted interpretation when nothing works.

    ``trust`` is threaded from :func:`hypertools.load`: it opts in to
    remote deserialization -- unpickling a remote payload and re-enabling
    ``allow_pickle`` for remote .npy/.npz payloads (see ``_parse_payload``).
    The default, ``trust=False``, REFUSES to unpickle a remote payload
    (raising ``HypertoolsTrustError``); it is a security boundary, not a
    warning.

    ``extra_attempts`` optionally seeds the "tried, in order" list with
    descriptions of resolvers already attempted by the caller (e.g. the
    scikit-learn/seaborn dataset-name lookups), so the final error message
    reflects the whole chain.

    ``cache``/``offline`` govern the on-disk URL cache (see
    :func:`url_cache_dir`): ``cache=True`` stores every URL/Drive/Dropbox/
    Sheets download and reuses it next time, ``offline=True`` reads ONLY
    from that cache and raises rather than touching the network.
    ``decode_labels`` is threaded to :func:`_load_hf`. Any remaining
    keyword arguments belong to the synthetic (step 6) and web-prefix
    (step 7) sources and are passed to whichever of those matches; passing
    them with any other kind of source raises ``TypeError``, so a typo
    can't be silently ignored.
    """
    attempts = list(extra_attempts) if extra_attempts else []

    # 6. built-in synthetic dataset: generated, never fetched. Checked
    # before local-file resolution, so the same shadowing rule as the
    # scikit-learn/seaborn names applies (pass './helix' or a path with an
    # extension to load a local file of that name instead).
    synthetic = synthetic_dataset(source, **source_kwargs) \
        if source in SYNTHETIC_DATASETS else None
    if synthetic is not None:
        return synthetic
    attempts.append('synthetic dataset: not one of '
                    f'{sorted(SYNTHETIC_DATASETS)}')

    # 7. web source with an explicit prefix ('wikipedia:', 'yahoo:',
    # 'sec:'): unambiguous, so a failure raises rather than falling
    # through the rest of the chain
    if isinstance(source, str) and source.startswith(WEB_SOURCE_PREFIXES):
        return web_source(source, **source_kwargs)

    if source_kwargs:
        raise TypeError(
            f'hypertools.load: unexpected keyword argument(s) '
            f'{sorted(source_kwargs)} for {source!r}. Extra keywords are '
            'only accepted by the synthetic datasets '
            f'({sorted(SYNTHETIC_DATASETS)}) and the web sources '
            f'({list(WEB_SOURCE_PREFIXES)}).')

    is_url_like = source.startswith(('http://', 'https://')) \
        or 'drive.google.com' in source or 'docs.google.com' in source \
        or 'dropbox.com' in source

    # 8. local file (skipped for explicit URLs, which are never local
    # paths -- the digest used to list a slash-collapsed 'https:/...'
    # local-file attempt for URL inputs, QC 2026-07 F19-013)
    if not is_url_like:
        path = Path(source).expanduser()
        try:
            is_file = path.is_file()
            is_dir = path.is_dir()
        except OSError:
            is_file = is_dir = False
        if is_file:
            return load_local_file(path)
        if is_dir:
            attempts.append(
                f'local file: {path} is a directory, not a file')
        else:
            attempts.append(f'local file: not found at {path}')

    # 9. Hugging Face dataset (skip for obvious URLs)
    if not is_url_like and _HF_ID_RE.match(source):
        try:
            return _load_hf(source, split=split, streaming=streaming,
                            decode_labels=decode_labels)
        except ImportError:
            raise
        except Exception as e:  # dataset not found, gated, etc.
            attempts.append(f'Hugging Face dataset: {type(e).__name__}: '
                            f'{str(e).splitlines()[0][:120]}')

    # 10. Google Sheets URL -> CSV export (checked before generic Drive id
    # extraction, since a Sheets URL also matches the '/d/<id>' pattern)
    sheet_url = _normalize_google_sheet(source)
    if sheet_url is not None:
        try:
            raw, name_hint = _fetch_bytes(sheet_url, cache=cache,
                                          offline=offline)
            return _parse_payload(raw, name_hint or 'sheet.csv',
                                  trust=trust, remote=True)
        except (HypertoolsTrustError, HypertoolsOfflineError):
            raise
        except Exception as e:
            attempts.append(f'Google Sheets: {type(e).__name__}: {e}')

    # 11. Google Drive URL or bare ID
    drive_id = _extract_drive_id(source)
    if drive_id is not None:
        url = f'https://drive.google.com/uc?export=download&id={drive_id}'
        try:
            raw, name_hint = _fetch_bytes(url, cache=cache,
                                          offline=offline)
            return _parse_payload(raw, name_hint or source,
                                  trust=trust, remote=True)
        except (HypertoolsTrustError, HypertoolsOfflineError):
            raise
        except Exception as e:
            attempts.append(f'Google Drive ({drive_id}): '
                            f'{type(e).__name__}: {e}')

    # 12. Dropbox URL or shared-link path
    dropbox_url = _normalize_dropbox(source)
    if dropbox_url is not None:
        try:
            raw, name_hint = _fetch_bytes(dropbox_url, cache=cache,
                                          offline=offline)
            return _parse_payload(raw, name_hint or source,
                                  trust=trust, remote=True)
        except (HypertoolsTrustError, HypertoolsOfflineError):
            raise
        except Exception as e:
            attempts.append(f'Dropbox: {type(e).__name__}: {e}')

    # 13. any URL, with or without a scheme
    url = None
    if source.startswith(('http://', 'https://')):
        url = source
    elif '/' not in source and \
            Path(source).suffix.lower() in _SUPPORTED_EXTENSIONS:
        # a bare data-file name (e.g. a typo'd local filename such as
        # 'results.csv') used to be promoted to https://<filename> and
        # trigger a real DNS lookup (QC 2026-07, F19-load-external-006)
        attempts.append(
            f'URL: not attempted -- {source!r} looks like a (missing) '
            'local file name; pass an explicit http(s):// URL to load it '
            'from the web')
    elif _DOMAIN_RE.match(source):
        url = 'https://' + source
    elif _HOST_PORT_RE.match(source):
        attempts.append(
            f'URL: not attempted -- {source!r} looks like a host:port '
            'address without a scheme; add an explicit http:// or '
            'https:// prefix')
    if url is not None:
        try:
            raw, name_hint = _fetch_bytes(url, cache=cache,
                                          offline=offline)
            return _parse_payload(raw, name_hint or source,
                                  trust=trust, remote=True)
        except (HypertoolsTrustError, HypertoolsOfflineError):
            raise
        except Exception as e:
            attempts.append(f'URL ({url}): {type(e).__name__}: {e}')

    tried = '\n  - '.join(attempts) if attempts else 'no interpretation ' \
        'matched (not a file, URL, Drive/Dropbox link, or dataset id)'
    message = f'could not load {source!r}. Tried, in order:\n  - {tried}'
    suggestion = _closest_dataset_name(source)
    if suggestion is not None:
        message += f"\nDid you mean {suggestion!r}?"
    raise HypertoolsIOError(message)


def _closest_dataset_name(source):
    """Near-miss suggestion for the could-not-load digest: the closest
    built-in example / scikit-learn / (cached) seaborn dataset name to
    ``source``, or None when nothing is close (QC 2026-07,
    F18-load-hosted-007)."""
    import difflib

    from .load import EXAMPLE_DATA
    if not isinstance(source, str):
        return None
    candidates = set(EXAMPLE_DATA) | set(SKLEARN_DATASETS) | \
        set(SYNTHETIC_DATASETS)
    if _seaborn_names_cache:
        candidates |= set(_seaborn_names_cache)
    matches = difflib.get_close_matches(
        source.strip().lower(), sorted(candidates), n=1, cutoff=0.85)
    return matches[0] if matches else None


def load_local_file(path):
    """Load a local data file by extension. Supports pickle/.geo,
    .npy/.npz, .csv/.tsv/.txt, .json, .parquet, .mat, .xlsx/.xls, and
    gzip-compressed variants (.gz). Files with no extension are parsed by
    content sniffing; files with any OTHER extension raise
    :class:`~hypertools.core.exceptions.HypertoolsIOError` unless their
    content matches a recognized binary format (pickle/npy/zip)."""
    path = Path(path)
    return _parse_payload(path.read_bytes(), path.name)


def _load_hf(name, split=None, streaming=False, decode_labels=True):
    """Load a Hugging Face dataset by id.

    A non-streaming load returns a DataFrame. ``ClassLabel`` columns are
    decoded to their string names by default (GH #285): the integer codes
    ``Dataset.to_pandas()`` produces are meaningless without the feature
    spec, and a streaming load already exposes the dataset's own label
    names, so the two paths now agree. Pass ``decode_labels=False`` to
    keep the raw integer codes.

    A streaming load returns the ``IterableDataset`` unchanged, for
    :func:`hypertools.plot` to consume chunk by chunk.
    """
    # quiet the Hugging Face download/progress chatter unless the user has
    # said otherwise -- these are read at IMPORT time by huggingface_hub and
    # tokenizers, so they have to be set before the lazy import below (the
    # same thing hypertools.tools.text2mat does before importing
    # sentence-transformers). setdefault: an explicit value always wins.
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    os.environ.setdefault('HF_HUB_VERBOSITY', 'error')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from .._shared.lazy_import import lazy_import
    datasets = lazy_import('datasets', purpose=f'loading the Hugging Face dataset {name!r}')
    ds = datasets.load_dataset(name, split=split, streaming=streaming)
    if split is None and hasattr(ds, 'keys'):  # (Iterable)DatasetDict
        keys = list(ds.keys())
        pick = 'train' if 'train' in keys else keys[0]
        ds = ds[pick]
    if streaming:
        return ds  # IterableDataset: stream it straight into hyp.plot
    df = ds.to_pandas()
    if decode_labels:
        df = _decode_class_labels(df, getattr(ds, 'features', None),
                                  datasets.ClassLabel)
    return df


def _decode_class_labels(df, features, class_label_type):
    """Replace every integer-coded ``ClassLabel`` column of ``df`` with its
    string names, in place (GH #285).

    Only top-level ``ClassLabel`` features are decoded; nested ones (e.g.
    a ``Sequence`` of ``ClassLabel``, as token-classification datasets
    use) keep their integer codes, since their names are per-element and
    the column holds arrays rather than scalars. Out-of-range or null
    codes are left as-is rather than raising."""
    for col, feature in (features or {}).items():
        if not isinstance(feature, class_label_type) or col not in df:
            continue
        names = list(feature.names)

        def _name(code, names=names):
            if pd.isna(code):
                return code
            code = int(code)
            return names[code] if 0 <= code < len(names) else code

        df[col] = df[col].map(_name)
    return df


def _extract_drive_id(s):
    """File id from a Google Drive URL, or the string itself when it is a
    bare Drive id."""
    if 'drive.google.com' in s or 'docs.google.com' in s:
        m = re.search(r'/d/([A-Za-z0-9_-]{20,})', s) or \
            re.search(r'[?&]id=([A-Za-z0-9_-]{20,})', s)
        if m:
            return m.group(1)
    if _DRIVE_ID_RE.match(s):
        return s
    return None


def _normalize_google_sheet(s):
    """Rewrite a Google Sheets URL to its CSV export URL, or None if ``s``
    isn't a Sheets URL."""
    m = _SHEETS_ID_RE.search(s)
    if m is None:
        return None
    return (f'https://docs.google.com/spreadsheets/d/{m.group(1)}'
            '/export?format=csv')


def _normalize_dropbox(s):
    """Direct-download URL from a Dropbox URL or shared-link path
    (e.g. 's/abc/file.pkl' or 'scl/fi/<id>/file.csv?rlkey=...')."""
    if 'dropbox.com' in s:
        url = s if s.startswith(('http://', 'https://')) else 'https://' + s
        url = url.replace('dl=0', 'dl=1')
        if 'dl=1' not in url:
            url += ('&' if '?' in url else '?') + 'dl=1'
        return url
    if s.startswith(('s/', 'scl/fi/')):
        return _normalize_dropbox('https://www.dropbox.com/' + s)
    return None


def _looks_like_html(raw, ctype):
    return raw[:1] == b'<' and ('html' in ctype or b'<html' in raw[:512].lower())


def _name_hint_from_url(url):
    """Filename hint from a URL's own tail (the fallback when a response
    carries no Content-Disposition, and the only hint available for a
    cached download read back without headers)."""
    tail = url.split('?')[0].rstrip('/').rsplit('/', 1)[-1]
    return tail if '.' in tail else None


def _name_hint(resp, url):
    dispo = resp.headers.get('Content-Disposition', '')
    m = re.search(r'filename="?([^";]+)"?', dispo)
    if m:
        return m.group(1)
    return _name_hint_from_url(url)


def parse_drive_interstitial(html):
    """Parse a Google Drive "can't scan this file for viruses" large-file
    interstitial page, returning (action_url, params) for the real
    download, or None if ``html`` isn't one of these pages."""
    m = _DRIVE_INTERSTITIAL_FORM_RE.search(html)
    if m is None:
        return None
    action_url = m.group(1)
    params = dict(_HIDDEN_INPUT_RE.findall(m.group(2)))
    return action_url, params


def _fetch_bytes(url, timeout=60, cache=False, offline=False):
    """Download url -> (bytes, filename_hint). Automatically follows the
    Google Drive large-file virus-scan interstitial (a confirm form served
    in place of the file); raises on HTTP errors and on any other HTML
    interstitial (e.g. rate-limit/permission pages).

    ``cache=True`` serves the download from the on-disk cache
    (:func:`url_cache_dir`) when it is already there, and stores it there
    (atomically) when it isn't. ``offline=True`` reads ONLY from that
    cache and never opens a connection: an uncached URL raises
    :class:`~hypertools.core.exceptions.HypertoolsIOError` naming the
    cache path it looked for. The default (``cache=False,
    offline=False``) downloads without writing anything to disk, exactly
    as before.
    """
    if cache or offline:
        path = cached_url_path(url)
        hit = _read_cached(path)
        if hit is not None:
            raw, name_hint = hit
            if not raw:
                raise HypertoolsIOError(
                    f'the cached copy of {url} at {path} is empty (0 '
                    'bytes) -- delete it and retry to re-download.')
            return raw, name_hint or _name_hint_from_url(url)
        if offline:
            raise HypertoolsOfflineError(
                f'offline=True, but {url} is not in the hypertools URL '
                f'cache (looked for {path}). Load it once with '
                'cache=True while online -- or set HYPERTOOLS_URL_CACHE '
                'to the directory holding an existing cache -- then '
                'offline=True will read it from there.')

    resp = requests.get(url, headers=_UA, timeout=timeout,
                        allow_redirects=True)
    resp.raise_for_status()
    raw = resp.content
    if not raw:
        raise HypertoolsIOError(f'empty response from {url}')

    if _looks_like_html(raw, resp.headers.get('Content-Type', '')):
        parsed = parse_drive_interstitial(raw.decode('utf-8', errors='replace'))
        if parsed is None:
            raise HypertoolsIOError(
                f'{url} returned an HTML page instead of data (rate '
                'limit, permission page, or a link that needs a '
                'direct-download form)')
        action_url, params = parsed
        resp = requests.get(action_url, params=params, headers=_UA,
                            timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        raw = resp.content
        if not raw:
            raise HypertoolsIOError(f'empty response from {action_url}')
        if _looks_like_html(raw, resp.headers.get('Content-Type', '')):
            raise HypertoolsIOError(
                f'{url} returned an HTML page instead of data (rate '
                'limit, permission page, or a link that needs a '
                'direct-download form)')
        name_hint = _name_hint(resp, action_url)
        if cache:
            _write_cached(cached_url_path(url), raw, name_hint)
        return raw, name_hint

    name_hint = _name_hint(resp, url)
    if cache:
        _write_cached(cached_url_path(url), raw, name_hint)
    return raw, name_hint


# Transparent gzip decompression is capped so a tiny payload that inflates
# to an enormous buffer (a "gzip bomb", up to ~1000x expansion per layer)
# cannot exhaust memory -- a few MB fetched from a remote URL could
# otherwise balloon into many GB (release-1.0 audit: security re-review of
# F19-load-external-005). 2 GiB is deliberately generous: far larger than
# any dataset hypertools could sensibly materialize in memory from a
# single file, while still bounding the damage.
_MAX_GZIP_INFLATED_BYTES = 2 * 1024 ** 3  # 2 GiB


def _complete_pickle_stream(raw):
    """True when ``raw`` parses as one complete pickle stream (any
    protocol), WITHOUT executing anything.

    Protocol >= 2 pickles are recognized by their ``b'\\x80'`` magic byte,
    but protocol-0 pickles are plain ASCII with no magic prefix, so an
    extensionless protocol-0 pickle used to fall through to the
    delimited-text parser and come back silently as a garbage DataFrame
    (release-1.0 audit: re-review of X2-error-quality-001).
    ``pickletools.genops`` only PARSES opcodes -- no unpickling happens
    here -- and the STOP opcode is required to be the payload's FINAL
    byte (modulo trailing whitespace), exactly where ``pickle.dumps``
    puts it. Real text/CSV data would have to be a valid pickle program
    for its ENTIRE length to false-positive: merely starting with
    opcode-like bytes (e.g. ``'0.5,...'`` parses as POP + STOP) is not
    enough."""
    import pickletools

    try:
        for opcode, _arg, pos in pickletools.genops(raw):
            if opcode.name == 'STOP':
                return not raw[pos + 1:].strip()
        return False
    except Exception:
        return False


def _gunzip_capped(raw, label):
    """Decompress gzip bytes, refusing (with ``HypertoolsIOError``) to
    inflate past ``_MAX_GZIP_INFLATED_BYTES``. Reads in bounded chunks so
    an oversized payload is rejected without ever materializing it."""
    import gzip

    chunks, total = [], 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            while True:
                chunk = gz.read(64 * 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_GZIP_INFLATED_BYTES:
                    raise HypertoolsIOError(
                        f'{label} decompresses to more than '
                        f'{_MAX_GZIP_INFLATED_BYTES // 1024 ** 3} GiB, so '
                        'hypertools refused to inflate it (transparent '
                        'gzip decompression is capped to keep a small '
                        'malicious "gzip bomb" payload from exhausting '
                        'memory). If the file is genuinely this large, '
                        'decompress it yourself (e.g. `gunzip`) and load '
                        'the result in a memory-appropriate way.')
                chunks.append(chunk)
    except HypertoolsIOError:
        # HypertoolsIOError subclasses OSError -- re-raise the cap error
        # before the corruption handler below can swallow it
        raise
    except (OSError, EOFError) as e:
        raise HypertoolsIOError(
            f'{label} looks gzip-compressed but could not be '
            f'decompressed ({e}); the file may be corrupted.') from e
    return b''.join(chunks)


def _parse_payload(raw, name_hint='', trust=False, remote=False):
    """Parse downloaded/read bytes into a dataset, by filename extension
    first and content sniffing second (extensionless payloads only).

    ``remote`` marks payloads fetched over the network (as opposed to a
    local file): unpickling a remote payload without ``trust=True`` is
    REFUSED, raising ``HypertoolsTrustError`` (arbitrary code in a pickle
    would run on load), and remote .npy/.npz use ``allow_pickle=False``
    unless ``trust=True`` (a remote .npy/.npz that needs pickled objects
    likewise raises ``HypertoolsTrustError``). Local files are never
    subject to this policy.
    """
    label = str(name_hint) or 'payload'
    if not raw:
        raise HypertoolsIOError(
            f'{label} is empty (0 bytes) -- nothing to load. If a save '
            'writing this file failed midway, re-run it.')

    # gzip-compressed payloads (e.g. data.csv.gz, a common scientific
    # artifact) are decompressed transparently -- capped at
    # _MAX_GZIP_INFLATED_BYTES to block gzip bombs -- and re-dispatched
    # on the inner name (QC 2026-07, F19-load-external-005)
    if raw[:2] == b'\x1f\x8b':
        inflated = _gunzip_capped(raw, label)
        inner = Path(label)
        inner_name = inner.stem if inner.suffix.lower() == '.gz' else label
        return _parse_payload(inflated, inner_name, trust=trust,
                              remote=remote)

    ext = Path(label).suffix.lower()
    allow_pickle = trust or not remote

    if ext == '.gz':
        raise HypertoolsIOError(
            f'{label} is named .gz but does not start with the gzip magic '
            'bytes; the file may be corrupted or mis-named.')
    if ext == '.npy':
        return _npy_load(raw, allow_pickle)
    if ext == '.npz':
        return _unpack_npz(raw, trust=trust, remote=remote)
    if ext in ('.csv', '.tsv', '.txt'):
        sep = '\t' if ext == '.tsv' else None
        return _read_delimited_text(raw, label, sep=sep)
    if ext == '.json':
        return pd.read_json(io.BytesIO(raw))
    if ext == '.parquet':
        return pd.read_parquet(io.BytesIO(raw))
    if ext == '.mat':
        return _unpack_mat(raw)
    if ext == '.xlsx':
        from .._shared.lazy_import import lazy_import
        lazy_import('openpyxl', purpose='.xlsx files')     # installs [io] on demand
        return pd.read_excel(io.BytesIO(raw))
    if ext == '.xls':
        return _read_xls(raw)
    if ext in ('.pkl', '.pickle', '.geo', '.p'):
        return _unpickle_bytes(raw, trust=trust, remote=remote)

    if ext:
        # unsupported extension: load recognized binary content by its
        # signature (e.g. hyp.save pickles regardless of extension), but
        # never fall through to a delimiter-sniffed CSV parse of arbitrary
        # bytes -- that silently fabricated garbage DataFrames (QC 2026-07,
        # X2-error-quality-001)
        if raw[:6] == b'\x93NUMPY':
            return _npy_load(raw, allow_pickle)
        if raw[:1] == b'\x80':
            return _unpickle_bytes(raw, trust=trust, remote=remote)
        if raw[:2] == b'PK':
            try:
                return _unpack_npz(raw, trust=trust, remote=remote)
            except Exception:
                return pd.read_parquet(io.BytesIO(raw))
        if _complete_pickle_stream(raw):
            # protocol-0 (ASCII) pickles carry no magic prefix (e.g.
            # hyp.save(..., protocol=0) to an arbitrary extension)
            return _unpickle_bytes(raw, trust=trust, remote=remote)
        raise HypertoolsIOError(
            f'cannot load {label!r}: unsupported file extension {ext!r}, '
            "and the content doesn't match a known binary format. "
            f"Supported extensions: {', '.join(_SUPPORTED_EXTENSIONS)}. "
            'If this is a delimited text file, rename it with a '
            '.csv/.tsv/.txt extension.')

    # no extension: sniff the content
    if raw[:6] == b'\x93NUMPY':
        return _npy_load(raw, allow_pickle)
    if raw[:1] == b'\x80':
        return _unpickle_bytes(raw, trust=trust, remote=remote)
    if raw[:2] == b'PK':
        try:
            return _unpack_npz(raw, trust=trust, remote=remote)
        except Exception:
            return pd.read_parquet(io.BytesIO(raw))
    if _complete_pickle_stream(raw):
        # protocol-0 (ASCII) pickles carry no magic prefix and DO decode
        # as UTF-8, so they must be sniffed BEFORE text parsing or they
        # come back silently CSV-parsed into a garbage DataFrame
        # (release-1.0 audit: re-review of X2-error-quality-001)
        try:
            return _unpickle_bytes(raw, trust=trust, remote=remote)
        except HypertoolsTrustError:
            # the remote-pickle refusal must surface as itself (with its
            # trust=True remedy), not be mislabeled "corrupted"
            raise
        except Exception as e:
            raise HypertoolsIOError(
                f'{label!r} contains a pickle stream that could not be '
                f'unpickled ({type(e).__name__}: {e}); the file may be '
                'corrupted, or may need a package that is not installed.'
            ) from e
    try:
        raw.decode('utf-8')
    except UnicodeDecodeError:
        # last resort: pickle protocols < 2 have no magic prefix
        try:
            return _unpickle_bytes(raw, trust=trust, remote=remote)
        except Exception as e:
            raise HypertoolsIOError(
                f'could not parse {label!r}: the content is not a numpy '
                'array, pickle, zip/parquet payload, or UTF-8 text.'
            ) from e
    return _read_delimited_text(raw, label)


def _read_delimited_text(raw, label, sep=None):
    """Parse delimited-text bytes (.csv/.txt/extensionless text payloads).

    Strategy (QC 2026-07, F19-load-external-001): parse with the comma
    default first; only when that yields a single column, consult
    ``csv.Sniffer`` restricted to common delimiter characters (never
    letters or digits), and accept the sniffed delimiter only when it
    appears in essentially every line and produces more columns. pandas'
    unrestricted ``sep=None`` sniffing used to pick an in-word letter as
    the delimiter for single-column files, silently corrupting them; a
    genuinely single-column file now round-trips exactly, while
    semicolon-/tab-/pipe-/whitespace-delimited files still parse.

    ``sep`` forces a delimiter (used for .tsv) and skips the fallback.
    """
    import csv as _csv

    if not raw.strip():
        raise HypertoolsIOError(f'{label} is empty -- nothing to load.')
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as e:
        raise HypertoolsIOError(
            f'{label} is not UTF-8 text, so it could not be parsed as a '
            'delimited (CSV-style) file -- is it binary or compressed?'
        ) from e

    try:
        if sep is not None:
            return pd.read_csv(io.StringIO(text), sep=sep)
        frame = pd.read_csv(io.StringIO(text), sep=',')
    except HypertoolsIOError:
        raise
    except Exception as e:
        raise HypertoolsIOError(
            f'could not parse {label} as delimited text '
            f'({type(e).__name__}: {e}).') from e
    if frame.shape[1] > 1:
        return frame

    # single column under the comma default: validated sniff fallback
    sample = text[:8192]
    try:
        dialect = _csv.Sniffer().sniff(sample, delimiters=',;\t| ')
    except _csv.Error:
        return frame
    if dialect.delimiter == ',':
        return frame
    # the sniffed delimiter must look structural: present in (essentially)
    # every non-empty line, not e.g. a space inside one quoted value
    lines = [ln for ln in sample.splitlines() if ln.strip()]
    hits = sum(dialect.delimiter in ln for ln in lines)
    if hits < max(1, int(0.9 * len(lines))):
        return frame
    try:
        sniffed = pd.read_csv(io.StringIO(text), sep=dialect.delimiter,
                              engine='python')
    except Exception:
        return frame
    return sniffed if sniffed.shape[1] > frame.shape[1] else frame


def _npy_load(raw, allow_pickle):
    """np.load wrapper that turns the "Object arrays cannot be loaded
    when allow_pickle=False" ValueError into a HypertoolsTrustError when
    ``allow_pickle`` was forced False by the remote-trust policy (never
    true for local files or trust=True), so that load_source()'s branch
    guards can distinguish it from an unrelated parse-failure
    ValueError."""
    try:
        return np.load(io.BytesIO(raw), allow_pickle=allow_pickle)
    except ValueError as e:
        if not allow_pickle:
            raise HypertoolsTrustError(
                f'{e}; pass trust=True to hypertools.load() once you '
                'have verified the source, to load pickled/object-array '
                'data from a remote source') from e
        raise


def _unpack_npz(raw, trust=False, remote=False):
    allow_pickle = trust or not remote
    try:
        z = np.load(io.BytesIO(raw), allow_pickle=allow_pickle)
        arrays = [z[k] for k in z.files]
    except ValueError as e:
        if not allow_pickle:
            raise HypertoolsTrustError(
                f'{e}; pass trust=True to hypertools.load() once you '
                'have verified the source, to load pickled/object-array '
                'data from a remote source') from e
        raise
    return arrays[0] if len(arrays) == 1 else arrays


def _unpack_mat(raw):
    from scipy.io import loadmat
    data = {k: v for k, v in loadmat(io.BytesIO(raw)).items()
            if not k.startswith('__')}
    if len(data) == 1:
        return next(iter(data.values()))
    return data


def _read_xls(raw):
    """Legacy .xls (OLE binary format) via pandas' xlrd engine, with a
    friendlier ImportError than pandas' own when xlrd isn't installed."""
    try:
        return pd.read_excel(io.BytesIO(raw), engine='xlrd')
    except ImportError as e:
        raise ImportError(
            'xlrd is required to load legacy .xls files; install it with '
            'pip install xlrd'
        ) from e


def _unpickle_bytes(raw, trust=False, remote=False):
    """pickle -> pandas unpickler -> dill, mirroring the tolerant chain
    used for the built-in example datasets.

    Remote payloads (``remote=True``) are REFUSED unless ``trust=True``:
    unpickling can execute arbitrary code, so a warning is not a sufficient
    security boundary (2026-07 release review, blocker #1). This is the
    single chokepoint for every remote-pickle path -- extension-based
    (.pkl/.pickle/.geo/.p), magic-byte-sniffed, and extensionless
    protocol-0 -- so refusing here covers all of them."""
    if remote and not trust:
        raise HypertoolsTrustError(_PICKLE_TRUST_REFUSAL)
    import pickle
    try:
        return pickle.loads(raw)
    except Exception:
        pass
    try:
        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            f.write(raw)
            f.flush()
            return pd.read_pickle(f.name)
    except Exception:
        pass
    import dill
    return dill.loads(raw)
