#!/usr/bin/env python
"""Universal data-source resolution for hyp.load / DataGeometry.

A string may name (tried in this order):

1. a built-in example dataset (``EXAMPLE_DATA`` in tools.load)
2. a scikit-learn bundled dataset name (``sklearn_dataset``)
3. a seaborn dataset name (``seaborn_dataset``)
4. a path to a local file
5. a Hugging Face dataset (including streaming datasets)
6. a Google Sheets URL
7. a Google Drive URL or bare file ID
8. a Dropbox URL or shared-link path
9. any other URL (with or without an ``https://`` scheme)

Lists of strings resolve element-wise to a list of datasets.

.. warning::
    Pickle-format payloads (``.pkl``/``.geo``/pickled arrays) can execute
    arbitrary code when loaded, exactly like ``pandas.read_pickle``. Only
    load pickled data from sources you trust. Non-pickle formats
    (csv/tsv/json/npy without object arrays/parquet/mat) do not have this
    risk.
"""

import io
import re
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .._shared.exceptions import HypertoolsIOError

_DRIVE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{25,}$')
_DOMAIN_RE = re.compile(r'^[\w-]+(\.[\w-]+)+(/\S*)?$')
_HF_ID_RE = re.compile(r'^[\w.-]+/[\w.-]+$')
_UA = {'User-Agent': 'hypertools'}

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

_PICKLE_TRUST_WARNING = (
    'unpickling data from a remote source can execute arbitrary code; '
    'pass trust=True to hypertools.load() once you have verified the '
    'source, to silence this warning')


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


def is_loadable_string(s):
    """Cheap (no-network) check: could this string plausibly name a data
    source? Used to decide whether to route strings through load() rather
    than treating them as raw text to embed."""
    from .load import EXAMPLE_DATA
    if not isinstance(s, str) or not s.strip() or any(c.isspace()
                                                      for c in s.strip()):
        return False
    if s in EXAMPLE_DATA:
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
                extra_attempts=None):
    """Resolve one non-builtin string source (steps 4-9 of the chain --
    scikit-learn/seaborn dataset names, steps 2-3, are tried by
    :func:`hypertools.load` before this function is called).

    Returns the loaded dataset (DataFrame, array, list, dict, or a
    Hugging Face [Iterable]Dataset). Raises HypertoolsIOError listing
    every attempted interpretation when nothing works.

    ``trust`` is threaded from :func:`hypertools.load`: it silences the
    remote-pickle security warning and re-enables ``allow_pickle`` for
    remote .npy/.npz payloads (see ``_parse_payload``).

    ``extra_attempts`` optionally seeds the "tried, in order" list with
    descriptions of resolvers already attempted by the caller (e.g. the
    scikit-learn/seaborn dataset-name lookups), so the final error message
    reflects the whole chain.
    """
    attempts = list(extra_attempts) if extra_attempts else []

    # 4. local file
    path = Path(source).expanduser()
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if is_file:
        return load_local_file(path)
    attempts.append(f'local file: not found at {path}')

    is_url_like = source.startswith(('http://', 'https://')) \
        or 'drive.google.com' in source or 'docs.google.com' in source \
        or 'dropbox.com' in source

    # 5. Hugging Face dataset (skip for obvious URLs)
    if not is_url_like and _HF_ID_RE.match(source):
        try:
            return _load_hf(source, split=split, streaming=streaming)
        except ImportError:
            raise
        except Exception as e:  # dataset not found, gated, etc.
            attempts.append(f'Hugging Face dataset: {type(e).__name__}: '
                            f'{str(e).splitlines()[0][:120]}')

    # 6. Google Sheets URL -> CSV export (checked before generic Drive id
    # extraction, since a Sheets URL also matches the '/d/<id>' pattern)
    sheet_url = _normalize_google_sheet(source)
    if sheet_url is not None:
        try:
            raw, name_hint = _fetch_bytes(sheet_url)
            return _parse_payload(raw, name_hint or 'sheet.csv',
                                  trust=trust, remote=True)
        except HypertoolsTrustError:
            raise
        except Exception as e:
            attempts.append(f'Google Sheets: {type(e).__name__}: {e}')

    # 7. Google Drive URL or bare ID
    drive_id = _extract_drive_id(source)
    if drive_id is not None:
        url = f'https://drive.google.com/uc?export=download&id={drive_id}'
        try:
            raw, name_hint = _fetch_bytes(url)
            return _parse_payload(raw, name_hint or source,
                                  trust=trust, remote=True)
        except HypertoolsTrustError:
            raise
        except Exception as e:
            attempts.append(f'Google Drive ({drive_id}): '
                            f'{type(e).__name__}: {e}')

    # 8. Dropbox URL or shared-link path
    dropbox_url = _normalize_dropbox(source)
    if dropbox_url is not None:
        try:
            raw, name_hint = _fetch_bytes(dropbox_url)
            return _parse_payload(raw, name_hint or source,
                                  trust=trust, remote=True)
        except HypertoolsTrustError:
            raise
        except Exception as e:
            attempts.append(f'Dropbox: {type(e).__name__}: {e}')

    # 9. any URL, with or without a scheme
    url = None
    if source.startswith(('http://', 'https://')):
        url = source
    elif _DOMAIN_RE.match(source):
        url = 'https://' + source
    if url is not None:
        try:
            raw, name_hint = _fetch_bytes(url)
            return _parse_payload(raw, name_hint or source,
                                  trust=trust, remote=True)
        except HypertoolsTrustError:
            raise
        except Exception as e:
            attempts.append(f'URL ({url}): {type(e).__name__}: {e}')

    tried = '\n  - '.join(attempts) if attempts else 'no interpretation ' \
        'matched (not a file, URL, Drive/Dropbox link, or dataset id)'
    raise HypertoolsIOError(
        f'could not load {source!r}. Tried, in order:\n  - {tried}')


def load_local_file(path):
    """Load a local data file by extension (with content sniffing as the
    fallback). Supports pickle/.geo, .npy/.npz, .csv/.tsv/.txt, .json,
    .parquet, .mat, and .xlsx/.xls."""
    path = Path(path)
    return _parse_payload(path.read_bytes(), path.name)


def _load_hf(name, split=None, streaming=False):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            f'{name!r} looks like a Hugging Face dataset id, but the '
            "`datasets` package is not installed. Install it with "
            "`pip install datasets`.") from e
    ds = load_dataset(name, split=split, streaming=streaming)
    if split is None and hasattr(ds, 'keys'):  # (Iterable)DatasetDict
        keys = list(ds.keys())
        pick = 'train' if 'train' in keys else keys[0]
        ds = ds[pick]
    if streaming:
        return ds  # IterableDataset: stream it straight into hyp.plot
    return ds.to_pandas()


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


def _name_hint(resp, url):
    dispo = resp.headers.get('Content-Disposition', '')
    m = re.search(r'filename="?([^";]+)"?', dispo)
    if m:
        return m.group(1)
    tail = url.split('?')[0].rstrip('/').rsplit('/', 1)[-1]
    return tail if '.' in tail else None


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


def _fetch_bytes(url, timeout=60):
    """Download url -> (bytes, filename_hint). Automatically follows the
    Google Drive large-file virus-scan interstitial (a confirm form served
    in place of the file); raises on HTTP errors and on any other HTML
    interstitial (e.g. rate-limit/permission pages)."""
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
        return raw, _name_hint(resp, action_url)

    return raw, _name_hint(resp, url)


def _parse_payload(raw, name_hint='', trust=False, remote=False):
    """Parse downloaded/read bytes into a dataset, by filename extension
    first and content sniffing second.

    ``remote`` marks payloads fetched over the network (as opposed to a
    local file): unpickling a remote payload without ``trust=True`` emits
    a ``UserWarning``, and remote .npy/.npz use ``allow_pickle=False``
    unless ``trust=True``. Local files are never subject to this policy.
    """
    ext = Path(str(name_hint)).suffix.lower()
    allow_pickle = trust or not remote

    if ext == '.npy':
        return _npy_load(raw, allow_pickle)
    if ext == '.npz':
        return _unpack_npz(raw, trust=trust, remote=remote)
    if ext in ('.csv', '.tsv', '.txt'):
        sep = '\t' if ext == '.tsv' else None
        return pd.read_csv(io.BytesIO(raw), sep=sep, engine='python')
    if ext == '.json':
        return pd.read_json(io.BytesIO(raw))
    if ext == '.parquet':
        return pd.read_parquet(io.BytesIO(raw))
    if ext == '.mat':
        return _unpack_mat(raw)
    if ext == '.xlsx':
        return pd.read_excel(io.BytesIO(raw))
    if ext == '.xls':
        return _read_xls(raw)
    if ext in ('.pkl', '.pickle', '.geo', '.p'):
        return _unpickle_bytes(raw, trust=trust, remote=remote)

    # no (useful) extension: sniff the content
    if raw[:6] == b'\x93NUMPY':
        return _npy_load(raw, allow_pickle)
    if raw[:1] == b'\x80':
        return _unpickle_bytes(raw, trust=trust, remote=remote)
    if raw[:2] == b'PK':
        try:
            return _unpack_npz(raw, trust=trust, remote=remote)
        except Exception:
            return pd.read_parquet(io.BytesIO(raw))
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        # last resort: pickle protocols < 2 have no magic prefix
        return _unpickle_bytes(raw, trust=trust, remote=remote)
    return pd.read_csv(io.StringIO(text), sep=None, engine='python')


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

    Remote payloads (``remote=True``) emit a ``UserWarning`` unless
    ``trust=True``: unpickling can execute arbitrary code."""
    if remote and not trust:
        warnings.warn(_PICKLE_TRUST_WARNING, UserWarning, stacklevel=3)
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
