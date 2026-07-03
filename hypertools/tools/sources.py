#!/usr/bin/env python
"""Universal data-source resolution for hyp.load / DataGeometry.

A string may name (tried in this order):

1. a built-in example dataset (``EXAMPLE_DATA`` in tools.load)
2. a path to a local file
3. a Hugging Face dataset (including streaming datasets)
4. a Google Drive URL or bare file ID
5. a Dropbox URL or shared-link path
6. any other URL (with or without an ``https://`` scheme)

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
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .._shared.exceptions import HypertoolsIOError

_DRIVE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{25,}$')
_DOMAIN_RE = re.compile(r'^[\w-]+(\.[\w-]+)+(/\S*)?$')
_HF_ID_RE = re.compile(r'^[\w.-]+/[\w.-]+$')
_UA = {'User-Agent': 'hypertools'}


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


def load_source(source, split=None, streaming=False):
    """Resolve one non-builtin string source (steps 2-6 of the chain).

    Returns the loaded dataset (DataFrame, array, list, dict, or a
    Hugging Face [Iterable]Dataset). Raises HypertoolsIOError listing
    every attempted interpretation when nothing works.
    """
    attempts = []

    # 2. local file
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

    # 3. Hugging Face dataset (skip for obvious URLs)
    if not is_url_like and _HF_ID_RE.match(source):
        try:
            return _load_hf(source, split=split, streaming=streaming)
        except ImportError:
            raise
        except Exception as e:  # dataset not found, gated, etc.
            attempts.append(f'Hugging Face dataset: {type(e).__name__}: '
                            f'{str(e).splitlines()[0][:120]}')

    # 4. Google Drive URL or bare ID
    drive_id = _extract_drive_id(source)
    if drive_id is not None:
        url = f'https://drive.google.com/uc?export=download&id={drive_id}'
        try:
            raw, name_hint = _fetch_bytes(url)
            return _parse_payload(raw, name_hint or source)
        except Exception as e:
            attempts.append(f'Google Drive ({drive_id}): '
                            f'{type(e).__name__}: {e}')

    # 5. Dropbox URL or shared-link path
    dropbox_url = _normalize_dropbox(source)
    if dropbox_url is not None:
        try:
            raw, name_hint = _fetch_bytes(dropbox_url)
            return _parse_payload(raw, name_hint or source)
        except Exception as e:
            attempts.append(f'Dropbox: {type(e).__name__}: {e}')

    # 6. any URL, with or without a scheme
    url = None
    if source.startswith(('http://', 'https://')):
        url = source
    elif _DOMAIN_RE.match(source):
        url = 'https://' + source
    if url is not None:
        try:
            raw, name_hint = _fetch_bytes(url)
            return _parse_payload(raw, name_hint or source)
        except Exception as e:
            attempts.append(f'URL ({url}): {type(e).__name__}: {e}')

    tried = '\n  - '.join(attempts) if attempts else 'no interpretation ' \
        'matched (not a file, URL, Drive/Dropbox link, or dataset id)'
    raise HypertoolsIOError(
        f'could not load {source!r}. Tried, in order:\n  - {tried}')


def load_local_file(path):
    """Load a local data file by extension (with content sniffing as the
    fallback). Supports pickle/.geo, .npy/.npz, .csv/.tsv/.txt, .json,
    .parquet, and .mat."""
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


def _fetch_bytes(url, timeout=60):
    """Download url -> (bytes, filename_hint). Raises on HTTP errors and on
    HTML interstitials (e.g. Google Drive rate-limit pages)."""
    resp = requests.get(url, headers=_UA, timeout=timeout,
                        allow_redirects=True)
    resp.raise_for_status()
    raw = resp.content
    if not raw:
        raise HypertoolsIOError(f'empty response from {url}')
    ctype = resp.headers.get('Content-Type', '')
    if raw[:1] == b'<' and ('html' in ctype or b'<html' in raw[:512].lower()):
        raise HypertoolsIOError(
            f'{url} returned an HTML page instead of data (rate limit, '
            'permission page, or a link that needs a direct-download form)')
    name_hint = None
    dispo = resp.headers.get('Content-Disposition', '')
    m = re.search(r'filename="?([^";]+)"?', dispo)
    if m:
        name_hint = m.group(1)
    else:
        tail = url.split('?')[0].rstrip('/').rsplit('/', 1)[-1]
        if '.' in tail:
            name_hint = tail
    return raw, name_hint


def _parse_payload(raw, name_hint=''):
    """Parse downloaded/read bytes into a dataset, by filename extension
    first and content sniffing second."""
    ext = Path(str(name_hint)).suffix.lower()

    if ext == '.npy':
        return np.load(io.BytesIO(raw), allow_pickle=True)
    if ext == '.npz':
        return _unpack_npz(raw)
    if ext in ('.csv', '.tsv', '.txt'):
        sep = '\t' if ext == '.tsv' else None
        return pd.read_csv(io.BytesIO(raw), sep=sep, engine='python')
    if ext == '.json':
        return pd.read_json(io.BytesIO(raw))
    if ext == '.parquet':
        return pd.read_parquet(io.BytesIO(raw))
    if ext == '.mat':
        return _unpack_mat(raw)
    if ext in ('.pkl', '.pickle', '.geo', '.p'):
        return _unpickle_bytes(raw)

    # no (useful) extension: sniff the content
    if raw[:6] == b'\x93NUMPY':
        return np.load(io.BytesIO(raw), allow_pickle=True)
    if raw[:1] == b'\x80':
        return _unpickle_bytes(raw)
    if raw[:2] == b'PK':
        try:
            return _unpack_npz(raw)
        except Exception:
            return pd.read_parquet(io.BytesIO(raw))
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        # last resort: pickle protocols < 2 have no magic prefix
        return _unpickle_bytes(raw)
    return pd.read_csv(io.StringIO(text), sep=None, engine='python')


def _unpack_npz(raw):
    z = np.load(io.BytesIO(raw), allow_pickle=True)
    arrays = [z[k] for k in z.files]
    return arrays[0] if len(arrays) == 1 else arrays


def _unpack_mat(raw):
    from scipy.io import loadmat
    data = {k: v for k, v in loadmat(io.BytesIO(raw)).items()
            if not k.startswith('__')}
    if len(data) == 1:
        return next(iter(data.values()))
    return data


def _unpickle_bytes(raw):
    """pickle -> pandas unpickler -> dill, mirroring the tolerant chain
    used for the built-in example datasets."""
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
