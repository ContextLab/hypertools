# -*- coding: utf-8 -*-
"""Remote-pickle trust boundary (2026-07 release review, blocker #1).

`trust=False` (the default) must be a real SECURITY BOUNDARY, not a
warning: a remote pickle is REFUSED before any deserialization, across all
three ways a pickle can arrive -- by extension (.pkl/.geo/...), by
content-sniffed magic byte (no/unknown extension, protocol >= 2), and as an
extensionless protocol-0 (ASCII) pickle. `trust=True` opts in. Local files
and built-in datasets are unaffected. Real loopback HTTP server -- no mocks;
the "malicious" payload is an object that records execution on unpickle.
"""
import http.server
import pickle
import threading
from functools import partial

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools.io.sources import HypertoolsTrustError


# a payload whose __reduce__ runs code on unpickle -- if hypertools ever
# deserializes it without trust, this module-level flag flips to True.
_EXECUTED = {'ran': False}


class _Evil:
    def __reduce__(self):
        return (_mark_executed, ())


def _mark_executed():
    _EXECUTED['ran'] = True
    return 'pwned'


@pytest.fixture
def http_dir(tmp_path):
    handler = partial(http.server.SimpleHTTPRequestHandler,
                      directory=str(tmp_path))
    server = http.server.HTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield tmp_path, f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()


def setup_function(_):
    _EXECUTED['ran'] = False


def _write(tmp_path, name, obj, protocol=pickle.DEFAULT_PROTOCOL):
    p = tmp_path / name
    p.write_bytes(pickle.dumps(obj, protocol=protocol))
    return p.name


# --- refusal across all three remote-pickle entry paths ----------------

def test_remote_pickle_by_extension_refused(http_dir):
    tmp_path, base = http_dir
    _write(tmp_path, 'd.pkl', pd.DataFrame({'a': [1, 2]}))
    with pytest.raises(HypertoolsTrustError, match='refusing to unpickle'):
        hyp.load(base + '/d.pkl')


def test_remote_pickle_content_sniffed_refused(http_dir):
    # a real pickle behind a NON-pickle extension: hypertools content-sniffs
    # the magic byte -- and must still refuse it, not sniff-then-execute
    tmp_path, base = http_dir
    _write(tmp_path, 'd.bin', pd.DataFrame({'a': [1, 2]}))
    with pytest.raises(HypertoolsTrustError, match='refusing to unpickle'):
        hyp.load(base + '/d.bin')


def test_remote_extensionless_protocol0_pickle_refused(http_dir):
    # protocol-0 pickles are plain ASCII with no magic prefix
    tmp_path, base = http_dir
    _write(tmp_path, 'noext', pd.DataFrame({'a': [1, 2]}), protocol=0)
    with pytest.raises(HypertoolsTrustError, match='refusing to unpickle'):
        hyp.load(base + '/noext')


def test_malicious_remote_pickle_never_executes(http_dir):
    # the crux: refusal happens BEFORE deserialization, so __reduce__ code
    # never runs
    tmp_path, base = http_dir
    _write(tmp_path, 'evil.pkl', _Evil())
    with pytest.raises(HypertoolsTrustError):
        hyp.load(base + '/evil.pkl')
    assert _EXECUTED['ran'] is False, 'remote pickle code executed under trust=False!'


# --- trust=True opts in -------------------------------------------------

def test_trust_true_allows_each_form(http_dir):
    tmp_path, base = http_dir
    _write(tmp_path, 'd.pkl', pd.DataFrame({'a': [1, 2]}))
    _write(tmp_path, 'd.bin', pd.DataFrame({'a': [3, 4]}))
    _write(tmp_path, 'noext', pd.DataFrame({'a': [5, 6]}), protocol=0)
    assert hyp.load(base + '/d.pkl', trust=True).shape == (2, 1)
    assert hyp.load(base + '/d.bin', trust=True).shape == (2, 1)
    assert hyp.load(base + '/noext', trust=True).shape == (2, 1)


# --- local files & non-executable formats are unaffected ---------------

def test_local_pickle_needs_no_trust(tmp_path):
    p = tmp_path / 'local.pkl'
    pd.to_pickle(pd.DataFrame({'a': [1, 2, 3]}), p)
    assert hyp.load(str(p)).shape == (3, 1)  # no trust= required


def test_remote_numeric_npz_needs_no_trust(http_dir):
    # a non-executable remote format loads without trust
    tmp_path, base = http_dir
    np.savez(tmp_path / 'nums.npz', x=np.arange(6.0).reshape(3, 2))
    out = hyp.load(base + '/nums.npz', trust=False)
    # single-array npz unwraps to the array itself (like _unpack_mat)
    arr = out['x'] if hasattr(out, 'keys') else out
    assert np.asarray(arr).shape == (3, 2)


def test_builtin_dataset_by_name_needs_no_trust():
    # built-in datasets load by NAME through the integrity-checked cache
    # path and never require trust= (they are not "remote user pickles")
    data = hyp.load('spiral')  # cached from the session; no trust needed
    assert isinstance(data, list) and np.asarray(data[0]).shape[1] == 3
