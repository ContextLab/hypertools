# -*- coding: utf-8 -*-
"""Regression tests for the 2026-07 release audit findings on hyp.load /
hypertools.io.sources (units F18-load-hosted, F19-load-external, and
cross-unit X2-error-quality-001). All tests use real files, real pickles,
and (where unavoidable) real network calls -- no mocks."""

import functools
import gzip
import http.server
import pickle
import threading
import warnings

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp
from hypertools._shared.exceptions import HypertoolsIOError
from hypertools.io.load import EXAMPLE_DATA


# --------------------------------------------------------------- F18-001
# load('sotus') must return the real State of the Union speeches (the
# legacy Google Drive id was duplicated with nips_model and served a
# pickled sklearn Pipeline instead of the documented speeches).

def test_load_sotus_returns_the_29_speeches(capsys):
    data = hyp.load('sotus')
    assert isinstance(data, list), f'expected list, got {type(data)}'
    assert len(data) == 29
    assert all(isinstance(doc, str) for doc in data)
    assert data[0].startswith('Mr. Speaker'), data[0][:80]
    # datawrangler's "loading corpus: sotus...done!" chatter is suppressed
    captured = capsys.readouterr()
    assert 'loading corpus' not in captured.out


# --------------------------------------------------------------- F18-002
# hosted *_model pipelines were pickled under sklearn 1.0.2; under the
# shipped sklearn, repr()/get_params() crashed with AttributeError
# ('Pipeline' object has no attribute 'transform_input') and loading
# spewed InconsistentVersionWarning.

def test_wiki_model_supports_standard_sklearn_surface():
    from sklearn.exceptions import InconsistentVersionWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = hyp.load('wiki_model')
    assert not [w for w in caught
                if issubclass(w.category, InconsistentVersionWarning)], \
        'InconsistentVersionWarning leaked from a hypertools-hosted model'
    repr(model)                    # crashed before the fix
    params = model.get_params()    # crashed before the fix
    assert 'steps' in params
    out = model.transform(['the quick brown fox jumps over the lazy dog'])
    assert out.shape == (1, 50)


# ----------------------------------------------------- F18-003 / F18-004
# load() docstring dataset table: every registered hosted name documented,
# types/typos corrected.

def test_load_docstring_documents_every_hosted_dataset():
    doc = hyp.load.__doc__
    missing = [name for name in EXAMPLE_DATA if name not in doc]
    assert missing == [], f'hosted datasets missing from docstring: {missing}'


def test_load_docstring_accuracy_fixes():
    doc = hyp.load.__doc__
    assert 'mushroomm' not in doc                    # typo
    assert 'never collide' not in doc                # false claim (F19-009)
    # spiral is a LIST of two arrays, not a bare numpy array
    assert 'list of two' in doc
    # mushrooms is returned as a DataFrame
    assert 'DataFrame' in doc


# ----------------------------------------------------- F18-005 / F19-010
# non-string dataset arguments must raise a hypertools-branded TypeError,
# not raw os.path / dict-hash internals.

@pytest.mark.parametrize('bad', [123, None, 3.14, b'weights', {'a': 1}])
def test_load_non_string_dataset_raises_friendly_typeerror(bad):
    with pytest.raises(TypeError, match='dataset must be'):
        hyp.load(bad)


# --------------------------------------------------------------- F18-006
# ~/hypertools_data existing as a regular FILE must produce a helpful
# HypertoolsIOError, not a raw FileExistsError.

def test_cache_dir_blocked_by_regular_file(tmp_path, monkeypatch):
    import importlib
    load_mod = importlib.import_module('hypertools.io.load')
    fake_cache = tmp_path / 'hypertools_data'
    fake_cache.write_text('accidentally a file')
    monkeypatch.setattr(load_mod, 'DATA_DIR', fake_cache)
    with pytest.raises(HypertoolsIOError, match='not a directory'):
        load_mod.load('spiral')


# --------------------------------------------------------------- F18-007
# the unknown-name digest must mention the built-in example dataset names
# and offer a near-miss suggestion.

def test_unknown_name_digest_lists_builtins_and_suggests():
    with pytest.raises(HypertoolsIOError) as excinfo:
        hyp.load('Weights')
    message = str(excinfo.value)
    assert 'built-in example dataset' in message
    assert "Did you mean 'weights'?" in message


# --------------------------------------------------------------- F19-001
# single-column .csv/.txt (and extensionless text) files must round-trip
# exactly -- pandas' sep=None sniffing used to split words on an in-word
# letter and silently return garbage.

def test_single_column_csv_roundtrips_exactly(tmp_path):
    path = tmp_path / 'single_col.csv'
    pd.DataFrame({'solo': [1, 2, 3]}).to_csv(path, index=False)
    out = hyp.load(str(path))
    assert out.shape == (3, 1)
    assert out.columns.tolist() == ['solo']
    assert out['solo'].tolist() == [1, 2, 3]


def test_single_column_txt_roundtrips_exactly(tmp_path):
    path = tmp_path / 'single_col.txt'
    path.write_text('solo\n1\n2\n3\n')
    out = hyp.load(str(path))
    assert out.shape == (3, 1)
    assert out.columns.tolist() == ['solo']


def test_single_column_extensionless_file_roundtrips_exactly(tmp_path):
    path = tmp_path / 'single_col_noext'
    path.write_text('solo\n1\n2\n3\n')
    out = hyp.load(str(path))
    assert out.shape == (3, 1)
    assert out.columns.tolist() == ['solo']


def test_semicolon_csv_still_sniffed(tmp_path):
    path = tmp_path / 'semi.csv'
    path.write_text('a;b\n1;2\n3;4\n')
    out = hyp.load(str(path))
    assert out.shape == (2, 2)
    assert out.columns.tolist() == ['a', 'b']


def test_whitespace_delimited_txt_still_parses(tmp_path):
    arr = np.arange(12, dtype=float).reshape(4, 3)
    path = tmp_path / 'matrix.txt'
    np.savetxt(path, arr)
    out = hyp.load(str(path))
    assert np.asarray(out).shape[1] == 3


def test_quoted_single_column_not_split_on_inner_spaces(tmp_path):
    path = tmp_path / 'names.csv'
    pd.DataFrame({'name': ['John Smith', 'Ada Lovelace']}).to_csv(
        path, index=False)
    out = hyp.load(str(path))
    assert out.shape == (2, 1)
    assert out['name'].tolist() == ['John Smith', 'Ada Lovelace']


# --------------------------------------------------------------- F19-005
# unparseable local files must raise HypertoolsIOError naming the file,
# not raw _csv.Error / UnpicklingError; .csv.gz is supported transparently.

def test_empty_csv_raises_friendly_error(tmp_path):
    path = tmp_path / 'empty.csv'
    path.touch()
    with pytest.raises(HypertoolsIOError, match='empty'):
        hyp.load(str(path))


def test_empty_pickle_raises_friendly_error(tmp_path):
    path = tmp_path / 'empty.pkl'
    path.touch()
    with pytest.raises(HypertoolsIOError, match='empty'):
        hyp.load(str(path))


def test_gzipped_csv_loads_transparently(tmp_path):
    path = tmp_path / 't.csv.gz'
    with gzip.open(path, 'wt') as f:
        f.write('a,b\n1,2\n3,4\n')
    out = hyp.load(str(path))
    assert out.shape == (2, 2)
    assert out.columns.tolist() == ['a', 'b']


def test_extensionless_binary_garbage_raises_friendly_error(tmp_path):
    path = tmp_path / 'garbagefile'
    path.write_bytes(bytes(range(2, 256)))  # not pickle/npy/zip/utf-8
    with pytest.raises(HypertoolsIOError, match='could not parse'):
        hyp.load(str(path))


# -------------------------------------------------- X2-error-quality-001
# a real file with an unsupported extension must NOT be silently parsed as
# delimiter-sniffed CSV garbage -- unless its content unambiguously matches
# a known binary format (pickle/npy/zip), which keeps hyp.save round-trips
# working for arbitrary extensions.

def test_unsupported_extension_text_raises_clear_error(tmp_path):
    path = tmp_path / 'junk.xyz'
    path.write_text('hello')
    with pytest.raises(HypertoolsIOError, match=r"\.xyz"):
        hyp.load(str(path))


def test_unsupported_extension_error_lists_supported_formats(tmp_path):
    path = tmp_path / 'export.dat'
    path.write_text('instrument output, not a table')
    with pytest.raises(HypertoolsIOError, match=r'\.csv'):
        hyp.load(str(path))


def test_unsupported_extension_binary_garbage_raises(tmp_path):
    path = tmp_path / 'binary.garbage'
    path.write_bytes(bytes(range(2, 256)))
    with pytest.raises(HypertoolsIOError, match=r'\.garbage'):
        hyp.load(str(path))


def test_unsupported_extension_pickle_content_still_roundtrips(tmp_path):
    # hyp.save writes pickles regardless of extension for unknown formats;
    # load must keep recognizing them by their magic bytes
    path = tmp_path / 'data.xyz'
    arr = np.arange(6, dtype=float).reshape(3, 2)
    with open(path, 'wb') as f:
        pickle.dump(arr, f)
    np.testing.assert_allclose(hyp.load(str(path)), arr)


# --------------------------------------------------------------- F19-006
# a mistyped bare filename must not trigger a real DNS lookup for the
# filename.

def test_bare_missing_filename_skips_url_guess():
    with pytest.raises(HypertoolsIOError) as excinfo:
        hyp.load('f19_missing_xyz_qqq.csv')
    message = str(excinfo.value)
    assert 'looks like a (missing) local file name' in message
    assert 'ConnectionError' not in message
    assert 'NameResolutionError' not in message


# --------------------------------------------------------------- F19-013
# failure-digest rough edges.

def test_directory_input_digest_says_directory(tmp_path):
    with pytest.raises(HypertoolsIOError, match='is a directory'):
        hyp.load(str(tmp_path))


def test_hostport_input_digest_suggests_scheme():
    with pytest.raises(HypertoolsIOError) as excinfo:
        hyp.load('127.0.0.1:9/t.csv')
    assert 'http://' in str(excinfo.value)


def test_url_input_digest_has_no_mangled_local_file_line():
    url = ('https://raw.githubusercontent.com/mwaskom/seaborn-data/'
           'master/zz-definitely-no-such-file-xyz.csv')
    with pytest.raises(HypertoolsIOError) as excinfo:
        hyp.load(url)
    assert 'local file: not found at https:/' not in str(excinfo.value)


# --------------------------------------------------------------- F19-003
# the anonymous 538 CSV download must retry transient 5xx gateway errors
# like the listing call does. Real loopback HTTP server, no mocks.

def test_538_anonymous_csv_fetch_retries_transient_502(monkeypatch):
    import hypertools.io.sources as sources

    hits = {'n': 0}

    class FlakyHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits['n'] += 1
            if hits['n'] <= 2:
                self.send_response(502)
                self.end_headers()
                return
            body = b'x,y\n1,2\n'
            self.send_response(200)
            self.send_header('Content-Type', 'text/csv')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), FlakyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f'http://127.0.0.1:{server.server_address[1]}'
        monkeypatch.setattr(sources, '_538_RAW', base)
        # make the retry loop fast for the test (same tunables the retry
        # helper exposes for exactly this purpose)
        orig = sources._github_get_with_retry
        monkeypatch.setattr(
            sources, '_github_get_with_retry',
            functools.partial(orig, backoff=0.05))
        monkeypatch.delenv('GITHUB_TOKEN', raising=False)
        monkeypatch.delenv('GH_TOKEN', raising=False)
        raw = sources._fetch_538_csv('someslug', 'file.csv')
        assert raw == b'x,y\n1,2\n'
        assert hits['n'] == 3, 'expected 2 x 502 then success on 3rd hit'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --------------------------------------------------------------- F19-011
# a stale/invalid ambient GITHUB_TOKEN must be named as the likely cause
# of a 401 on 538 loads (real GitHub API call; anonymous access works).

def test_538_invalid_env_token_401_names_the_token(monkeypatch):
    import time as _time
    monkeypatch.setenv('GITHUB_TOKEN', 'ghp_invalid_token_for_audit_test')
    slug = f'fivethirtyeight/zz-no-cache-{_time.time_ns()}'
    with pytest.raises(HypertoolsIOError, match='GITHUB_TOKEN'):
        hyp.load(slug)


# --------------------------------------------------------------- F20-004
# load() must not hijack arbitrary unpickled objects exposing .get_data()
# (only legacy DataGeometry pickles are unwrapped).

def test_pickled_line2d_roundtrips_as_line2d(tmp_path):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots()
    ln, = ax.plot([1, 2, 3], [4, 5, 6])
    path = tmp_path / 'line2d.pkl'
    hyp.save(ln, str(path))
    out = hyp.load(str(path))
    assert isinstance(out, Line2D), \
        f'expected Line2D back, got {type(out)}'
    plt.close('all')


def test_datageometry_pickle_still_unwrapped_to_raw_data(tmp_path):
    from hypertools.datageometry import DataGeometry
    geo = DataGeometry(data=[np.arange(6, dtype=float).reshape(3, 2)])
    path = tmp_path / 'legacy.geo'
    with open(path, 'wb') as f:
        pickle.dump(geo, f)
    out = hyp.load(str(path))
    assert isinstance(out, list)
    np.testing.assert_allclose(out[0], np.arange(6, dtype=float).reshape(3, 2))


# --------------------------------------------------------------- F20-007
# corrupt/truncated pickles must produce one clear error naming the path.

def test_corrupt_pickle_error_names_corruption(tmp_path):
    path = tmp_path / 'truncated.pkl'
    path.write_bytes(b'\x80\x04TRUNC')
    with pytest.raises(HypertoolsIOError,
                       match='truncated or corrupted') as excinfo:
        hyp.load(str(path))
    assert 'truncated.pkl' in str(excinfo.value)


def test_zero_byte_pickle_raises_hypertools_error_not_eoferror(tmp_path):
    path = tmp_path / 'halfwritten.pkl'
    path.touch()
    with pytest.raises(HypertoolsIOError):
        hyp.load(str(path))


# --------------------------------------------------------------- F19-008
# stale module references / step numbering in comments+docstrings.

def test_sources_docstrings_use_current_module_paths():
    import hypertools.io.sources as sources
    assert 'tools.load' not in sources.__doc__
    assert 'steps 4-9' not in sources.load_source.__doc__
