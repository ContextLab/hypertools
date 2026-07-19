# -*- coding: utf-8 -*-
"""hyp.load() gains two explicit-prefix resolvers (GH #116), inserted right
after the seaborn resolver and before local-file resolution:

- 'fivethirtyeight/<slug>' -- a dataset from
  https://github.com/fivethirtyeight/data, loaded by listing the slug's
  folder via the GitHub contents API and downloading every CSV it
  contains from raw.githubusercontent.com.
- 'kaggle/<owner>/<dataset>' -- a Kaggle dataset, downloaded anonymously
  via kagglehub.dataset_download.

Both are explicit, unambiguous prefixes: no shared index is maintained
(Jeremy's directive on GH #116), and a matching-but-failing name raises
HypertoolsIOError directly instead of falling through to the rest of the
resolution chain.

All tests use real function calls -- no mocks. They hit the real GitHub
API / raw.githubusercontent.com and the real Kaggle Hub over the network,
using small, public datasets chosen to keep downloads tiny and avoid
GitHub's unauthenticated 60 requests/hour rate limit (the fivethirtyeight
folder listing is cached per-process in hypertools.io.sources, so repeated
loads of the same slug within a test run don't re-hit the API).
"""

import os

import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools._shared.exceptions import HypertoolsIOError


def test_load_538_bechdel_single_csv():
    # bechdel's folder has exactly one CSV (movies.csv) -> a DataFrame
    df = hyp.load('fivethirtyeight/bechdel')
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1000
    assert 'binary' in df.columns


def test_load_538_multi_csv_returns_dict():
    # college-majors' folder has five (small) CSVs -> a dict of DataFrames
    data = hyp.load('fivethirtyeight/college-majors')
    assert isinstance(data, dict)
    assert len(data) > 1
    expected = {'all-ages', 'grad-students', 'majors-list', 'recent-grads',
               'women-stem'}
    assert expected.issubset(data.keys())
    for df in data.values():
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


def test_load_538_bad_slug_raises_immediately():
    # an explicit prefix that doesn't resolve raises directly -- it does
    # NOT silently fall through to local-file/HF/URL resolution
    with pytest.raises(HypertoolsIOError) as excinfo:
        hyp.load('fivethirtyeight/definitely-not-real-xyz')
    message = str(excinfo.value)
    assert 'fivethirtyeight' in message


def test_load_kaggle_tiny_dataset():
    # kagglehub is only required for this test and the one below (GH #205
    # skip pattern: the skip must be scoped to the tests that actually need
    # the optional dependency, not the whole module -- see
    # test_ci_has_kagglehub, which fails hard on CI if it's ever missing).
    pytest.importorskip(
        'kagglehub',
        reason="kagglehub is required to exercise hyp.load('kaggle/...') "
              "-- install it with `pip install hypertools[kaggle]`")
    # uciml/iris is a tiny (~3.6KB) public dataset that downloads
    # anonymously (no Kaggle credentials needed) via kagglehub
    df = hyp.load('kaggle/uciml/iris')
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.shape[1] > 1


def test_load_kaggle_malformed_name_raises_clear_error():
    # malformed-name validation happens before kagglehub is even imported
    # (hypertools/io/sources.py's kaggle_dataset checks the id shape
    # first), so this test doesn't need kagglehub installed at all -- no
    # importorskip here.
    with pytest.raises(HypertoolsIOError) as excinfo:
        hyp.load('kaggle/no-slash')
    message = str(excinfo.value)
    assert 'kaggle' in message.lower()
    assert '<owner>' in message or 'owner' in message.lower()


def test_ci_has_kagglehub():
    # GH #205-style CI guard: on GitHub Actions, kagglehub must be
    # importable so test_load_kaggle_tiny_dataset actually exercises the
    # kaggle_dataset() code path instead of silently skipping on every PR.
    # Outside CI, a missing kagglehub is expected (it's an optional extra)
    # and this test is a no-op.
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        pytest.skip("only meaningful on CI (GITHUB_ACTIONS=true); a "
                    "missing kagglehub on a local machine is expected and "
                    "handled by importorskip in the kaggle-specific tests")
    import kagglehub
    assert kagglehub is not None


def test_load_538_bechdel_plot_end_to_end():
    df = hyp.load('fivethirtyeight/bechdel')
    numeric = df.select_dtypes(include='number').dropna()
    fig = hyp.plot(numeric, show=False)
    assert fig is not None


# --- transient-5xx retry (real GitHub-API 502 flake) --------------------
# CI intermittently failed the 538 loader on a transient "502 Bad Gateway"
# from api.github.com (a shared runner-IP pool hammering the API). The loader
# now retries transient gateway errors via
# hypertools.io.sources._github_get_with_retry. Exercised here against a REAL
# local HTTP server (a real socket over the loopback interface -- NOT a mock
# object) that returns 502 a few times before 200, so the retry actually
# runs against live HTTP.
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from hypertools.io.sources import _github_get_with_retry


def _start_flaky_server(n_502, body=b'ok'):
    """Start a real loopback HTTP server that answers its first ``n_502``
    GETs with 502, then 200 + ``body``. Returns (url, server, hits)."""
    hits = [0]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits[0] += 1
            if hits[0] <= n_502:
                self.send_response(502)
                self.end_headers()
                self.wfile.write(b'Bad Gateway')
            else:
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_address[1]}/data'
    return url, server, hits


def test_github_get_with_retry_recovers_from_transient_502():
    url, server, hits = _start_flaky_server(n_502=2, body=b'recovered')
    try:
        resp = _github_get_with_retry(url, {}, timeout=5, backoff=0.01)
    finally:
        server.shutdown()
    assert resp.status_code == 200
    assert resp.content == b'recovered'
    assert hits[0] == 3  # two 502s retried away, the third GET succeeded


def test_github_get_with_retry_returns_final_502_after_exhausting():
    url, server, hits = _start_flaky_server(n_502=99)  # always 502
    try:
        resp = _github_get_with_retry(url, {}, timeout=5, attempts=3,
                                      backoff=0.01)
    finally:
        server.shutdown()
    # a persistent 502 is returned (not raised) after the retries are spent,
    # so each caller's own status handling still produces the actionable
    # HypertoolsIOError / raise_for_status it did before.
    assert resp.status_code == 502
    assert hits[0] == 3
