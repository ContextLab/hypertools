# -*- coding: utf-8 -*-
"""``hyp.load(..., offline=True)`` really is offline (1.1 release review, I1).

Before 1.1, ``hyp.load(url, offline=True)`` consulted the seaborn dataset
listing (an urlopen with no timeout) for EVERY non-builtin string -- cached
URLs included -- before the URL cache was even looked at, and a failed
listing fetch was forgotten immediately, so on a dead network every call
blocked for the full OS connect timeout (75 s measured behind an
unroutable proxy) while the docstring promised "never open a connection".

The observable here is a real one: a local TCP "proxy" that accepts every
connection and never answers (``_Blackhole``). The library runs in a
subprocess whose ``HTTP(S)_PROXY`` point at it, so any attempt to reach
the network shows up as an accepted connection at the blackhole and as a
stall until the client's own timeout. No mocks, no monkeypatched
functions: the data server is a real ``http.server``, the cache is real
files on disk, and every assertion is on a returned value, an exception
type, an elapsed time or a connection count.
"""

import functools
import json
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp                                       # noqa: E402
from hypertools.io.sources import (HypertoolsOfflineError,      # noqa: E402
                                   SEABORN_LISTING_TIMEOUT,
                                   cached_url_path, url_cache_dir)

CSV_TEXT = 'a,b\n1,2\n3,4\n'
EXPECTED = pd.DataFrame({'a': [1, 3], 'b': [2, 4]})
#: wall-clock budget (seconds) for one offline call inside the subprocess;
#: a call that touches the network stalls for at least the seaborn listing
#: timeout (10 s), and for ~75 s before 1.1
CALL_BUDGET = 5.0
#: hard cap on a whole subprocess run (interpreter start-up + imports
#: included) so a regression fails instead of hanging the suite
SUBPROCESS_CAP = 120.0


class _Blackhole:
    """A local TCP proxy that accepts every connection and never replies.

    A client routed through it hangs until its own timeout, and every
    connection is recorded (with the first bytes the client sent, e.g.
    ``CONNECT raw.githubusercontent.com:443``), so a test can assert both
    "no network attempt was made" and "exactly one was".
    """

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('127.0.0.1', 0))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        self.connections = []
        self._held = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self._held.append(conn)
            threading.Thread(target=self._peek, args=(conn,),
                             daemon=True).start()

    def _peek(self, conn):
        try:
            conn.settimeout(5)
            head = conn.recv(512)
        except OSError:
            head = b''
        self.connections.append(head)

    def env(self, no_proxy):
        url = f'http://127.0.0.1:{self.port}'
        return {'HTTP_PROXY': url, 'HTTPS_PROXY': url,
                'http_proxy': url, 'https_proxy': url,
                'NO_PROXY': no_proxy, 'no_proxy': no_proxy}

    def close(self):
        for conn in self._held:
            conn.close()
        self.sock.close()


@pytest.fixture
def blackhole():
    proxy = _Blackhole()
    yield proxy
    proxy.close()


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    path = tmp_path / 'urlcache'
    monkeypatch.setenv('HYPERTOOLS_URL_CACHE', str(path))
    assert url_cache_dir() == path
    return path


@pytest.fixture
def csv_server(tmp_path):
    """A real http.server serving ``data.csv``; ``.stop()`` shuts it down
    so a later direct fetch would be refused rather than served."""
    root = tmp_path / 'srv'
    root.mkdir()
    (root / 'data.csv').write_text(CSV_TEXT)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    httpd = HTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    class Server:
        url = f'http://127.0.0.1:{httpd.server_port}/data.csv'
        other_url = f'http://127.0.0.1:{httpd.server_port}/other.csv'

        @staticmethod
        def stop():
            httpd.shutdown()
            httpd.server_close()

    yield Server
    Server.stop()


_RUNNER = textwrap.dedent('''
    import json, sys, time, traceback
    import hypertools as hyp
    out = []
    for case in json.loads(sys.argv[1]):
        t = time.monotonic()
        try:
            data = hyp.load(case['source'], **case.get('kwargs', {}))
            rec = {'ok': True, 'shape': list(getattr(data, 'shape', [])),
                   'records': data.to_dict('list')
                   if hasattr(data, 'to_dict') else None}
        except Exception as e:
            rec = {'ok': False, 'type': type(e).__name__, 'msg': str(e)}
        rec['elapsed'] = time.monotonic() - t
        rec['source'] = case['source']
        out.append(rec)
    print('RESULT ' + json.dumps(out))
''')


def _run(cases, env, cwd):
    """Run ``hyp.load`` for each case in a fresh interpreter under ``env``
    and return the per-case records (elapsed measured around the call)."""
    full_env = {k: v for k, v in os.environ.items()
                if k.upper() not in ('HTTP_PROXY', 'HTTPS_PROXY',
                                     'NO_PROXY', 'ALL_PROXY')}
    full_env.update(env)
    full_env['MPLBACKEND'] = 'Agg'
    full_env['HYPERTOOLS_AUTO_INSTALL'] = '0'
    proc = subprocess.run(
        [sys.executable, '-c', _RUNNER, json.dumps(cases)],
        env=full_env, cwd=str(cwd), capture_output=True, text=True,
        timeout=SUBPROCESS_CAP)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith('RESULT ')]
    assert lines, (f'subprocess produced no result (rc={proc.returncode})\n'
                   f'stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-3000:]}')
    return json.loads(lines[-1][len('RESULT '):])


# ------------------------------------------------------- offline=True hits

def test_offline_hit_is_served_from_the_cache_without_any_connection(
        blackhole, cache_dir, csv_server, tmp_path):
    frame = hyp.load(csv_server.url, cache=True)       # populate, online
    pd.testing.assert_frame_equal(frame, EXPECTED)
    assert cached_url_path(csv_server.url).is_file()
    csv_server.stop()  # a direct re-fetch would now be refused, not served

    [rec] = _run([{'source': csv_server.url, 'kwargs': {'offline': True}}],
                 blackhole.env('nothing.invalid'), tmp_path)
    assert rec['ok'], rec
    pd.testing.assert_frame_equal(pd.DataFrame(rec['records']), EXPECTED)
    assert rec['elapsed'] < CALL_BUDGET, rec['elapsed']
    assert blackhole.connections == []        # never opened a connection


def test_offline_misses_raise_offline_error_without_any_connection(
        blackhole, cache_dir, csv_server, tmp_path):
    csv_server.stop()
    uncached = csv_server.other_url
    cases = [
        (uncached, 'a never-cached URL'),
        ('https://hypertools-offline-test.invalid/data.csv',
         'an unresolvable URL'),
        ('yahoo:AAPL', 'a web source'),
        ('fivethirtyeight/bechdel', 'a FiveThirtyEight dataset'),
        ('kaggle/uciml/iris', 'a Kaggle dataset'),
        ('scikit-learn/iris', 'a Hugging Face dataset id'),
        ('penguins', 'a seaborn dataset name'),
    ]
    recs = _run([{'source': s, 'kwargs': {'offline': True}}
                 for s, _ in cases],
                blackhole.env('nothing.invalid'), tmp_path)
    for (source, what), rec in zip(cases, recs):
        assert not rec['ok'], (what, rec)
        assert rec['type'] == HypertoolsOfflineError.__name__, (what, rec)
        assert 'offline=True' in rec['msg'], (what, rec)
        assert rec['elapsed'] < CALL_BUDGET, (what, rec['elapsed'])
    # the cacheable-URL miss names the cache path it looked for
    assert str(cached_url_path(uncached)) in recs[0]['msg']
    assert blackhole.connections == []


def test_offline_still_serves_local_sources(cache_dir, tmp_path):
    # built-in-by-package, synthetic and local-file sources need no
    # network, so offline=True must not refuse them
    local = tmp_path / 'local.csv'
    local.write_text(CSV_TEXT)
    pd.testing.assert_frame_equal(hyp.load(str(local), offline=True),
                                  EXPECTED)
    assert hyp.load('iris', offline=True).shape == (150, 5)
    assert hyp.load('helix', n_samples=12, offline=True).shape == (12, 3)


# ---------------------------------------------- the seaborn listing itself

def test_url_string_never_consults_the_seaborn_listing(
        blackhole, cache_dir, csv_server, tmp_path):
    # ONLINE load of a plain URL: the local server is reachable (NO_PROXY),
    # everything else is blackholed. A URL can never be a seaborn dataset
    # name, so the listing must not be fetched -- before 1.1 this call
    # went to the proxy for raw.githubusercontent.com first and stalled.
    [rec] = _run([{'source': csv_server.url}],
                 blackhole.env('127.0.0.1'), tmp_path)
    assert rec['ok'], rec
    pd.testing.assert_frame_equal(pd.DataFrame(rec['records']), EXPECTED)
    assert rec['elapsed'] < CALL_BUDGET, rec['elapsed']
    assert blackhole.connections == []


def test_seaborn_listing_fetch_is_bounded_and_its_failure_remembered(
        blackhole, cache_dir, tmp_path):
    # a plain name that resolves nowhere DOES consult the listing; behind a
    # dead network that fetch must (a) give up within its timeout and (b)
    # not be retried by the very next call
    name = 'no_such_dataset_zz'
    first, second = _run([{'source': name}, {'source': name}],
                         blackhole.env('nothing.invalid'), tmp_path)
    for rec in (first, second):
        assert not rec['ok'] and rec['type'] == 'HypertoolsIOError', rec
        assert 'seaborn dataset' in rec['msg']
    assert first['elapsed'] < SEABORN_LISTING_TIMEOUT + CALL_BUDGET, first
    assert second['elapsed'] < 1.0, second['elapsed']
    time.sleep(0.2)   # let the blackhole thread record the request head
    assert len(blackhole.connections) == 1, blackhole.connections
    assert blackhole.connections[0].startswith(b'CONNECT ')


def test_reset_seaborn_names_cache_forgets_a_remembered_failure():
    from hypertools.io import sources
    sources._seaborn_names_cache = None
    sources._seaborn_names_failed_at = time.monotonic()
    assert sources.seaborn_dataset('penguins') is None     # remembered miss
    sources.reset_seaborn_names_cache()
    assert sources._seaborn_names_failed_at is None
    assert sources._seaborn_names_cache is None
