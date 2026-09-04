# -*- coding: utf-8 -*-
"""Universal loader: hyp.load resolves strings through
builtin -> local file -> Hugging Face -> Google Sheets -> Google Drive ->
Dropbox -> URL, and lists of strings resolve to lists of datasets. All
tests use real files and real network calls (no mocks)."""

import functools
import http.server
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp
from hypertools._shared.exceptions import HypertoolsIOError
from hypertools.io.sources import (is_loadable_string, HypertoolsTrustError,
                                    _table_file_keys)
from tests._netskip import (is_transient_network as _is_transient_network,
                            skip_on_transient_network as
                            _skip_on_transient_network)

IRIS_CSV = 'raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
# legacy hypertools 'spiral' pickle, still hosted on Google Drive
DRIVE_SPIRAL_ID = '1nHAusn2VsQinJk35xvJSd7CtWPC1uOwK'
DROPBOX_BUNNY = 'https://www.dropbox.com/s/7d9vo9idqk1hn31/bunny.pkl?dl=0'
# public 476MB file that trips Google Drive's "can't scan this file for
# viruses" large-file interstitial instead of serving the file directly
DRIVE_BIG_FILE_ID = '1l_5RK28JRL19wpT22B-DY9We3TVXnnQQ'
# Google's own Sheets-API-quickstart public sample sheet ("Class Data")
GOOGLE_SHEETS_SAMPLE_URL = (
    'https://docs.google.com/spreadsheets/d/'
    '1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit#gid=0')

# Hosted-dataset matrix tests must not fail unrelated CI on a TRANSIENT network
# error (a Hugging Face ReadTimeout, a 5xx, a dropped connection) -- skip on
# those, matching tests/test_dataset_compat.py's convention, while still
# exercising the real load path when the host is reachable. Real (non-transient)
# errors still propagate and fail. (2026-07: a HF ReadTimeout on
# test_load_huggingface_dataset flaked one ubuntu-3.13 matrix cell.)
#
# 2026-07-30: applied to EVERY test in this file that fetches over the network,
# not just the Hugging Face pair. `test_load_google_sheet_live` failed a full
# local run when Google Sheets read-timed-out and Drive answered 500, then
# passed on re-run 2 minutes later -- it was the only live-fetch test here
# without the guard its siblings already had. The guarded set is now: Google
# Sheets, Google Drive (spiral + the bigdata interstitial download), Dropbox,
# the generic-URL pair, and plot()'s auto-load. Guarding is NOT weakening:
# _is_transient_network is a narrow, unit-tested predicate that matches
# timeouts / 5xx / dropped connections / DNS failures and deliberately does
# NOT match assertion or key errors, so a genuine regression still fails.
# Verified end-to-end both ways before shipping.
# The classifier itself now lives in tests/_netskip.py, because more than one
# test file needs it (tests/test_load_sklearn_seaborn.py fetches seaborn-data
# over the network too). Imported under the original private names so this
# file's own unit tests below keep testing the real implementation.
#
# 2026-07-31 (review round 13): that predicate was substring-matching the whole
# lowercased message, including a bare 'timeout', so THREE classes of real
# defect read as transient -- most seriously an aggregate holding both the
# intended resolver's genuine failure and an unrelated fallback's timeout,
# which load_source() produces routinely (sources.py:664-670). Classification is
# now structural (exception type + HTTP status), and any defect-shaped attempt
# line vetoes the whole aggregate. The cases below pin all three.


def test_is_transient_network_recognizes_wrapped_load_errors():
    # hyp.load() wraps the underlying network error into its own diagnostic, so
    # the classifier must recognize it from that wrapped message -- including a
    # DNS failure (urllib3 NameResolutionError -> "Failed to resolve ...").
    dns = HypertoolsIOError(
        "could not load 'scikit-learn/iris'. Tried, in order:\n"
        "  - Hugging Face dataset: NameResolutionError: Failed to resolve "
        "'huggingface.co' ([Errno -3] Temporary failure in name resolution)")
    timeout = HypertoolsIOError(
        "could not load 'scikit-learn/iris'. Tried, in order:\n"
        "  - Hugging Face dataset: ReadTimeout: The read operation timed out")
    unavailable = HypertoolsIOError('503 Server Error: Service Unavailable')
    # verbatim from a real 2026-07-30 failure of test_load_google_sheet_live:
    # every fallback failed at once, Drive with a 500 and Sheets/URL with
    # 60s read timeouts. The test passed again on re-run 2 minutes later.
    sheets_outage = HypertoolsIOError(
        "could not load 'https://docs.google.com/spreadsheets/d/1BxiMV/edit'."
        " Tried, in order:\n"
        "  - Google Sheets: ReadTimeout: HTTPSConnectionPool("
        "host='docs.google.com', port=443): Read timed out. (read timeout=60)\n"
        "  - Google Drive (1BxiMV): HTTPError: 500 Server Error: Internal "
        "Server Error for url: https://drive.usercontent.google.com/download")
    # and a 500 on its own, with no timeout anywhere in the message
    only_500 = HypertoolsIOError(
        "  - Google Drive (x): HTTPError: 500 Server Error: Internal Server "
        "Error for url: https://drive.usercontent.google.com/download")
    for exc in (dns, timeout, unavailable, sheets_outage, only_500):
        # the exception itself is the supported input; str() is the wrapped
        # aggregate case, where per-resolver detail exists only as text
        assert _is_transient_network(exc), str(exc)
        assert _is_transient_network(str(exc)), str(exc)
    # a genuine failure is NOT transient and must still surface
    assert not _is_transient_network("AssertionError: shape (150,) != (149,)")
    assert not _is_transient_network("KeyError: 'SepalLengthCm'")
    # ...including one whose numbers happen to look like HTTP status codes.
    # This is why 5xx is matched as `\b5\d\d server error\b`, never a bare '500'.
    assert not _is_transient_network("AssertionError: shape 500 != 499")
    assert not _is_transient_network("assert (502, 3) == (504, 3)")


def test_transient_classifier_does_not_excuse_a_real_resolver_failure():
    """Review round 13. load_source() lists EVERY resolver it tried in one
    HypertoolsIOError (sources.py:664-670), so the resolver a test actually
    wanted can fail for a real reason while an unrelated fallback times out.
    The old predicate substring-matched the whole message and skipped on the
    timeout, hiding the regression. Any defect-shaped attempt line must veto."""
    mixed = HypertoolsIOError(
        "could not load 'penguins'. Tried, in order:\n"
        "  - seaborn dataset: KeyError: parser regression\n"
        "  - Hugging Face dataset: ReadTimeout: timed out")
    assert not _is_transient_network(mixed)
    assert not _is_transient_network(str(mixed))

    # 'timeout' is also a keyword argument and ordinary English -- it must
    # carry no verdict on its own, as an exception OR as text.
    for exc in (ValueError('timeout must be positive'),
                AssertionError('timeout metadata missing'),
                AssertionError('request timed out')):
        assert not _is_transient_network(exc), exc
        assert not _is_transient_network(f'{type(exc).__name__}: {exc}'), exc

    # a 4xx is OUR bug (the dataset URL moved), not the host having a bad day
    assert not _is_transient_network(
        '  - URL (http://x/iris.csv): HTTPError: 404 Client Error: '
        'Not Found for url: http://x/iris.csv')

    # expected chain noise ('local file: not found') must not be read as a
    # defect, or a genuine outage would stop skipping
    assert _is_transient_network(HypertoolsIOError(
        "could not load 'https://x/iris.csv'. Tried, in order:\n"
        "  - local file: not found at /nonexistent/https:/x/iris.csv\n"
        "  - URL (https://x/iris.csv): ReadTimeout: Read timed out."))


def test_transient_classifier_uses_exception_type_not_message_text():
    """A real requests/urllib3 stack is recognized by TYPE, so a host that
    words its outage unusually is still caught -- and a defect raised inside an
    `except Timeout:` block does not inherit that timeout's verdict."""
    requests = pytest.importorskip('requests')

    assert _is_transient_network(requests.exceptions.ReadTimeout('...'))
    assert _is_transient_network(requests.exceptions.ConnectionError('...'))

    # requests wrapping urllib3, the shape real failures actually arrive in
    try:
        try:
            raise requests.exceptions.ConnectTimeout('urllib3 said so')
        except Exception as inner:
            raise requests.exceptions.ConnectionError('wrapped') from inner
    except Exception as wrapped:
        assert _is_transient_network(wrapped)

    # 5xx is the host; 4xx is us. The old '5xx' marker list could not tell them
    # apart because it only ever saw text.
    class _Resp:
        def __init__(self, code):
            self.status_code = code

    for code, transient in ((500, True), (503, True), (404, False),
                            (403, False)):
        err = requests.exceptions.HTTPError(f'{code} Server Error')
        err.response = _Resp(code)
        assert _is_transient_network(err) is transient, code

    # a genuine defect raised while handling a timeout is still a defect
    try:
        try:
            raise requests.exceptions.ReadTimeout('read timed out')
        except Exception:
            raise KeyError('SepalLengthCm')
    except Exception as during:
        assert not _is_transient_network(during)


def test_skip_on_transient_network_skips_dns_but_reraises_real(monkeypatch):
    # pin the DEFAULT (non-strict) behaviour explicitly: this is a unit test of
    # the machinery, so it must not change verdict with the ambient
    # environment. Without this the `live-source-gate` job, which exports
    # HYPERTOOLS_REQUIRE_LIVE_SOURCES=1 to disable skipping, fails it.
    monkeypatch.delenv('HYPERTOOLS_REQUIRE_LIVE_SOURCES', raising=False)
    dns = HypertoolsIOError("NameResolutionError: Failed to resolve 'huggingface.co'")
    with pytest.raises(BaseException) as excinfo:      # pytest.skip -> Skipped
        with _skip_on_transient_network('loading x'):
            raise dns
    assert excinfo.type.__name__ == 'Skipped'
    # a real error propagates unchanged (not skipped)
    with pytest.raises(ValueError, match='genuine bug'):
        with _skip_on_transient_network('loading x'):
            raise ValueError('genuine bug')


def test_require_live_sources_turns_every_skip_into_a_failure(monkeypatch):
    """HYPERTOOLS_REQUIRE_LIVE_SOURCES=1 (set by the `live-source-gate` CI job)
    must re-raise even a genuine outage, so a live source that has been
    unreachable for a week cannot hide behind a green suite. Real environment
    variable, real context manager -- nothing is stubbed."""
    dns = HypertoolsIOError("NameResolutionError: Failed to resolve 'x'")

    monkeypatch.delenv('HYPERTOOLS_REQUIRE_LIVE_SOURCES', raising=False)
    with pytest.raises(BaseException) as excinfo:
        with _skip_on_transient_network('loading x'):
            raise dns
    assert excinfo.type.__name__ == 'Skipped'

    monkeypatch.setenv('HYPERTOOLS_REQUIRE_LIVE_SOURCES', '1')
    with pytest.raises(HypertoolsIOError, match='Failed to resolve'):
        with _skip_on_transient_network('loading x'):
            raise dns


@pytest.fixture
def http_dir_server(tmp_path):
    """Serve tmp_path over a real HTTP socket on a random localhost port
    (no mocks: requests made against this fixture hit a real socket)."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield tmp_path, f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_is_loadable_string_discrimination():
    assert is_loadable_string('spiral')                      # builtin
    assert is_loadable_string('https://example.com/x.csv')   # URL
    assert is_loadable_string(IRIS_CSV)                      # schemeless URL
    assert is_loadable_string('scikit-learn/iris')           # HF id
    assert is_loadable_string(DRIVE_SPIRAL_ID)               # bare Drive id
    assert is_loadable_string(DROPBOX_BUNNY)
    # raw text must NOT be treated as a data source
    assert not is_loadable_string('the dog is happy')
    assert not is_loadable_string('')
    assert not is_loadable_string('nonexistent_dataset_name')


def test_table_file_keys_unique_stems_use_plain_stem(tmp_path):
    # flat, no-collision case: nicer UX keys by bare filename stem
    a = tmp_path / 'alpha.csv'
    b = tmp_path / 'beta.csv'
    a.write_text('x,y\n1,2\n')
    b.write_text('x,y\n3,4\n')
    keys = _table_file_keys(tmp_path, [a, b])
    assert keys == {a: 'alpha', b: 'beta'}


def test_table_file_keys_colliding_stems_use_relative_path(tmp_path):
    # GH review finding: two same-stem CSVs in different subdirectories
    # must NOT silently overwrite each other in the resulting dict --
    # real files on disk, no mocks.
    sub = tmp_path / 'sub'
    other = tmp_path / 'other'
    sub.mkdir()
    other.mkdir()
    f1 = sub / 'data.csv'
    f2 = other / 'data.csv'
    f1.write_text('x,y\n1,2\n')
    f2.write_text('x,y\n5,6\n')
    keys = _table_file_keys(tmp_path, [f1, f2])
    assert keys == {f1: 'sub/data', f2: 'other/data'}
    assert len(set(keys.values())) == 2, "colliding stems overwrote a key"


def test_load_local_formats(tmp_path):
    arr = np.random.default_rng(0).standard_normal((12, 4))
    np.save(tmp_path / 'a.npy', arr)
    np.savez(tmp_path / 'b.npz', x=arr, y=arr * 2)
    pd.DataFrame(arr).to_csv(tmp_path / 'c.csv', index=False)
    pd.DataFrame(arr).to_csv(tmp_path / 'd.tsv', sep='\t', index=False)
    pd.DataFrame(arr).to_json(tmp_path / 'e.json')
    pd.to_pickle(pd.DataFrame(arr), tmp_path / 'f.pkl')

    np.testing.assert_allclose(hyp.load(str(tmp_path / 'a.npy')), arr)
    npz = hyp.load(str(tmp_path / 'b.npz'))
    assert isinstance(npz, list) and len(npz) == 2
    assert hyp.load(str(tmp_path / 'c.csv')).shape == (12, 4)
    assert hyp.load(str(tmp_path / 'd.tsv')).shape == (12, 4)
    assert hyp.load(str(tmp_path / 'e.json')).shape == (12, 4)
    assert hyp.load(str(tmp_path / 'f.pkl')).shape == (12, 4)


def test_load_huggingface_dataset():
    pytest.importorskip('datasets')
    with _skip_on_transient_network('loading scikit-learn/iris'):
        df = hyp.load('scikit-learn/iris')
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 150


def test_load_huggingface_streaming_flows_to_plot():
    pytest.importorskip('datasets')
    # wrap ONLY the network-touching calls (the stream open + the plot that
    # pulls chunks over the wire), so a genuine regression in the non-network
    # assertions can't be masked just because its message happens to contain a
    # word like "timeout" (release review hardening).
    with _skip_on_transient_network('streaming scikit-learn/iris'):
        ds = hyp.load('scikit-learn/iris', streaming=True)
    from hypertools.io.streaming import is_stream
    assert is_stream(ds)
    ds = ds.select_columns(['SepalLengthCm', 'SepalWidthCm',
                            'PetalLengthCm', 'PetalWidthCm'])
    # iris' later rows fall outside the display box fitted on the first 50,
    # provoking the clamped-samples notice. Wrap ONLY the network-touching plot
    # (it pulls chunks over the wire) in the transient-skip, and RECORD warnings
    # so the assertion that the clamped-samples RuntimeWarning fired lives
    # OUTSIDE the skip -- a genuine regression that stops emitting it then FAILS
    # the test rather than being masked by a transient-network marker (e.g.
    # "timeout") in some unrelated message (release review hardening).
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with _skip_on_transient_network('reading the scikit-learn/iris stream'):
            fig = hyp.plot(ds, '.', show=False, stream_init=50, stream_chunk=50)
    assert any('outside the display box' in str(w.message) for w in caught), (
        'expected the clamped-samples RuntimeWarning from the streaming plot')
    assert fig.stream_info['n_samples'] == 150
    plt.close('all')


def test_load_streaming_rejects_load_time_transforms():
    pytest.importorskip('datasets')
    with pytest.raises(ValueError, match='stream'):
        hyp.load('scikit-learn/iris', streaming=True, ndims=3)


def test_load_google_drive_id_and_url():
    # loading a remote pickle by raw Drive id / URL now REQUIRES trust=True
    # (2026-07 release review, blocker #1: a warning is not a security
    # boundary -- unpickling executes arbitrary code). Without it, load
    # refuses; with it, load() returns the raw data (a list of arrays),
    # never a DataGeometry.
    url = ('https://drive.google.com/uc?export=download&id=' + DRIVE_SPIRAL_ID)
    with _skip_on_transient_network('loading the spiral pickle from Drive'):
        for src in (DRIVE_SPIRAL_ID, url):
            # a network outage here surfaces as HypertoolsIOError rather than
            # HypertoolsTrustError, so pytest.raises lets it through to the
            # transient guard instead of failing the trust assertion
            with pytest.raises(HypertoolsTrustError,
                               match='refusing to unpickle'):
                hyp.load(src)
        data = hyp.load(DRIVE_SPIRAL_ID, trust=True)
        data2 = hyp.load(url, trust=True)
    assert isinstance(data, list)
    assert isinstance(data2, list)


def test_load_dropbox_url_forms():
    # remote pickles fetched by URL are refused without trust=True; every
    # URL form normalizes to the same download and, with trust=True, yields
    # the same (N, 3) point cloud
    forms = (DROPBOX_BUNNY,                                # dl=0 -> normalized
             'www.dropbox.com/s/7d9vo9idqk1hn31/bunny.pkl',
             's/7d9vo9idqk1hn31/bunny.pkl')               # shared-link path
    with _skip_on_transient_network('loading the bunny pickle from Dropbox'):
        for src in forms:
            with pytest.raises(HypertoolsTrustError,
                               match='refusing to unpickle'):
                hyp.load(src)
        shapes = [np.asarray(hyp.load(src, trust=True)).shape[1]
                  for src in forms]
    assert shapes == [3, 3, 3]


def test_load_generic_url_with_and_without_scheme():
    with _skip_on_transient_network('loading iris.csv by URL'):
        df1 = hyp.load('https://' + IRIS_CSV)
        df2 = hyp.load(IRIS_CSV)
    assert df1.shape == df2.shape == (150, 5)


def test_load_list_of_strings(tmp_path):
    arr = np.random.default_rng(1).standard_normal((9, 3))
    np.save(tmp_path / 'x.npy', arr)
    out = hyp.load(['spiral', str(tmp_path / 'x.npy')])
    assert isinstance(out, list) and len(out) == 2
    # 'spiral' resolves to its raw data (a list of arrays), not a geo
    assert isinstance(out[0], list)
    np.testing.assert_allclose(out[1], arr)


def test_plot_accepts_source_strings():
    # plot() auto-loads a source-string as its data (the old geo.plot()
    # replay method is gone in 2.0; plot() returns a Figure). Auto-loading
    # this mixed numeric/text CSV yields a single-observation component
    # internally, provoking the cannot-reduce-a-single-observation notice
    # (verified: the only warning this call emits)
    with _skip_on_transient_network('auto-loading iris.csv through plot()'):
        with pytest.warns(UserWarning,
                          match='Cannot reduce a single observation'):
            fig = hyp.plot(IRIS_CSV, show=False)
    assert type(fig).__module__.startswith('matplotlib')
    plt.close('all')


def test_plot_mixed_dtype_dataframe():
    """Regression: pandas>=2.0 get_dummies returns bool, which made
    df2mat produce object arrays that crashed np.isnan in format_data."""
    df = pd.DataFrame({'a': np.random.rand(20), 'b': np.random.rand(20),
                       'c': ['x', 'y'] * 10})
    from hypertools.tools.df2mat import df2mat
    m = df2mat(df)
    assert m.dtype == np.float64
    # `plot()` returns a matplotlib Figure in 1.0 (the pre-1.0 DataGeometry it
    # used to return is gone) -- assert that rather than binding an unused
    # `geo`, so the call's result is actually checked, not just its not-raising
    fig = hyp.plot(df, '.', show=False)
    assert type(fig).__module__.startswith('matplotlib')
    plt.close('all')


def test_load_unresolvable_string_lists_attempts():
    with pytest.raises(HypertoolsIOError, match='Tried, in order'):
        hyp.load('no_such_dataset_or_file_xyz123')


# ---- Google Drive large-file interstitial ----

def test_parse_drive_interstitial_from_real_capture():
    """Parses the real interstitial HTML captured live from
    DRIVE_BIG_FILE_ID at implementation time (before confirm params were
    supplied); saved verbatim as a fixture since it's small (2.4KB)."""
    from hypertools.io.sources import parse_drive_interstitial
    html = (Path(__file__).parent / 'data' /
            'drive_large_file_interstitial.html').read_text()
    parsed = parse_drive_interstitial(html)
    assert parsed is not None
    action_url, params = parsed
    assert action_url == 'https://drive.usercontent.google.com/download'
    assert params == {
        'id': DRIVE_BIG_FILE_ID,
        'export': 'download',
        'confirm': 't',
        'uuid': 'ea85f6bc-9da9-43b1-8109-20a4e7459918',
    }


def test_parse_drive_interstitial_returns_none_for_other_html():
    from hypertools.io.sources import parse_drive_interstitial
    assert parse_drive_interstitial('<html><body>hi</body></html>') is None
    assert parse_drive_interstitial('not html at all') is None


@pytest.mark.bigdata
def test_load_drive_large_file_interstitial_live():
    """Live end-to-end: hyp.load() on a 476MB Drive file
    ('fcn8s_from_caffe.npz', a Caffe FCN-8s model) that Google Drive
    answers with a "can't scan this file for viruses" HTML interstitial
    instead of the file. Proves _fetch_bytes parses the interstitial's
    confirm form (id/export/confirm/uuid hidden inputs) and re-fetches
    from drive.usercontent.google.com/download rather than raising on the
    HTML. 42 named arrays (conv/fc/score/upscore weights+biases).
    Manually verified (see task-2-report.md) to download exactly
    498,881,336 bytes -- matching Drive's own Content-Length -- in ~24s;
    shipped here as a full download since that proved fast enough to run
    routinely under the bigdata marker."""
    with _skip_on_transient_network('downloading the 476MB Drive file'):
        arrays = hyp.load(DRIVE_BIG_FILE_ID)
    assert isinstance(arrays, list) and len(arrays) == 42
    assert all(isinstance(a, np.ndarray) and a.dtype == np.float32
              for a in arrays)
    # ~124M total float32 params -> the file is genuinely ~476MB, not a
    # truncated/HTML stand-in
    assert sum(a.size for a in arrays) > 1e8


# ---- Google Sheets ----

def test_normalize_google_sheet_url_rewrite():
    from hypertools.io.sources import _normalize_google_sheet
    rewritten = _normalize_google_sheet(GOOGLE_SHEETS_SAMPLE_URL)
    assert rewritten == (
        'https://docs.google.com/spreadsheets/d/'
        '1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/export?format=csv')
    assert _normalize_google_sheet('https://example.com/not-a-sheet') is None
    assert _normalize_google_sheet(
        'https://drive.google.com/file/d/abcdefghijklmnopqrstuvwxyzabcde/'
        'view') is None


def test_load_google_sheet_live():
    with _skip_on_transient_network('loading the Google Sheets sample'):
        df = hyp.load(GOOGLE_SHEETS_SAMPLE_URL)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (30, 6)


# ---- Excel (.xlsx / .xls) ----

def test_load_xlsx_roundtrip(tmp_path):
    arr = np.random.default_rng(2).standard_normal((8, 3))
    df = pd.DataFrame(arr, columns=['a', 'b', 'c'])
    path = tmp_path / 'data.xlsx'
    df.to_excel(path, index=False)
    out = hyp.load(str(path))
    pd.testing.assert_frame_equal(out, df)


def test_load_xls_without_xlrd_raises_friendly_error(tmp_path):
    """xlrd is intentionally NOT installed (it's only needed for the
    legacy binary .xls format, which pandas can't even start parsing
    without it -- read_excel raises ImportError before touching the
    bytes). So a genuinely-valid .xls file isn't needed to exercise the
    friendly-error path: any real file named '.xls' hits the same
    missing-dependency check. We use arbitrary bytes (rather than a
    parseable file) precisely to prove the ImportError fires before any
    content parsing is attempted."""
    import importlib.util
    if importlib.util.find_spec('xlrd') is not None:
        pytest.skip('xlrd is installed in this environment; the '
                    'missing-dependency path is not exercised here')
    path = tmp_path / 'legacy.xls'
    path.write_bytes(b'not a real xls file -- exercises the '
                     b'missing-xlrd error path')
    with pytest.raises(ImportError, match='pip install xlrd'):
        hyp.load(str(path))


# ---- Remote pickle / npy trust policy ----

def test_remote_pickle_refused_without_trust(http_dir_server):
    # a warning is not a security boundary: a remote pickle must be REFUSED
    # (not merely warned about, then unpickled) unless trust=True is passed
    # (2026-07 release review, blocker #1)
    tmp_path, base_url = http_dir_server
    pd.to_pickle(pd.DataFrame({'a': [1, 2, 3]}), tmp_path / 'd.pkl')
    with pytest.raises(HypertoolsTrustError, match='refusing to unpickle'):
        hyp.load(base_url + '/d.pkl')


def test_remote_pickle_trust_true_allows_and_is_silent(http_dir_server):
    tmp_path, base_url = http_dir_server
    pd.to_pickle(pd.DataFrame({'a': [1, 2, 3]}), tmp_path / 'd.pkl')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        df = hyp.load(base_url + '/d.pkl', trust=True)
    assert df.shape == (3, 1)


def test_remote_npy_object_array_blocked_without_trust(http_dir_server):
    tmp_path, base_url = http_dir_server
    np.save(tmp_path / 'obj.npy', np.array([{'a': 1}], dtype=object))
    with pytest.raises(HypertoolsTrustError, match='allow_pickle'):
        hyp.load(base_url + '/obj.npy')


def test_remote_npy_object_array_allowed_with_trust(http_dir_server):
    tmp_path, base_url = http_dir_server
    np.save(tmp_path / 'obj.npy', np.array([{'a': 1}], dtype=object))
    out = hyp.load(base_url + '/obj.npy', trust=True)
    assert out[0] == {'a': 1}


def test_remote_npy_numeric_no_warning_no_trust_needed(http_dir_server):
    tmp_path, base_url = http_dir_server
    arr = np.random.default_rng(3).standard_normal((5, 3))
    np.save(tmp_path / 'num.npy', arr)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = hyp.load(base_url + '/num.npy')
    np.testing.assert_allclose(out, arr)


def test_remote_malformed_csv_raises_digest_not_raw_parser_error(
        http_dir_server):
    """A remote parse failure (pandas ParserError, which subclasses
    ValueError like the trust-policy error does) must NOT be mistaken
    for the trust-policy error and escape raw -- it should join the
    HypertoolsIOError "Tried, in order" digest instead."""
    tmp_path, base_url = http_dir_server
    # unterminated quoted field: the python csv engine raises a real
    # pandas.errors.ParserError ("unexpected end of data") for this.
    (tmp_path / 'bad.csv').write_bytes(b'a,b\n"unterminated,1\n')
    with pytest.raises(HypertoolsIOError, match='Tried, in order'):
        hyp.load(base_url + '/bad.csv')


def test_local_pickle_never_warns_or_restricts(tmp_path):
    pd.to_pickle(pd.DataFrame({'a': [1, 2, 3]}), tmp_path / 'd.pkl')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        df = hyp.load(str(tmp_path / 'd.pkl'))
    assert df.shape == (3, 1)


def test_builtin_example_data_exempt_from_trust_policy():
    # 'spiral' is remote (hosted on Google Drive) but a built-in example
    # name -- it never goes through sources.py's trust-gated parsing.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        data = hyp.load('spiral')
    assert isinstance(data, list)
