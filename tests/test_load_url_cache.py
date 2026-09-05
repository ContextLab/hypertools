# -*- coding: utf-8 -*-
"""On-disk cache for arbitrary URL downloads (GH #285).

``load_source(url, cache=True)`` stores the download under
``~/hypertools_data/urls`` (``HYPERTOOLS_URL_CACHE`` overrides the
directory) with an atomic ``.part`` -> ``os.replace`` write, and reuses it
on the next load. ``offline=True`` reads ONLY from that cache and raises
``HypertoolsOfflineError`` -- naming the cache path it looked for -- when
the URL was never cached. ``cache=False`` (the default, matching
``trust=False``: hypertools does not write to the user's disk unless asked)
downloads without touching the cache at all.

Real downloads of a real small file from this repository -- no mocks. The
offline paths use the cache the earlier tests wrote, and one of them points
at a deliberately unreachable host to prove that ``offline=True`` never
opens a connection.
"""

import os

import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp                                       # noqa: E402
from hypertools._shared.exceptions import HypertoolsIOError     # noqa: E402
from hypertools.io.load import DATA_DIR                         # noqa: E402
from hypertools.io.sources import (HypertoolsOfflineError,      # noqa: E402
                                   _write_cached, cached_url_path,
                                   load_source, url_cache_dir)
from tests._netskip import skip_on_transient_network            # noqa: E402

# a small (42 KB), stable CSV in this repository, served raw over https
CSV_URL = ('https://raw.githubusercontent.com/ContextLab/hypertools/'
           'master/docs/tutorials/stock_closes_cached.csv')
# a host that cannot resolve: any attempt to fetch it fails loudly, which
# is what makes the offline-hit test meaningful
UNREACHABLE_URL = 'https://hypertools-offline-test.invalid/data.csv'


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the URL cache at a temp directory (so tests never write to
    the user's ~/hypertools_data) and return it."""
    path = tmp_path / 'urlcache'
    monkeypatch.setenv('HYPERTOOLS_URL_CACHE', str(path))
    assert url_cache_dir() == path
    return path


# ------------------------------------------------------------ cache layout

def test_cache_dir_defaults_under_the_hypertools_data_dir(monkeypatch):
    monkeypatch.delenv('HYPERTOOLS_URL_CACHE', raising=False)
    assert url_cache_dir() == DATA_DIR / 'urls'


def test_cached_path_is_url_specific_and_keeps_the_extension(cache_dir):
    path = cached_url_path(CSV_URL)
    assert path.parent == cache_dir
    assert path.suffix == '.csv'
    assert len(path.stem) == 64                       # sha256 hex digest
    assert cached_url_path(CSV_URL + '?v=2') != path  # query string counts


# --------------------------------------------------------------- downloads

def test_cache_true_downloads_once_then_serves_from_disk(cache_dir):
    path = cached_url_path(CSV_URL)
    assert not path.exists()

    with skip_on_transient_network(f'downloading {CSV_URL}'):
        first = load_source(CSV_URL, cache=True)
    assert isinstance(first, pd.DataFrame)
    assert first.shape[0] > 100
    assert path.is_file()
    assert path.stat().st_size > 1000

    # no leftover .part files: the write is atomic
    assert not [p for p in cache_dir.iterdir() if p.name.endswith('.part')]

    # a second load with the network unavailable still works, so it came
    # from disk rather than the wire
    second = load_source(CSV_URL, cache=True, offline=True)
    pd.testing.assert_frame_equal(first, second)


def test_cache_false_is_the_default_and_writes_nothing(cache_dir):
    with skip_on_transient_network(f'downloading {CSV_URL}'):
        data = hyp.load(CSV_URL)                      # public API, default
    assert isinstance(data, pd.DataFrame)
    assert not cached_url_path(CSV_URL).exists()
    assert not cache_dir.exists() or not any(cache_dir.iterdir())

    with skip_on_transient_network(f'downloading {CSV_URL}'):
        load_source(CSV_URL, cache=False)
    assert not cached_url_path(CSV_URL).exists()


# ----------------------------------------------------------------- offline

def test_offline_miss_raises_naming_the_cache_path(cache_dir):
    with pytest.raises(HypertoolsIOError) as excinfo:
        load_source(CSV_URL, offline=True)
    message = str(excinfo.value)
    assert str(cached_url_path(CSV_URL)) in message
    assert 'offline=True' in message
    assert 'cache=True' in message
    # the refusal escapes the "tried, in order" digest rather than being
    # listed as one failed guess among several
    assert 'Tried, in order' not in message
    assert isinstance(excinfo.value, HypertoolsOfflineError)
    assert isinstance(excinfo.value, HypertoolsIOError)


def test_offline_hit_never_touches_the_network(cache_dir):
    # populate the cache for a host that cannot resolve, then read it back
    payload = b'a,b\n1,2\n3,4\n'
    _write_cached(cached_url_path(UNREACHABLE_URL), payload, 'data.csv')
    df = load_source(UNREACHABLE_URL, offline=True)
    assert df.shape == (2, 2)
    assert list(df.columns) == ['a', 'b']

    # and without offline= (or cache=) the same URL genuinely fails
    with pytest.raises(HypertoolsIOError):
        load_source(UNREACHABLE_URL)


def test_offline_hit_is_used_for_cache_true_as_well(cache_dir):
    _write_cached(cached_url_path(UNREACHABLE_URL), b'a,b\n5,6\n', 'd.csv')
    df = load_source(UNREACHABLE_URL, cache=True)
    assert df.iloc[0].tolist() == [5, 6]


def test_atomic_write_leaves_no_partial_file(cache_dir):
    path = cached_url_path(UNREACHABLE_URL)
    _write_cached(path, b'a\n1\n', 'x.csv')
    names = sorted(p.name for p in cache_dir.iterdir())
    assert names == [path.name, f'{path.name}.meta.json']
    assert path.read_bytes() == b'a\n1\n'


def test_empty_cached_file_is_reported_not_silently_parsed(cache_dir):
    path = cached_url_path(UNREACHABLE_URL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')
    with pytest.raises(HypertoolsIOError, match='empty'):
        load_source(UNREACHABLE_URL, offline=True)


def test_name_hint_survives_the_cache_round_trip(cache_dir, tmp_path):
    # an extensionless URL: the stored hint is what tells the parser the
    # payload is CSV on the way back out of the cache
    url = 'https://hypertools-offline-test.invalid/download'
    _write_cached(cached_url_path(url), b'a,b\n7,8\n', 'table.csv')
    df = load_source(url, offline=True)
    assert df.shape == (1, 2)


# ------------------------------------------------------- keyword plumbing

def test_unexpected_kwargs_for_a_url_raise_typeerror(cache_dir):
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        load_source(CSV_URL, n_samples=10)


def test_cache_env_override_is_read_per_call(tmp_path, monkeypatch):
    monkeypatch.setenv('HYPERTOOLS_URL_CACHE', str(tmp_path / 'one'))
    assert url_cache_dir().name == 'one'
    monkeypatch.setenv('HYPERTOOLS_URL_CACHE', str(tmp_path / 'two'))
    assert url_cache_dir().name == 'two'
    assert os.environ['HYPERTOOLS_URL_CACHE'].endswith('two')
