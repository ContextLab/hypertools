# -*- coding: utf-8 -*-
"""Built-in dataset integrity (2026-07 release review, blocker #1).

Every built-in is verified against a hard-coded SHA-256 BEFORE it is
deserialized; a mismatch (corrupt/rate-limited download, poisoned cache,
changed upstream file) is a HARD error, never a silent redownload-and-
reparse, and every cache hit is validated. Real files; the "download" is
stubbed to fixed bytes so these run offline and fast (no network).
"""
import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import importlib

import hypertools as hyp
from hypertools.core.exceptions import HypertoolsIOError

# hypertools.io.load the MODULE (the `load` function shadows the submodule
# name under hypertools.io, so import it explicitly)
L = importlib.import_module('hypertools.io.load')


def test_all_downloadable_builtins_are_pinned():
    # every registry entry that is actually downloaded must have a pinned
    # hash (sotus is excluded -- it loads via datawrangler, not the registry)
    downloadable = {n for n in L.EXAMPLE_DATA
                    if not str(L.EXAMPLE_DATA[n]).startswith('datawrangler')}
    unpinned = downloadable - set(L._EXAMPLE_DATA_SHA256)
    assert not unpinned, f'unpinned downloadable datasets: {sorted(unpinned)}'


def test_spiral_loads_and_passes_integrity():
    # real cached file -> integrity verified -> deserialized
    data = hyp.load('spiral')
    assert isinstance(data, list) and np.asarray(data[0]).shape[1] == 3
    assert L._integrity_ok(L.DATA_DIR.joinpath('spiral'), 'spiral')


def test_integrity_ok_detects_mismatch(tmp_path, monkeypatch):
    f = tmp_path / 'x'
    f.write_bytes(b'definitely not the real bytes')
    monkeypatch.setitem(L._EXAMPLE_DATA_SHA256, 'x', 'deadbeef' * 8)
    assert L._integrity_ok(f, 'x') is False
    # a name with no pin is not gated
    assert L._integrity_ok(f, 'not-pinned') is True


def test_load_hard_fails_on_integrity_mismatch(tmp_path, monkeypatch):
    # a download whose bytes do not match the pin must raise a hard integrity
    # error -- NOT be unpickled, and NOT loop forever reparsing
    monkeypatch.setattr(L, 'DATA_DIR', tmp_path)

    def fake_download(path):
        path.write_bytes(b'not the real spiral file')  # wrong hash
    monkeypatch.setattr(L, '_download_example_data', fake_download)

    with pytest.raises(HypertoolsIOError, match='integrity check failed'):
        hyp.load('spiral')
    # the unverified file must not be left cached
    assert not tmp_path.joinpath('spiral').exists()


def test_corrupt_cache_hit_is_revalidated_then_hard_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(L, 'DATA_DIR', tmp_path)
    tmp_path.joinpath('spiral').write_bytes(b'stale corrupt cache')
    calls = []

    def fake_download(path):
        calls.append(1)
        path.write_bytes(b'still not the real file')
    monkeypatch.setattr(L, '_download_example_data', fake_download)

    with pytest.raises(HypertoolsIOError, match='integrity check failed'):
        hyp.load('spiral')
    assert calls, 'a corrupt cache hit must trigger a fresh re-download'


def test_download_loop_rejects_wrong_checksum_bytes(tmp_path, monkeypatch):
    # _download_example_data must treat checksum-mismatched bytes (e.g. a
    # rate-limit HTML page served with status 200) as a failed attempt
    monkeypatch.setattr(L, 'DATA_DIR', tmp_path)
    monkeypatch.setattr('time.sleep', lambda *_: None)

    def fake_once(path, name):
        path.write_bytes(b'<html>rate limited</html>')
    monkeypatch.setattr(L, '_download_example_data_once', fake_once)

    with pytest.raises(HypertoolsIOError, match='checksum'):
        L._download_example_data(tmp_path / 'spiral', max_attempts=2)


def test_rehosted_npz_never_unpickles(tmp_path):
    # the re-hosted DATA datasets are read with allow_pickle=False, so a
    # file containing an object array (which would need pickle to load) is
    # REFUSED, not executed -- proving the non-executable guarantee
    import numpy as np
    npz = tmp_path / 'x.npz'
    np.savez(npz, arr_0=np.array([{'a': 1}], dtype=object))
    plain = tmp_path / 'bunny'          # cache path has no extension
    npz.rename(plain)
    with pytest.raises(ValueError, match='allow_pickle'):
        L._parse_rehosted(plain, 'bunny')   # bunny -> npz_array


def test_rehosted_datasets_are_not_on_the_pickle_path():
    # every re-hosted DATA dataset must be handled by _parse_rehosted, never
    # by the pickle branch (regression guard for the 2026-07 re-hosting)
    assert set(L._REHOSTED) <= set(L._EXAMPLE_DATA_SHA256)
    for name, url in L.EXAMPLE_DATA.items():
        if name in L._REHOSTED:
            assert str(url).startswith('http'), f'{name} should be a URL'


# ------------------------------------------------------- atomic downloads
# finding #6 (2026-07 review): downloads must be atomic and concurrency-safe
# -- a reader never sees a partial/unverified cache file, a failed download
# leaves no corrupt file, and racing processes can't clobber one another.

def _payload_and_sha():
    import hashlib
    import io
    buf = io.BytesIO()
    np.savez(buf, arr_0=np.arange(6.0).reshape(3, 2))
    b = buf.getvalue()
    return b, hashlib.sha256(b).hexdigest()


def test_download_writes_to_temp_then_atomically_replaces(tmp_path, monkeypatch):
    # the bytes are written to a PRIVATE temp file, verified, and only then
    # os.replace'd into the final path -- the final cache path is never the
    # write target, so a concurrent reader never sees a half-written file
    monkeypatch.setattr(L, 'DATA_DIR', tmp_path)
    final = tmp_path / 'spiral'
    payload, sha = _payload_and_sha()
    monkeypatch.setitem(L._EXAMPLE_DATA_SHA256, 'spiral', sha)
    saw = []

    def fake_once(dest, name):
        # download target is a temp file, and `final` does not exist yet
        saw.append((dest != final, not final.exists()))
        dest.write_bytes(payload)
    monkeypatch.setattr(L, '_download_example_data_once', fake_once)

    L._download_example_data(final)

    assert final.is_file() and L._integrity_ok(final, 'spiral')
    assert saw == [(True, True)]                        # wrote to temp
    assert not list(tmp_path.glob('.spiral.*.part'))   # temp cleaned up


def test_failed_download_leaves_no_partial_cache_file(tmp_path, monkeypatch):
    # a download whose bytes never match the pin must leave NO file at the
    # final cache path and NO leftover temp files
    monkeypatch.setattr(L, 'DATA_DIR', tmp_path)
    monkeypatch.setattr('time.sleep', lambda *_: None)
    final = tmp_path / 'spiral'
    _, sha = _payload_and_sha()
    monkeypatch.setitem(L._EXAMPLE_DATA_SHA256, 'spiral', sha)

    def fake_once(dest, name):
        dest.write_bytes(b'corrupt bytes that do not match the pin')
    monkeypatch.setattr(L, '_download_example_data_once', fake_once)

    with pytest.raises(HypertoolsIOError, match='checksum'):
        L._download_example_data(final, max_attempts=2)
    assert not final.exists()
    assert not list(tmp_path.glob('.spiral.*.part'))


def test_concurrent_downloads_are_consistent_and_leak_no_temp_files(
        tmp_path, monkeypatch):
    # many workers racing to fetch the same dataset all end up with the one
    # correct, fully-verified file -- never a corrupt or partial one. Real
    # threads + real files; the "download" copies valid bytes in with a small
    # delay to widen the race window.
    import threading
    import time as _time
    monkeypatch.setattr(L, 'DATA_DIR', tmp_path)
    final = tmp_path / 'spiral'
    payload, sha = _payload_and_sha()
    monkeypatch.setitem(L._EXAMPLE_DATA_SHA256, 'spiral', sha)

    def fake_once(dest, name):
        _time.sleep(0.05)
        dest.write_bytes(payload)
    monkeypatch.setattr(L, '_download_example_data_once', fake_once)

    errors = []

    def worker():
        try:
            L._download_example_data(final)
        except Exception as e:                     # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert final.is_file() and L._integrity_ok(final, 'spiral')
    assert final.read_bytes() == payload
    assert not list(tmp_path.glob('.spiral.*.part'))
