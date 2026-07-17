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

    def fake_once(path):
        path.write_bytes(b'<html>rate limited</html>')
    monkeypatch.setattr(L, '_download_example_data_once', fake_once)

    with pytest.raises(HypertoolsIOError, match='checksum'):
        L._download_example_data(tmp_path / 'spiral', max_attempts=2)
