# -*- coding: utf-8 -*-
"""Regression tests for the 2026-07 release audit findings on hyp.save
(unit F20-save). Real files, real serialization -- no mocks."""

import os
import pickle

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp
from hypertools._shared.exceptions import HypertoolsIOError


class _Unpicklable:
    """Holds an open file handle, which pickle cannot serialize."""

    def __init__(self, path):
        self.handle = open(path, 'rb')

    def close(self):
        self.handle.close()


# --------------------------------------------------------------- F20-001
# a failed save must never destroy the pre-existing file at that path,
# and must not leave partial files behind.

def test_failed_save_preserves_existing_file(tmp_path):
    target = tmp_path / 'precious.pkl'
    good = np.arange(1000, dtype=float)
    hyp.save(good, str(target))
    original_bytes = target.read_bytes()

    bad = _Unpicklable(str(target))
    try:
        with pytest.raises(HypertoolsIOError):
            hyp.save({'results': good, 'log': bad.handle}, str(target))
    finally:
        bad.close()

    assert target.read_bytes() == original_bytes, \
        'failed save destroyed the pre-existing file'
    np.testing.assert_allclose(hyp.load(str(target)), good)


def test_failed_save_leaves_no_partial_file(tmp_path):
    target = tmp_path / 'brand_new.pkl'
    bad = _Unpicklable(__file__)
    try:
        with pytest.raises(HypertoolsIOError):
            hyp.save({'log': bad.handle}, str(target))
    finally:
        bad.close()
    assert not target.exists(), 'failed save left a partial file behind'
    leftovers = [p for p in tmp_path.iterdir()]
    assert leftovers == [], f'failed save left temp files: {leftovers}'


# ----------------------------------------------- F20-002 / F20-010 / F19-012
# extension-aware export: files named .csv/.npy/... must be readable by
# their nominal tools (and still round-trip through hyp.load).

def test_save_csv_is_real_csv(tmp_path):
    df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    path = tmp_path / 'data.csv'
    hyp.save(df, str(path))
    external = pd.read_csv(path)          # crashed (pickle bytes) before
    pd.testing.assert_frame_equal(external, df)
    roundtrip = hyp.load(str(path))
    pd.testing.assert_frame_equal(roundtrip, df)


def test_save_tsv_is_real_tsv(tmp_path):
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    path = tmp_path / 'data.tsv'
    hyp.save(df, str(path))
    external = pd.read_csv(path, sep='\t')
    pd.testing.assert_frame_equal(external, df)


def test_save_npy_is_real_npy(tmp_path):
    arr = np.arange(12, dtype=float).reshape(4, 3)
    path = tmp_path / 'data.npy'
    hyp.save(arr, str(path))
    np.testing.assert_allclose(np.load(path), arr)   # crashed before
    np.testing.assert_allclose(hyp.load(str(path)), arr)


def test_save_npz_list_of_arrays(tmp_path):
    arrays = [np.arange(6.).reshape(2, 3), np.ones((3, 2))]
    path = tmp_path / 'data.npz'
    hyp.save(arrays, str(path))
    z = np.load(path)
    assert len(z.files) == 2
    out = hyp.load(str(path))
    assert isinstance(out, list) and len(out) == 2
    np.testing.assert_allclose(out[0], arrays[0])
    np.testing.assert_allclose(out[1], arrays[1])


def test_save_json_parquet_xlsx_mat_roundtrip(tmp_path):
    df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    for ext in ('.json', '.parquet', '.xlsx'):
        path = tmp_path / f'data{ext}'
        hyp.save(df, str(path))
        out = hyp.load(str(path))
        np.testing.assert_allclose(np.asarray(out, dtype=float),
                                   df.to_numpy())
    arr = np.arange(6, dtype=float).reshape(2, 3)
    mat_path = tmp_path / 'data.mat'
    hyp.save(arr, str(mat_path))
    from scipy.io import loadmat
    np.testing.assert_allclose(loadmat(mat_path)['data'], arr)
    np.testing.assert_allclose(hyp.load(str(mat_path)), arr)


def test_save_pickle_extensions_still_pickle(tmp_path):
    arr = np.arange(5, dtype=float)
    for ext in ('.pkl', '.pickle', '.p', '.geo'):
        path = tmp_path / f'data{ext}'
        hyp.save(arr, str(path))
        assert path.read_bytes()[:1] == b'\x80'
        np.testing.assert_allclose(hyp.load(str(path)), arr)


def test_save_unknown_extension_defaults_to_pickle(tmp_path):
    arr = np.arange(5, dtype=float)
    path = tmp_path / 'data.xyz'
    hyp.save(arr, str(path))
    assert path.read_bytes()[:1] == b'\x80'
    np.testing.assert_allclose(hyp.load(str(path)), arr)


def test_save_object_incompatible_with_format_raises(tmp_path):
    nested = {'a': np.arange(3), 'b': 'text'}
    with pytest.raises(HypertoolsIOError, match=r'\.csv'):
        hyp.save(nested, str(tmp_path / 'bundle.csv'))


# --------------------------------------------------------------- F20-005
# kwargs must not be silently swallowed; protocol= is honored.

def test_save_unknown_kwarg_raises_typeerror(tmp_path):
    with pytest.raises(TypeError):
        hyp.save(np.arange(3), str(tmp_path / 'x.pkl'), banana=42)


def test_save_protocol_honored(tmp_path):
    path = tmp_path / 'proto2.pkl'
    hyp.save(np.arange(3), str(path), protocol=2)
    assert path.read_bytes()[:2] == b'\x80\x02'


def test_save_protocol_rejected_for_non_pickle_format(tmp_path):
    with pytest.raises(ValueError, match='protocol'):
        hyp.save(pd.DataFrame({'a': [1]}), str(tmp_path / 'x.csv'),
                 protocol=2)


# --------------------------------------------------------------- F20-006
# save() must expand ~ and $ENVVARS like load() does.

def test_save_expands_tilde(tmp_path, monkeypatch):
    # both variables: POSIX expanduser reads HOME, Windows prefers
    # USERPROFILE -- setting only HOME would silently write into the real
    # Windows profile directory and fail the assert below
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    arr = np.arange(4, dtype=float)
    hyp.save(arr, '~/tilde_test.pkl')
    saved = tmp_path / 'tilde_test.pkl'
    assert saved.exists()
    np.testing.assert_allclose(hyp.load('~/tilde_test.pkl'), arr)


def test_save_expands_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv('HYP_SAVE_TEST_DIR', str(tmp_path))
    arr = np.arange(4, dtype=float)
    hyp.save(arr, '$HYP_SAVE_TEST_DIR/envvar_test.pkl')
    assert (tmp_path / 'envvar_test.pkl').exists()
    np.testing.assert_allclose(
        hyp.load('$HYP_SAVE_TEST_DIR/envvar_test.pkl'), arr)


# --------------------------------------------------------------- F20-009
# save failures must be hypertools-branded and actionable.

def test_save_missing_parent_dir_friendly_error(tmp_path):
    with pytest.raises(HypertoolsIOError, match='does not exist'):
        hyp.save(np.arange(3), str(tmp_path / 'no' / 'such' / 'dir' / 'f.pkl'))


def test_save_directory_as_fname_friendly_error(tmp_path):
    with pytest.raises(HypertoolsIOError, match='directory'):
        hyp.save(np.arange(3), str(tmp_path))


def test_save_swapped_args_friendly_error(tmp_path):
    with pytest.raises(TypeError, match=r'save\(obj, fname\)'):
        hyp.save(str(tmp_path / 'f.pkl'), np.arange(3))


def test_save_unpicklable_object_friendly_error(tmp_path):
    bad = _Unpicklable(__file__)
    try:
        with pytest.raises(HypertoolsIOError, match='pickle'):
            hyp.save(bad, str(tmp_path / 'f.pkl'))
    finally:
        bad.close()


# --------------------------------------------------------------- F20-003
# saving the result of hyp.plot(animate=True) must explain the supported
# path instead of leaking a pickle-internal AttributeError.

def test_save_animation_result_friendly_error(tmp_path):
    data = np.random.default_rng(5).standard_normal((30, 3)).cumsum(axis=0)
    result = hyp.plot(data, animate=True, show=False)
    with pytest.raises(HypertoolsIOError, match='save_path'):
        hyp.save(result, str(tmp_path / 'anim.pkl'))
    assert not (tmp_path / 'anim.pkl').exists()
    plt.close('all')


# --------------------------------------------------------------- F20-008
# public docs: full numpydoc docstring.

def test_save_docstring_has_numpydoc_sections():
    doc = hyp.save.__doc__
    for section in ('Parameters', 'Returns', 'Examples'):
        assert section in doc, f'save docstring missing {section} section'
    assert 'pickle' in doc.lower()


# ---------------------------------------------------------------- misc
# atomic overwrite: a successful re-save replaces the file completely.

def test_successful_overwrite_replaces_file(tmp_path):
    path = tmp_path / 'x.pkl'
    hyp.save(np.arange(10, dtype=float), str(path))
    hyp.save(np.arange(3, dtype=float), str(path))
    np.testing.assert_allclose(hyp.load(str(path)), np.arange(3, dtype=float))


def test_save_accepts_pathlib_path(tmp_path):
    arr = np.arange(3, dtype=float)
    hyp.save(arr, tmp_path / 'p.pkl')
    np.testing.assert_allclose(hyp.load(str(tmp_path / 'p.pkl')), arr)


def test_save_returns_none(tmp_path):
    assert hyp.save(np.arange(3), str(tmp_path / 'r.pkl')) is None
