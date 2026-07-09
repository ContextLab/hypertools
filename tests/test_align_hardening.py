"""align robustness (QC 2026-07 release hunt).

- hyp.align crashed by default on datasets with DIFFERENT column counts -- the
  canonical hyperalignment case (each subject padded to a common width) -- from a
  vstack in format_data's NaN check;
- align silently trimmed datasets to their common rows (data loss, no warning).

Real data, no mocks.
"""
import warnings

import numpy as np
import pytest

import hypertools as hyp


def _rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize('model', ['HyperAlign', 'Procrustes',
                                   'SharedResponseModel'])
def test_align_mismatched_columns_pads_not_crashes(model):
    a = _rng().normal(size=(50, 5))
    b = _rng().normal(size=(50, 7))
    out = hyp.align([a, b], model=model)
    assert len(out) == 2
    assert all(np.asarray(o).shape == (50, 7) for o in out)  # padded to common width


def test_align_same_columns_regression():
    a = _rng().normal(size=(40, 5))
    b = _rng().normal(size=(40, 5))
    out = hyp.align([a, b], model='HyperAlign')
    assert all(np.asarray(o).shape == (40, 5) for o in out)


def test_align_mismatched_rows_warns_about_trimming():
    a = _rng().normal(size=(50, 5))
    b = _rng().normal(size=(40, 5))
    with pytest.warns(UserWarning, match='keeps only'):
        out = hyp.align([a, b], model='HyperAlign')
    assert all(np.asarray(o).shape[0] == 40 for o in out)


def test_align_matching_rows_no_trim_warning():
    a = _rng().normal(size=(40, 5))
    b = _rng().normal(size=(40, 5))
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any "keeps only" warning would fail
        try:
            hyp.align([a, b], model='HyperAlign')
        except UserWarning as e:  # pragma: no cover
            assert 'keeps only' not in str(e)
