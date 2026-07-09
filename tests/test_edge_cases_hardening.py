"""Assorted edge-case hardening (QC 2026-07 release hunt).

- empty (0-row) input crashed cryptically inside sklearn/PCA;
- predict(t<=0) silently returned an empty forecast; a float t was cryptic;
- a streaming reduce spec read only the legacy 'params' key, ignoring the
  canonical 'kwargs';
- the default text path emitted a scary sklearn version-mismatch warning from a
  known-safe pretrained model.

Real data, no mocks; headless (Agg).
"""
import matplotlib
matplotlib.use('Agg')
import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp


def _rng():
    return np.random.default_rng(0)


# --- empty input -------------------------------------------------------

def test_empty_input_clear_error():
    with pytest.raises(ValueError, match='no observations'):
        hyp.plot(np.zeros((0, 5)), show=False)


# --- predict horizon validation ----------------------------------------

@pytest.mark.parametrize('t', [-3, 0])
def test_predict_nonpositive_t_clear_error(t):
    df = pd.DataFrame({'a': np.arange(40.0)})
    with pytest.raises(ValueError, match='forecast horizon'):
        hyp.predict(df, model='GaussianProcess', t=t)


def test_predict_float_t_clear_error():
    df = pd.DataFrame({'a': np.arange(40.0)})
    with pytest.raises(ValueError, match='not a float'):
        hyp.predict(df, model='GaussianProcess', t=2.5)


def test_predict_datetime_horizon_still_works():
    idx = pd.date_range('2026-01-01', periods=40, freq='D')
    df = pd.DataFrame({'a': np.arange(40.0)}, index=idx)
    out = hyp.predict(df, model='GaussianProcess', t=idx[-1] + pd.Timedelta(days=3))
    assert len(out) == 3


def test_predict_positive_int_t_works():
    df = pd.DataFrame({'a': np.arange(40.0)})
    assert np.asarray(hyp.predict(df, model='GaussianProcess', t=5)).shape == (5, 1)


# --- streaming reduce spec honors canonical kwargs ---------------------

def test_streaming_reduce_spec_reads_canonical_kwargs():
    from hypertools.io import streaming
    import inspect
    src = inspect.getsource(streaming)
    # the streaming reduce-spec parser must consult the canonical 'kwargs' key
    assert "reduce.get('kwargs'" in src


# --- default text path is quiet about the known-safe pretrained model --

def test_default_text_path_no_version_warning():
    pytest.importorskip('sklearn')
    docs = ['the cat sat on the mat', 'dogs are loyal', 'stocks rose today',
            'interest rates rose', 'kittens love yarn', 'the fed raised rates']
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hyp.plot(docs, 'o', show=False)
    version_warnings = [w for w in caught if 'version' in str(w.message).lower()]
    assert len(version_warnings) == 0
