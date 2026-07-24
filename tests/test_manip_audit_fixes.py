# noinspection PyPackageRequirements
"""Regression tests for the 2026-07 release-1.0 audit findings against
hyp.manip() + the Normalize/ZScore/Smooth/Resample manipulators and
hyp.normalize() (unit F14-manip-normalize, plus cross-unit findings
D01-readme-001, X2-error-quality-004 [manip part], and X2-error-quality-007).

All tests use real data and real hypertools calls (no mocks).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.manip import Normalize, Smooth, Resample


# --- F14-001 / D01-readme-001: Smooth must be PER-DATASET on a list --------

def test_smooth_list_is_per_dataset_savgol():
    # a constant signal is invariant under smoothing, so any deviation at the
    # dataset boundary proves cross-dataset bleed (the audited bug: Smooth was
    # applied to the row-stacked list, mixing subjects' data).
    a = np.zeros((30, 3))
    b = np.ones((30, 3))
    sm = hyp.manip([a, b], model='Smooth', kernel_width=11)
    # exact under default maintain_bounds=True (clipped to each dataset's own
    # constant range); the buggy stacked behavior gave values up to ~0.4
    assert np.all(np.asarray(sm[0], dtype=float)[-3:] == 0.0)
    assert np.all(np.asarray(sm[1], dtype=float)[:3] == 1.0)
    # the entire datasets, not just the edges, must be untouched
    assert np.all(np.asarray(sm[0], dtype=float) == 0.0)
    assert np.all(np.asarray(sm[1], dtype=float) == 1.0)
    # without bounds-clipping, savgol on a constant is exact up to float eps
    # (the bug produced boundary deviations of ~0.4)
    sm = hyp.manip([a, b], model='Smooth', kernel_width=11,
                   maintain_bounds=False)
    assert np.allclose(np.asarray(sm[0], dtype=float), 0.0, atol=1e-12)
    assert np.allclose(np.asarray(sm[1], dtype=float), 1.0, atol=1e-12)


def test_smooth_list_is_per_dataset_gaussian_and_boxcar():
    a = np.zeros((30, 2))
    b = np.ones((30, 2))
    for kernel in ('gaussian', 'boxcar'):
        sm = hyp.manip([a, b], model='Smooth', kernel=kernel, kernel_width=11,
                       maintain_bounds=False)
        assert np.all(np.asarray(sm[0], dtype=float) == 0.0), kernel
        assert np.all(np.asarray(sm[1], dtype=float) == 1.0), kernel


def test_smooth_list_per_dataset_with_default_bounds():
    # README repro (D01-readme-001): dict spec, kernel_width=5, default
    # maintain_bounds=True
    a = np.zeros((30, 3))
    b = np.ones((30, 3))
    out = hyp.manip([a, b], model={'model': 'Smooth',
                                   'kwargs': {'kernel_width': 5}})
    assert np.all(np.asarray(out[0], dtype=float)[-1] == 0.0)
    assert np.all(np.asarray(out[1], dtype=float)[0] == 1.0)


# upstream: datawrangler calls pd.concat(copy=...), deprecated in pandas 4.
# the filter names the DeprecationWarning BASE class on purpose:
# pandas.errors.Pandas4Warning subclasses it but does not exist on
# pandas 2.x, and pytest aborts (UsageError, exit 4) on filter
# categories it cannot import
@pytest.mark.filterwarnings(
    'ignore:The copy keyword is deprecated:DeprecationWarning')
def test_smooth_multiindex_stacked_input_is_per_dataset():
    # pipelines pass stacked (multiindex) frames between stages; Smooth must
    # unstack and smooth each dataset independently
    import datawrangler as dw
    a = pd.DataFrame(np.zeros((30, 2)), columns=list('xy'))
    b = pd.DataFrame(np.ones((30, 2)), columns=list('xy'))
    stacked = dw.stack([a, b])
    out = Smooth(kernel_width=11, maintain_bounds=False).fit_transform(stacked)
    arr = np.asarray(out, dtype=float)
    # exact up to float eps (the buggy stacked behavior gave deviations ~0.4)
    assert np.allclose(arr[:30], 0.0, atol=1e-12)
    assert np.allclose(arr[30:], 1.0, atol=1e-12)


# --- F14-003: fitted Smooth reuse on data with different column labels -----

def test_smooth_fitted_reuse_different_column_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    _, ms = hyp.manip(X, model='Smooth', kernel_width=11, return_model=True)
    newdf = pd.DataFrame(rng.normal(size=(40, 3)),
                         columns=['roi1', 'roi2', 'roi3'])
    out = hyp.manip(newdf, model=ms)
    assert np.asarray(out).shape == (40, 3)
    assert np.isfinite(np.asarray(out, dtype=float)).all()


def test_smooth_fitted_reuse_different_column_count():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    _, ms = hyp.manip(X, model='Smooth', kernel_width=11, return_model=True)
    new4 = pd.DataFrame(rng.normal(size=(40, 4)), columns=list('wxyz'))
    # smoothing is column-independent, so a different column count is fine
    # once bounds are derived from the data being transformed
    out = hyp.manip(new4, model=ms)
    assert np.asarray(out).shape == (40, 4)


# --- F14-004: fitted Smooth reuse must not replay fit-time bounds ----------

def test_smooth_fitted_reuse_bounds_derive_from_new_data():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(size=(40, 3)), columns=list('abc'))
    _, ms = hyp.manip(df, model='Smooth', kernel_width=11, return_model=True)
    new = pd.DataFrame(rng.normal(loc=100.0, size=(30, 3)),
                       columns=list('abc'))
    out = np.asarray(hyp.manip(new, model=ms), dtype=float)
    # smoothed values must stay near 100 (the new data's own range), not be
    # clipped to the fit-time range (which was roughly [-3, 3])
    assert out.min() > 50.0
    assert abs(out.mean() - 100.0) < 5.0
    # clipping must still respect the NEW data's own per-column bounds
    for j, c in enumerate('abc'):
        assert out[:, j].max() <= np.asarray(new[c]).max() + 1e-12
        assert out[:, j].min() >= np.asarray(new[c]).min() - 1e-12


# --- F14-005: pandas Series input --------------------------------------------

def test_manip_series_input_treated_as_single_column():
    s = pd.Series(np.sin(np.linspace(0, 6, 40)), name='sig')
    out = hyp.manip(s, model='ZScore')
    arr = np.asarray(out, dtype=float)
    assert arr.shape == (40, 1)
    assert np.allclose(arr.mean(), 0.0, atol=1e-12)


def test_manip_list_of_series_input():
    s1 = pd.Series(np.linspace(0., 1., 25))
    s2 = pd.Series(np.linspace(1., 2., 25))
    out = hyp.manip([s1, s2], model='Smooth', kernel_width=5)
    assert isinstance(out, list) and len(out) == 2
    assert np.asarray(out[0]).shape == (25, 1)


# --- F14-006: single-row / 1-D ZScore must not silently return NaN ---------

def test_manip_zscore_single_row_returns_zeros_not_nan():
    out = np.asarray(hyp.manip(np.array([[1., 2., 3.]]), model='ZScore'),
                     dtype=float)
    assert out.shape == (1, 3)
    assert np.all(out == 0.0)


def test_manip_zscore_1d_input_returns_finite():
    out = np.asarray(hyp.manip(np.array([1., 2., 3., 4.]), model='ZScore'),
                     dtype=float)
    assert not np.isnan(out).any()


# --- F14-007 / X2-error-quality-004: empty & None inputs -------------------

def test_manip_none_input_clear_typeerror():
    with pytest.raises(TypeError, match='Unsupported data type'):
        hyp.manip(None, model='ZScore')


def test_manip_empty_dataframe_clear_error():
    with pytest.raises(ValueError, match='no observations'):
        hyp.manip(pd.DataFrame(), model='ZScore')


def test_manip_empty_array_clear_error():
    with pytest.raises(ValueError, match='no observations'):
        hyp.manip(np.zeros((0, 3)), model='ZScore')


def test_manip_empty_list_clear_error_no_corpus_load():
    with pytest.raises(ValueError, match='no observations'):
        hyp.manip([], model='ZScore')


# --- F14-008: non-integer kernel_width must actually ROUND ------------------

def test_smooth_noninteger_kernel_width_rounds_to_nearest():
    from scipy.signal import savgol_filter
    y = np.sin(np.linspace(0, 12, 60))
    df = pd.DataFrame({'y': y})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        out = hyp.manip(df, model='Smooth', kernel_width=11.7,
                        maintain_bounds=False)
    messages = [str(x.message) for x in w]
    assert any('Rounding' in m for m in messages)
    got = np.asarray(out, dtype=float).ravel()
    # 11.7 -> rounds to 12 -> even, bumped to 13 (NOT truncated to 11)
    assert np.allclose(got, savgol_filter(y, 13, 3))
    assert not np.allclose(got, savgol_filter(y, 11, 3))


# --- F14-009: documented ValueErrors must actually be ValueError ------------

def test_normalize_manipulator_min_ge_max_raises_valueerror():
    with pytest.raises(ValueError, match='min'):
        hyp.manip(np.random.default_rng(2).normal(size=(10, 2)),
                  model='Normalize', min=2, max=1)


def test_smooth_negative_kernel_width_raises_valueerror():
    with pytest.raises(ValueError, match='kernel_width'):
        hyp.manip(np.random.default_rng(3).normal(size=(20, 2)),
                  model='Smooth', kernel_width=-3)


def test_tools_normalize_invalid_mode_raises_valueerror_naming_param():
    with pytest.raises(ValueError, match='normalize'):
        hyp.normalize(np.random.default_rng(4).normal(size=(10, 2)),
                      normalize='banana')


# --- F14-010: dict spec with unknown model name -----------------------------

def test_manip_dict_spec_unknown_model_raises_valueerror():
    with pytest.raises(ValueError, match="unknown manip model 'Bar'"):
        hyp.manip(pd.DataFrame(np.random.default_rng(5).normal(size=(30, 3))),
                  model={'model': 'Bar', 'kwargs': {}})


# --- F14-011: invalid legacy mode= must not be silently accepted ------------

def test_smooth_invalid_legacy_mode_raises_valueerror():
    with pytest.raises(ValueError, match='mode'):
        hyp.manip(pd.DataFrame(np.random.default_rng(6).normal(size=(30, 3))),
                  model='Smooth', mode='hann')


def test_smooth_valid_legacy_modes_still_work():
    df = pd.DataFrame(np.random.default_rng(8).normal(size=(30, 3)))
    for mode in ('savgol', 'gaussian'):
        out = hyp.manip(df, model='Smooth', mode=mode)
        assert np.asarray(out).shape == (30, 3)


# --- F14-012: fitted axis=1 models must refuse NEW data ---------------------

def test_fitted_axis1_normalize_refuses_new_data():
    rng = np.random.default_rng(9)
    df = pd.DataFrame(rng.normal(size=(30, 3)), columns=list('abc'))
    _, m = hyp.manip(df, model='Normalize', axis=1, return_model=True)
    with pytest.raises(NotImplementedError, match='axis=1'):
        hyp.manip(pd.DataFrame(rng.normal(size=(30, 5))), model=m)


# upstream: datawrangler calls pd.concat(copy=...), deprecated in pandas 4.
# the filter names the DeprecationWarning BASE class on purpose:
# pandas.errors.Pandas4Warning subclasses it but does not exist on
# pandas 2.x, and pytest aborts (UsageError, exit 4) on filter
# categories it cannot import
@pytest.mark.filterwarnings(
    'ignore:The copy keyword is deprecated:DeprecationWarning')
def test_fitted_axis1_zscore_refuses_new_data_but_replays_fit_data():
    rng = np.random.default_rng(10)
    df = pd.DataFrame(rng.normal(size=(20, 4)))
    out, m = hyp.manip(df, model='ZScore', axis=1, return_model=True)
    # the fit_transform result itself is fine (row means ~ 0)
    assert np.allclose(np.asarray(out, dtype=float).mean(axis=1), 0.0,
                       atol=1e-12)
    # replaying the fit-time data (transform with no new data) still works
    replay = m.transform()
    assert np.asarray(replay).shape == (20, 4)
    with pytest.raises(NotImplementedError, match='axis=1'):
        hyp.manip(pd.DataFrame(rng.normal(size=(20, 4))), model=m)


def test_fitted_axis1_resample_reuse_still_allowed():
    # Resample re-derives everything from the data being transformed, so
    # axis=1 reuse is well-defined and must keep working (round17 semantics)
    rng = np.random.default_rng(11)
    df = pd.DataFrame(rng.normal(size=(4, 20)))
    _, r = hyp.manip(df, model='Resample', axis=1, n_samples=9,
                     return_model=True)
    out = hyp.manip(pd.DataFrame(rng.normal(size=(4, 20))), model=r)
    assert np.asarray(out).shape == (4, 9)


# --- X2-error-quality-007: Resample n_samples validation --------------------

@pytest.mark.parametrize('bad', [0, -10, 7.5, 10.0])
def test_resample_invalid_n_samples_raises_clear_valueerror(bad):
    X = np.random.default_rng(12).random((20, 4))
    with pytest.raises(ValueError, match='n_samples') as excinfo:
        hyp.manip(X, model='Resample', n_samples=bad)
    # the message must name the offending value
    assert repr(bad) in str(excinfo.value) or str(bad) in str(excinfo.value)


def test_resample_constructor_validates_n_samples_directly():
    with pytest.raises(ValueError, match='n_samples'):
        Resample(n_samples=0)


def test_resample_valid_n_samples_still_works():
    X = np.random.default_rng(13).random((20, 4))
    out = hyp.manip(X, model='Resample', n_samples=50)
    assert np.asarray(out).shape == (50, 4)


# --- F14-002: Resample axis=1 with string column labels ---------------------

def test_resample_axis1_string_columns_uses_positions():
    rng = np.random.default_rng(14)
    df = pd.DataFrame(rng.normal(size=(30, 3)), columns=list('abc'))
    out = hyp.manip(df, model='Resample', axis=1, n_samples=9)
    assert np.asarray(out).shape == (30, 9)


def test_resample_axis1_integer_columns_still_works():
    rng = np.random.default_rng(15)
    df = pd.DataFrame(rng.normal(size=(4, 20)))
    out = hyp.manip(df, model='Resample', axis=1, n_samples=9)
    assert np.asarray(out).shape == (4, 9)


# --- F14-018: raw scipy/numpy internals get hypertools context ---------------

def test_resample_duplicate_index_clear_error():
    dup = pd.DataFrame({'y': [1., 2., 3., 4.]}, index=[0, 1, 1, 2])
    with pytest.raises(ValueError, match='index'):
        hyp.manip(dup, model='Resample', n_samples=8)


def test_resample_decreasing_index_clear_error():
    dec = pd.DataFrame({'y': [1., 2., 3., 4.]}, index=[3, 2, 1, 0])
    with pytest.raises(ValueError, match='index'):
        hyp.manip(dec, model='Resample', n_samples=8)


def test_normalize_across_mismatched_columns_clear_error():
    rng = np.random.default_rng(16)
    a, b = rng.normal(size=(10, 3)), rng.normal(size=(10, 4))
    with pytest.raises(ValueError, match='column'):
        hyp.normalize([a, b], normalize='across')


# --- regression guards: previously-passing behavior must be unchanged -------

def test_smooth_single_dataset_unchanged_by_per_dataset_fix():
    from scipy.signal import savgol_filter
    y = np.sin(np.linspace(0, 12, 60))
    df = pd.DataFrame({'y': y})
    out = np.asarray(hyp.manip(df, model='Smooth', kernel_width=11,
                               maintain_bounds=False), dtype=float).ravel()
    assert np.allclose(out, savgol_filter(y, 11, 3))


def test_smooth_maintain_bounds_still_clips_to_own_range():
    step = np.r_[np.zeros(30), np.ones(30)]
    df = pd.DataFrame({'y': step})
    out = np.asarray(hyp.manip(df, model='Smooth', kernel_width=11),
                     dtype=float)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_zscore_list_shared_stats_unchanged():
    # documented semantics (F14-013): ZScore fits ONE shared mean/std across
    # a list of datasets
    z = hyp.manip([np.zeros((30, 2)), np.ones((30, 2))], model='ZScore')
    v = np.asarray(z[0], dtype=float)[0, 0]
    assert np.isclose(v, -0.99163165, atol=1e-6)


def test_smooth_bare_array_input_still_works():
    # hypertools.Pipeline passes bare arrays between steps; the old
    # apply_stacked decorator wrangled these implicitly, so the per-dataset
    # rewrite must keep accepting them
    out = Smooth(kernel_width=5).fit_transform(
        np.random.default_rng(18).normal(size=(20, 3)))
    assert np.asarray(out).shape == (20, 3)


def test_smooth_axis1_still_works():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(rng.normal(size=(3, 40)))
    out = hyp.manip(df, model='Smooth', axis=1, kernel_width=11)
    assert np.asarray(out).shape == (3, 40)
