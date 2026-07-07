"""Tests for round17 Task 5: `hyp.manip` list-chaining via `Pipeline`,
interleaved reduce/align/cluster steps inside a manip list (GH #153),
`return_model=`/legacy-dict/fitted-reuse support, and `Smooth`'s new
`kernel=` options (GH #274). All data is real (small) numeric arrays/
DataFrames -- no mocks.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import hypertools as hyp
from hypertools.core.pipeline import Pipeline
from hypertools.manip.manip import manip
from hypertools.manip.smooth import Smooth
from hypertools.manip.resample import Resample
from hypertools.manip.zscore import ZScore


def _rng():
    return np.random.RandomState(0)


def _wiggly(n=300, ncols=3, seed=0):
    """A noisy random-walk trajectory -- has real local jaggedness for the
    smoothing tests to reduce."""
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.randn(n, ncols), axis=0) + rng.randn(n, ncols) * 3


def _roughness(a):
    """Mean absolute second difference -- a real (non-tautological) measure
    of local jaggedness; smoothing should reduce it."""
    a = np.asarray(a, dtype=float)
    return np.mean(np.abs(np.diff(a, n=2, axis=0)))


def _mean_step_displacement(a):
    a = np.asarray(a, dtype=float)
    return np.mean(np.linalg.norm(np.diff(a, axis=0), axis=1))


# --- Jeremy's exact #274/#275 manip spec --------------------------------

def test_jeremy_manip_spec_end_to_end():
    """`manip=[{'model': 'Smooth', ...}, {'model': 'Resample', ...},
    'ZScore']` (verbatim from the round17 plan) must smooth, resample to
    1000 rows, then zscore -- for each dataset independently."""
    raw = [_wiggly(seed=0), _wiggly(seed=1)]
    manip_spec = [
        {'model': 'Smooth', 'args': [], 'kwargs': {'kernel_width': 25}},
        {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 1000}},
        'ZScore',
    ]

    out = hyp.manip(raw, model=manip_spec)
    assert isinstance(out, list) and len(out) == 2

    for o, r in zip(out, raw):
        arr = np.asarray(o)
        # resampled to exactly 1000 rows, same number of columns as input
        assert arr.shape == (1000, r.shape[1])

    # ZScore is fit JOINTLY across all datasets in the list (matching
    # `hypertools.manip.zscore.fitter`'s pooling behavior for list input),
    # so it's the POOLED (stacked-across-datasets) column mean/std that is
    # ~0/~1, not each dataset's individually.
    pooled = np.concatenate([np.asarray(o) for o in out], axis=0)
    np.testing.assert_allclose(pooled.mean(axis=0), 0, atol=1e-8)
    # ddof=0 (population) vs ddof=1 (sample, used to fit ZScore) differ
    # slightly at finite n -- loose tolerance, not an exactness claim.
    np.testing.assert_allclose(pooled.std(axis=0), 1, atol=1e-3)


def test_jeremy_manip_spec_smoothing_reduces_step_displacement():
    """Isolate smoothing's effect (before the zscore step rescales
    everything): a Smooth->Resample chain should have smaller mean
    per-step displacement than a Resample-only chain on the same raw
    trajectory."""
    raw = _wiggly(seed=2)

    resample_only = hyp.manip(raw, model=[{'model': 'Resample', 'kwargs': {'n_samples': 1000}}])
    smoothed_then_resampled = hyp.manip(
        raw, model=[{'model': 'Smooth', 'args': [], 'kwargs': {'kernel_width': 25}},
                    {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 1000}}])

    raw_disp = _mean_step_displacement(resample_only)
    smoothed_disp = _mean_step_displacement(smoothed_then_resampled)
    assert smoothed_disp < raw_disp


# --- #153: reduce/align/cluster names resolve inside a manip list -------

def test_manip_list_resolves_reduce_name_umap():
    x = _wiggly(n=30, ncols=5, seed=3)
    out = hyp.manip(x, model=['Smooth', 'UMAP'])
    arr = np.asarray(out)
    assert arr.shape == (30, 2)  # UMAP's default ndims
    assert np.all(np.isfinite(arr))


def test_manip_list_resolves_align_name_hyperalign():
    d1 = _wiggly(n=25, ncols=4, seed=4)
    d2 = _wiggly(n=25, ncols=4, seed=5)
    out = hyp.manip([d1, d2], model=['Smooth', 'HyperAlign'])
    assert isinstance(out, list) and len(out) == 2
    for o, d in zip(out, [d1, d2]):
        arr = np.asarray(o)
        assert arr.shape == d.shape
        assert np.all(np.isfinite(arr))


def test_manip_list_priority_manipulators_before_reduce_align_cluster():
    """Documented resolution order (#153): MANIPULATORS -> REDUCERS ->
    ALIGNERS -> CLUSTERERS. A manip-registry name ('ZScore') must resolve
    to the Manipulator, not accidentally collide with another registry."""
    pipe = Pipeline(['ZScore'])
    assert isinstance(pipe.steps[0][1], ZScore)


# --- return_model / legacy dict / fitted reuse ---------------------------

def test_return_model_false_by_default_single_step():
    out = manip(_rng().rand(20, 3), model='ZScore')
    assert not isinstance(out, tuple)


def test_return_model_true_single_step_returns_fitted_manipulator():
    out, fitted = manip(_rng().rand(20, 3), model='ZScore', return_model=True)
    assert isinstance(fitted, ZScore)
    assert fitted.is_fitted


def test_return_model_true_list_returns_fitted_pipeline():
    out, fitted = manip(_rng().rand(20, 3), model=['ZScore', 'Normalize'], return_model=True)
    assert isinstance(fitted, Pipeline)
    assert fitted.is_fitted
    assert [name for name, _ in fitted.steps] == ['zscore', 'normalize']


def test_legacy_params_dict_warns_and_matches_canonical_form():
    x = _rng().rand(30, 3) * 10
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        legacy_out = manip(x, model={'model': 'Smooth', 'params': {'kernel_width': 11}})
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    canonical_out = manip(x, model={'model': 'Smooth', 'args': [], 'kwargs': {'kernel_width': 11}})
    np.testing.assert_array_equal(np.asarray(legacy_out), np.asarray(canonical_out))


def test_canonical_dict_class_and_instance_forms_agree():
    x = _rng().rand(20, 3)

    out_str = manip(x, model='ZScore')
    out_dict = manip(x, model={'model': 'ZScore', 'args': [], 'kwargs': {}})
    out_class = manip(x, model=ZScore)
    out_instance = manip(x, model=ZScore())

    for o in (out_dict, out_class, out_instance):
        np.testing.assert_array_equal(np.asarray(out_str), np.asarray(o))


def test_unknown_model_name_raises_value_error():
    with pytest.raises(ValueError, match='unknown manip model'):
        manip(_rng().rand(10, 2), model='NotARealManipulator')


def test_fitted_manipulator_reused_via_transform_not_refit():
    """Poison pill: fit ZScore on A, hand the fitted instance back in as
    `model=` on DIFFERENT data B. The result must be B transformed with
    A's fitted mean/std (transform/reuse) -- NOT B refit from scratch
    (which would use B's own mean/std and give a different answer)."""
    A = _rng().rand(50, 3) * 10 + 5
    B = np.random.RandomState(1).rand(50, 3) * 2 - 100

    _, fitted_on_a = manip(A, model='ZScore', return_model=True)
    reused = manip(B, model=fitted_on_a)

    expected = (B - fitted_on_a.mean.values) / fitted_on_a.std.values
    np.testing.assert_allclose(np.asarray(reused), expected)

    refit_on_b = manip(B, model='ZScore')
    assert not np.allclose(np.asarray(reused), np.asarray(refit_on_b))


def test_fitted_pipeline_reused_via_transform_not_refit():
    A = _rng().rand(50, 3) * 10 + 5
    B = np.random.RandomState(2).rand(50, 3) * 2 - 100

    _, fitted_pipeline = manip(A, model=['ZScore'], return_model=True)
    reused = manip(B, model=fitted_pipeline)

    fitted_zscore = fitted_pipeline.steps[0][1]
    expected = (B - fitted_zscore.mean.values) / fitted_zscore.std.values
    np.testing.assert_allclose(np.asarray(reused), expected)

    refit_on_b = manip(B, model=['ZScore'])
    assert not np.allclose(np.asarray(reused), np.asarray(refit_on_b))


def test_transform_before_fit_raises_not_fitted():
    with pytest.raises(NotFittedError):
        ZScore().transform(_rng().rand(5, 2))


# --- Smooth kernel= options (GH #274/#153) --------------------------------

def test_smooth_kernel_default_is_savgol_and_matches_explicit_savgol():
    df = pd.DataFrame(_wiggly(seed=6))
    default_out = Smooth(kernel_width=25).fit_transform(df)
    explicit_out = Smooth(kernel='savgol', kernel_width=25).fit_transform(df)
    pd.testing.assert_frame_equal(default_out, explicit_out)


def test_smooth_savgol_regression_byte_identical_to_prior_behavior():
    """Regression guard: the (pre-round17) savgol code path -- direct
    `scipy.signal.savgol_filter` per column -- must be untouched."""
    from scipy.signal import savgol_filter
    df = pd.DataFrame(_wiggly(seed=7))
    out = Smooth(kernel_width=21, order=3, maintain_bounds=False).fit_transform(df)
    for c in df.columns:
        expected = savgol_filter(df[c].values, 21, 3)
        np.testing.assert_array_equal(out[c].values, expected)


def test_smooth_gaussian_and_boxcar_differ_from_savgol_and_reduce_roughness():
    raw = _wiggly(seed=8)
    df = pd.DataFrame(raw)
    raw_roughness = _roughness(raw)

    savgol_out = Smooth(kernel='savgol', kernel_width=25).fit_transform(df)
    gaussian_out = Smooth(kernel='gaussian', kernel_width=25).fit_transform(df)
    boxcar_out = Smooth(kernel='boxcar', kernel_width=25).fit_transform(df)

    assert not np.allclose(gaussian_out.values, savgol_out.values)
    assert not np.allclose(boxcar_out.values, savgol_out.values)
    assert not np.allclose(gaussian_out.values, boxcar_out.values)

    for out in (savgol_out, gaussian_out, boxcar_out):
        assert _roughness(out.values) < raw_roughness


def test_smooth_invalid_kernel_raises_value_error_listing_options():
    df = pd.DataFrame(_wiggly(seed=9))
    with pytest.raises(ValueError, match=r"savgol.*gaussian.*boxcar|kernel"):
        Smooth(kernel='not-a-real-kernel', kernel_width=11).fit_transform(df)


def test_smooth_legacy_mode_gaussian_var_unchanged_when_kernel_left_default():
    """Backward compat: `mode='gaussian'`/`var=` (added pre-round17 for the
    weights-trajectory recipe) keeps its own sigma=sqrt(var) behavior when
    `kernel=` is left at its default."""
    from scipy.ndimage import gaussian_filter1d
    df = pd.DataFrame(_wiggly(seed=10))
    out = Smooth(mode='gaussian', var=300, maintain_bounds=False).fit_transform(df)
    for c in df.columns:
        expected = gaussian_filter1d(np.asarray(df[c], dtype=float), sigma=np.sqrt(300))
        np.testing.assert_allclose(out[c].values, expected)


def test_manip_dispatcher_threads_smooth_kernel_kwarg():
    x = _wiggly(seed=11)
    out = manip(x, model='Smooth', kernel='gaussian', kernel_width=25)
    assert _roughness(np.asarray(out)) < _roughness(x)


def test_smooth_explicit_kernel_savgol_wins_over_mode_gaussian():
    """Round17 fix wave 1, finding 2 (GH #274/#153 regression guard): an
    EXPLICIT `kernel='savgol'` must take precedence over `mode='gaussian'`
    -- before the sentinel-default fix, `kernel='savgol'` was
    indistinguishable from "left at default" and silently fell through to
    the legacy gaussian path."""
    from scipy.signal import savgol_filter
    df = pd.DataFrame(_wiggly(seed=12))
    out = Smooth(kernel='savgol', mode='gaussian', var=300,
                 kernel_width=25, maintain_bounds=False).fit_transform(df)
    for c in df.columns:
        expected = savgol_filter(df[c].values, 25, 3)
        np.testing.assert_array_equal(out[c].values, expected)


def test_smooth_kernel_unspecified_mode_gaussian_unchanged():
    """Companion to the above: with `kernel` left UNSPECIFIED (the new
    `None` sentinel default) and `mode='gaussian'` explicit, the legacy
    gaussian (sigma=sqrt(var)) path must still apply -- unchanged."""
    from scipy.ndimage import gaussian_filter1d
    df = pd.DataFrame(_wiggly(seed=13))
    out = Smooth(mode='gaussian', var=300, maintain_bounds=False).fit_transform(df)
    for c in df.columns:
        expected = gaussian_filter1d(np.asarray(df[c], dtype=float), sigma=np.sqrt(300))
        np.testing.assert_allclose(out[c].values, expected)


# --- Resample.transform must use NEW data's values, not fit-time's -------
# (round17 fix wave 1, finding 1: a fitted Resample previously replayed
# fit-time interpolators/values for ANY same-shape new data.)

def test_resample_transform_uses_new_data_values_not_fit_time():
    n = 50
    a = pd.DataFrame({'x': np.linspace(0, 10, n)})
    b = pd.DataFrame({'x': np.linspace(100, 200, n)})

    r = Resample(n_samples=20)
    fit_time_out = r.fit_transform(a)
    new_out = r.transform(b)

    # row count equals the fitted n_samples
    assert len(new_out) == 20 == r.n_samples

    vals = new_out['x'].values
    # resampled monotone ramp stays monotone
    assert np.all(np.diff(vals) > 0)
    # values are derived from B's own range, not A's fit-time range
    np.testing.assert_allclose(vals.min(), 100, atol=1e-6)
    np.testing.assert_allclose(vals.max(), 200, atol=1e-6)
    # differs from the fit-time output
    assert not np.allclose(vals, fit_time_out['x'].values)


def test_resample_fit_transform_on_fit_time_data_unchanged():
    """fit_transform behavior on fit-time data must remain byte-identical
    to calling transform(None) / the pre-fix implementation."""
    n = 60
    df = pd.DataFrame({'x': np.linspace(0, 1, n), 'y': np.linspace(5, 6, n)})
    out_fit_transform = Resample(n_samples=17).fit_transform(df)

    r = Resample(n_samples=17)
    r.fit(df)
    out_transform_none = r.transform(None)

    pd.testing.assert_frame_equal(out_fit_transform, out_transform_none)
    assert out_fit_transform.shape[0] == 17


def test_resample_transform_multi_dataset_list_reuse():
    n = 40
    a_list = [pd.DataFrame({'x': np.linspace(0, 1, n)}),
              pd.DataFrame({'x': np.linspace(0, 1, n)})]
    b_list = [pd.DataFrame({'x': np.linspace(0, 10, n)}),
              pd.DataFrame({'x': np.linspace(20, 30, n)})]

    r = Resample(n_samples=15)
    r.fit_transform(a_list)
    out = r.transform(b_list)

    assert isinstance(out, list) and len(out) == 2
    for o, b in zip(out, b_list):
        assert len(o) == 15
        vals = o['x'].values
        np.testing.assert_allclose(vals.min(), b['x'].values.min(), atol=1e-6)
        np.testing.assert_allclose(vals.max(), b['x'].values.max(), atol=1e-6)
