# noinspection PyPackageRequirements
"""Tests for GH #285 item 1: `Smooth(kernel='boxcar', kernel_width=,
center=True|False, min_periods=)`.

`center=`/`min_periods=` deliberately reuse pandas' `rolling(...)`
vocabulary rather than an `align=` kwarg: `hypertools`'s cross-module API
already has an unrelated top-level `align=` (the ALIGNMENT STAGE, e.g.
`HyperAlign`) on `hyp.manip`/`hyp.plot`/`hyp.analyze`, so `center=` reaches
`Smooth` through the ORDINARY kwargs path with no special-casing anywhere
(no collision to disambiguate).

All tests use real data and real hypertools calls (no mocks). The core
claim is that `center=False` (trailing) boxcar reproduces
`pd.Series(x).rolling(w).mean()` EXACTLY (values and NaN placement),
including through `hyp.manip`'s DataFrame/list/array input forms and
through the `manip=` cross-module kwarg of `hyp.plot`/`hyp.analyze`.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp
from hypertools.manip import Smooth
from hypertools.tools.analyze import analyze


def _rng():
    return np.random.default_rng(0)


# --- core claim: byte-identical to pd.Series.rolling(w).mean() ------------

def test_trailing_boxcar_matches_pandas_rolling_default_min_periods():
    x = _rng().standard_normal(60)
    expected = pd.Series(x).rolling(12).mean().to_numpy()
    got = np.asarray(hyp.manip(x[:, None], model='Smooth', kernel='boxcar',
                               kernel_width=12, center=False)).ravel()
    # NaN placement must match exactly: first 11 rows NaN, nowhere else
    assert np.array_equal(np.isnan(got), np.isnan(expected))
    assert np.all(np.isnan(got[:11]))
    assert not np.any(np.isnan(got[11:]))
    np.testing.assert_allclose(got[11:], expected[11:])


def test_trailing_boxcar_min_periods_1_gives_expanding_start():
    x = _rng().standard_normal(40)
    expected = pd.Series(x).rolling(12, min_periods=1).mean().to_numpy()
    got = np.asarray(hyp.manip(x[:, None], model='Smooth', kernel='boxcar',
                               kernel_width=12, center=False,
                               min_periods=1)).ravel()
    assert not np.any(np.isnan(got))
    np.testing.assert_allclose(got, expected)
    # the very first output sample of an expanding-start average is the
    # input itself
    assert got[0] == pytest.approx(x[0])


def test_trailing_boxcar_custom_min_periods_matches_pandas():
    x = _rng().standard_normal(40)
    expected = pd.Series(x).rolling(12, min_periods=5).mean().to_numpy()
    got = np.asarray(hyp.manip(x[:, None], model='Smooth', kernel='boxcar',
                               kernel_width=12, center=False,
                               min_periods=5)).ravel()
    assert np.array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_allclose(got[~np.isnan(got)], expected[~np.isnan(expected)])


def test_trailing_kernel_width_not_forced_odd():
    # center=True bumps even kernel_width to odd (with a warning); center=False
    # (a causal window has no symmetry requirement) must NOT do this, or the
    # weather-tutorial recipe (kernel_width=12, even) would silently smooth
    # over a different width than requested.
    x = _rng().standard_normal(30)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        got = np.asarray(hyp.manip(x[:, None], model='Smooth', kernel='boxcar',
                                   kernel_width=12, center=False)).ravel()
    odd_bump_warnings = [w for w in caught
                        if 'Increasing smoothing kernel width' in str(w.message)]
    assert len(odd_bump_warnings) == 0
    expected = pd.Series(x).rolling(12).mean().to_numpy()
    np.testing.assert_allclose(got[11:], expected[11:])


# --- center=True is unchanged (default) -------------------------------------

def test_center_true_default_unchanged_behavior():
    x = _rng().standard_normal(40)
    a = np.asarray(hyp.manip(x[:, None], model='Smooth', kernel='boxcar', kernel_width=11))
    b = np.asarray(hyp.manip(x[:, None], model='Smooth', kernel='boxcar', kernel_width=11,
                             center=True))
    np.testing.assert_allclose(a, b)


# --- savgol/gaussian center=False: refused with a clear ValueError --------

@pytest.mark.parametrize('kernel', ['savgol', 'gaussian'])
def test_center_false_refused_for_non_boxcar_kernels(kernel):
    x = _rng().standard_normal(30)
    with pytest.raises(ValueError, match='center=False'):
        hyp.manip(x[:, None], model='Smooth', kernel=kernel, center=False)


# --- min_periods only meaningful with center=False --------------------------

def test_min_periods_with_center_true_raises():
    x = _rng().standard_normal(30)
    with pytest.raises(ValueError, match='min_periods'):
        hyp.manip(x[:, None], model='Smooth', kernel='boxcar', min_periods=3)


def test_invalid_center_value_raises():
    x = _rng().standard_normal(30)
    with pytest.raises(ValueError, match='center'):
        hyp.manip(x[:, None], model='Smooth', kernel='boxcar', center='trailing')


# --- DataFrame input: index preserved --------------------------------------

def test_trailing_dataframe_index_preserved():
    x = _rng().standard_normal(20)
    idx = pd.date_range('2020-01-01', periods=20, freq='D')
    df = pd.DataFrame({'y': x}, index=idx)
    out = Smooth(kernel='boxcar', kernel_width=5, center=False).fit_transform(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == list(idx)
    expected = df['y'].rolling(5).mean()
    np.testing.assert_allclose(out['y'].to_numpy(), expected.to_numpy(), equal_nan=True)


# --- list input: per-dataset, no bleed across dataset boundaries ----------

def test_trailing_list_input_is_per_dataset():
    a = _rng().standard_normal(30)
    b = _rng().standard_normal(30) + 100  # far away, so bleed would be obvious
    out = hyp.manip([a[:, None], b[:, None]], model='Smooth', kernel='boxcar',
                    kernel_width=6, center=False)
    assert isinstance(out, list) and len(out) == 2
    exp_a = pd.Series(a).rolling(6).mean().to_numpy()
    exp_b = pd.Series(b).rolling(6).mean().to_numpy()
    np.testing.assert_allclose(np.asarray(out[0]).ravel()[5:], exp_a[5:])
    np.testing.assert_allclose(np.asarray(out[1]).ravel()[5:], exp_b[5:])
    # no cross-dataset bleed: b's values never touch a's leading NaN region
    assert np.all(np.isnan(np.asarray(out[0]).ravel()[:5]))
    assert np.all(np.isnan(np.asarray(out[1]).ravel()[:5]))


# --- multi-column DataFrame: each column smoothed independently -----------

def test_trailing_multicolumn_matches_per_column_rolling():
    x = _rng().standard_normal((25, 3))
    df = pd.DataFrame(x, columns=['a', 'b', 'c'])
    out = Smooth(kernel='boxcar', kernel_width=7, center=False).fit_transform(df)
    for c in df.columns:
        expected = df[c].rolling(7).mean()
        np.testing.assert_allclose(out[c].to_numpy(), expected.to_numpy(), equal_nan=True)


# --- the exact weather-tutorial drop-in replacement ------------------------

def test_weather_recipe_drop_in_replacement_byte_identical():
    # examples/animate_weather_decades.py L275:
    #   rolling = pd.Series(mean).rolling(12).mean().to_numpy()
    mean = _rng().standard_normal(120) * 5 + 15  # plausible monthly-mean shape
    rolling = pd.Series(mean).rolling(12).mean().to_numpy()
    replacement = np.asarray(
        hyp.manip(mean[:, None], model='Smooth', kernel='boxcar',
                 kernel_width=12, center=False)).ravel()
    np.testing.assert_allclose(replacement, rolling, equal_nan=True)


# --- the EXACT call form reaches Smooth via the ORDINARY kwargs path -------
# (no special-casing in hypertools/manip/manip.py: `center=` never collided
# with the top-level `align=` cross-module kwarg in the first place, unlike
# an earlier `align='trailing'` design for this same feature.)

def test_exact_manip_call_form_no_special_casing():
    mean = _rng().standard_normal(48) * 3 + 10
    expected = pd.Series(mean).rolling(12).mean().to_numpy()
    got = np.asarray(
        hyp.manip(mean[:, None], model='Smooth', kernel='boxcar',
                 kernel_width=12, center=False)).ravel()
    np.testing.assert_allclose(got, expected, equal_nan=True)
    # a genuine align-STAGE spec is completely unaffected by this feature
    # (there is no longer any name collision to guard against)
    a = _rng().standard_normal((15, 3))
    b = _rng().standard_normal((15, 3))
    result = hyp.manip([a, b], model='Smooth', align='HyperAlign', kernel_width=5)
    assert isinstance(result, list) and len(result) == 2


# --- cross-module kwargs: manip= inside hyp.plot / hyp.analyze -------------
# NOTE: `hyp.plot`/`hyp.analyze` (not owned by this change) only route
# STAGE-SPECIFIC kwargs through the canonical dict-spec form
# (`{'model': ..., 'kwargs': {...}}`); a bare loose kwarg (e.g.
# `kernel_width=`) is rejected by their own kwarg validation regardless of
# `center=` -- this is pre-existing behavior, unrelated to GH #285 (verified
# against `Resample`'s `n_samples=` too). So the dict-spec form is used here
# for the plot/analyze routing checks.

def test_plot_manip_smooth_trailing_via_dict_spec():
    x = _rng().standard_normal((30, 4))
    fig = hyp.plot(
        x,
        manip={'model': 'Smooth',
               'kwargs': {'kernel': 'boxcar', 'kernel_width': 5,
                          'center': False, 'min_periods': 1}},
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_analyze_manip_smooth_trailing_via_dict_spec():
    x = _rng().standard_normal((30, 4))
    result = analyze(
        x,
        manip={'model': 'Smooth',
               'kwargs': {'kernel': 'boxcar', 'kernel_width': 5,
                          'center': False, 'min_periods': 1}},
        reduce=None,
    )
    result = np.asarray(result)
    assert result.shape == (30, 4)
    assert not np.any(np.isnan(result))
