"""The vendored PPCA imputer must not emit ``RuntimeWarning: divide by zero
encountered in log`` (2026-09-04, seen by a tutorial author imputing a
~100-feature matrix with a few percent of NaNs, both via ``hyp.impute(...,
model='PPCA')`` and via ``hyp.plot(x_with_nans)`` which imputes at format
time).

Root cause: the EM convergence objective in ``hypertools/external/ppca.py``
computed ``np.log(np.linalg.det(Sx))``; ``Sx = inv(I + C'C/ss)`` has every
eigenvalue in (0, 1], so its determinant underflows to 0.0 for a few dozen
latent dimensions and the log of it is -inf (with the warning). The upstream
fallback then used ``abs(slogdet(Sx)[1])`` -- a sign flip. The fix computes
``log|Sx|`` with ``slogdet`` directly.

Every test here makes real calls with RuntimeWarning promoted to an error
and NumPy's divide-by-zero floating-point flag set to raise -- the exact
conditions under which the bug surfaced.
"""
import contextlib
import warnings

import numpy as np
import pytest

import hypertools as hyp
from hypertools.external.ppca import PPCA as ExternalPPCA


@contextlib.contextmanager
def _runtime_warnings_are_errors():
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with np.errstate(divide='raise', invalid='raise'):
            yield


def _gaussian_with_nans(seed, shape, frac=0.05):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    mask = rng.random(x.shape) < frac
    x[mask] = np.nan
    return x, mask


def _low_rank_with_nans(seed, n=200, rank=5, p=100, frac=0.05, noise=0.0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, rank))
    b = rng.standard_normal((rank, p))
    full = a @ b
    if noise:
        full = full + noise * rng.standard_normal(full.shape)
    x = full.copy()
    mask = rng.random(x.shape) < frac
    x[mask] = np.nan
    return full, x, mask


# --- the triggering inputs (all reproduced the warning before the fix) ------

@pytest.mark.parametrize('seed', [0, 1, 2])
def test_external_ppca_no_log_warning_on_wide_gaussian_data(seed):
    # 200 x 100 with ~5% NaN at the default d (= 99 latent dims): det(Sx)
    # underflows to 0.0 -> np.log(0) warned before the fix.
    x, _ = _gaussian_with_nans(seed, (200, 100))
    np.random.seed(seed)
    m = ExternalPPCA()
    with _runtime_warnings_are_errors():
        m.fit(x.copy(), d=99)
    assert np.isfinite(m.data).all()
    assert np.isfinite(m.C).all()


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_impute_ppca_no_log_warning_on_wide_gaussian_data(seed):
    x, mask = _gaussian_with_nans(seed, (200, 100))
    with _runtime_warnings_are_errors():
        out = np.asarray(hyp.impute(x.copy(), model='PPCA', random_state=seed))
    assert out.shape == x.shape
    assert np.isfinite(out).all()
    # observed values are preserved exactly (the imputer's splice contract)
    assert np.array_equal(out[~mask], x[~mask])


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('noise', [0.0, 1e-3])
def test_impute_ppca_no_log_warning_on_low_rank_data(seed, noise):
    # rank-5 200 x 100 matrix (with and without a little noise), default d
    _, x, mask = _low_rank_with_nans(seed, noise=noise)
    with _runtime_warnings_are_errors():
        out = np.asarray(hyp.impute(x.copy(), model='PPCA', random_state=seed))
    assert np.isfinite(out).all()
    assert np.array_equal(out[~mask], x[~mask])


@pytest.mark.parametrize('d', [5, 10, 50, 100])
def test_impute_ppca_no_log_warning_across_latent_dims(d):
    _, x, _ = _low_rank_with_nans(0)
    with _runtime_warnings_are_errors():
        out = np.asarray(hyp.impute(x.copy(), model='PPCA', d=d,
                                    random_state=0))
    assert np.isfinite(out).all()


# --- the fit is still correct ------------------------------------------------

def test_impute_ppca_reconstructs_low_rank_matrix():
    full, x, mask = _low_rank_with_nans(0)
    with _runtime_warnings_are_errors():
        out = np.asarray(hyp.impute(x.copy(), model='PPCA', d=5,
                                    random_state=0))
    truth = full[mask]
    recovered = out[mask]
    assert np.isfinite(recovered).all()
    # the ~1000 imputed entries of an exactly rank-5 matrix are recovered
    # to a tiny fraction of the data's scale ...
    rel_rmse = np.sqrt(np.mean((recovered - truth) ** 2)) / np.std(full)
    assert rel_rmse < 0.05, rel_rmse
    # ... and are nearly perfectly correlated with the held-out truth
    assert np.corrcoef(recovered, truth)[0, 1] > 0.99


def test_slogdet_objective_matches_log_det_when_representable():
    # The convergence objective must be numerically the same quantity as
    # before whenever det(Sx) does not underflow: on a small
    # well-conditioned problem, log(det(Sx)) and slogdet(Sx)[1] agree to
    # rounding, so the fitted values are unchanged. Check the identity on
    # SPD matrices of the form the EM builds.
    rng = np.random.default_rng(0)
    for d in (2, 5, 8):
        c = rng.standard_normal((20, d))
        sx = np.linalg.inv(np.eye(d) + c.T @ c / 0.7)
        direct = np.log(np.linalg.det(sx))
        sign, logdet = np.linalg.slogdet(sx)
        assert sign == 1.0
        assert abs(direct - logdet) < 1e-10


def test_external_ppca_converges_rather_than_hitting_max_iter():
    # The sign-flipped fallback made the objective jump between iterations;
    # with a consistent log|Sx| the EM converges on the reported case well
    # inside the iteration bound and without the non-convergence warning.
    x, _ = _gaussian_with_nans(0, (200, 100))
    np.random.seed(0)
    m = ExternalPPCA()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with np.errstate(divide='raise', invalid='raise'):
            m.fit(x.copy(), d=99, max_iter=500)
    assert np.isfinite(m.data).all()


# --- the public plot path (imputes at format time) --------------------------

def test_plot_with_nans_no_log_warning():
    x, _ = _gaussian_with_nans(0, (200, 100))
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        # format_data's "missing values" notice is not a RuntimeWarning and
        # is the documented behavior; only the numerics are under test.
        with np.errstate(divide='raise', invalid='raise'):
            fig = hyp.plot(x, show=False)
    assert fig is not None
