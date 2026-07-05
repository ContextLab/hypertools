import numpy as np
import pandas as pd
import pytest

from hypertools.impute.sklearn_imputers import SimpleImputer, KNNImputer, IterativeImputer
from hypertools.impute.ppca import PPCA
from hypertools.impute.kalman import Kalman

from .conftest import make_df_with_nans


@pytest.mark.parametrize('cls,kwargs', [
    (SimpleImputer, {}),
    (KNNImputer, {}),
    (IterativeImputer, {'random_state': 0}),
])
def test_sklearn_imputers_fill_scattered_nans_preserve_rest(cls, kwargs):
    original, missing, mask = make_df_with_nans()
    imputer = cls(**kwargs)
    out = imputer.fit_transform(missing)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == original.shape
    assert not out.isna().any().any()

    unchanged = ~mask
    assert np.array_equal(out.to_numpy()[unchanged], original.to_numpy()[unchanged])


def test_kalman_imputer_fills_scattered_nans_preserves_rest():
    pytest.importorskip('pykalman')
    original, missing, mask = make_df_with_nans(n=80, ncols=3, n_missing=10)
    imputer = Kalman(n_iter=3)
    out = imputer.fit_transform(missing)

    assert not out.isna().any().any()
    unchanged = ~mask
    assert np.array_equal(out.to_numpy()[unchanged], original.to_numpy()[unchanged])


def test_kalman_imputer_fills_fully_missing_interior_rows():
    """Regression test for GH #169: rows where EVERY feature is missing must
    still be filled -- this is the gap PPCA cannot close (see
    test_ppca_warns_and_leaves_nan_on_fully_missing_rows below)."""
    pytest.importorskip('pykalman')

    n = 60
    t = np.arange(n)
    signal = np.stack([
        np.sin(t / 5.0) + 0.01 * t,
        np.cos(t / 5.0) + 0.01 * t,
        np.sin(t / 3.0) - 0.01 * t,
    ], axis=1)
    original = pd.DataFrame(signal, columns=['a', 'b', 'c'])

    fully_missing_rows = [28, 29, 30]
    missing = original.copy()
    missing.iloc[fully_missing_rows, :] = np.nan

    imputer = Kalman(n_iter=5)
    out = imputer.fit_transform(missing)

    assert np.isfinite(out.to_numpy()).all()

    keep = np.ones(n, dtype=bool)
    keep[fully_missing_rows] = False
    assert np.array_equal(out.to_numpy()[keep], original.to_numpy()[keep])

    # Filled values should be within a loose plausible range set by the
    # surrounding (neighboring) observations -- not exact, just sane.
    neighborhood = original.iloc[max(0, 28 - 5):min(n, 30 + 6)]
    lo = neighborhood.min() - 1.0
    hi = neighborhood.max() + 1.0
    filled_rows = out.iloc[fully_missing_rows]
    assert (filled_rows >= lo).all().all()
    assert (filled_rows <= hi).all().all()


def test_ppca_fills_scattered_nans():
    original, missing, mask = make_df_with_nans(n=100, ncols=10, n_missing=20)
    imputer = PPCA()
    out = imputer.fit_transform(missing)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == original.shape
    assert not out.isna().any().any()


def test_ppca_warns_and_leaves_nan_on_fully_missing_rows():
    rng = np.random.RandomState(2)
    n, d = 100, 10
    df = pd.DataFrame(rng.rand(n, d))
    df.iloc[[5, 6]] = np.nan  # fully-missing rows -- PPCA cannot reconstruct these

    imputer = PPCA()
    with pytest.warns(UserWarning, match='cannot fill'):
        out = imputer.fit_transform(df)

    assert out.iloc[5].isna().all()
    assert out.iloc[6].isna().all()

    other_rows = out.drop(index=[5, 6])
    assert not other_rows.isna().any().any()
