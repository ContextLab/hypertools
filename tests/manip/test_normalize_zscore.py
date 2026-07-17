import numpy as np
import pandas as pd
import pytest
from hypertools.manip.normalize import Normalize
from hypertools.manip.zscore import ZScore


# upstream: datawrangler calls pd.concat(copy=...), deprecated in pandas 4
@pytest.mark.filterwarnings(
    'ignore:The copy keyword is deprecated:pandas.errors.Pandas4Warning')
def test_normalize_scales_to_unit_range():
    df = pd.DataFrame(np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]]), columns=["a", "b"])
    out = Normalize(min=0, max=1, axis=0).fit_transform(df)
    assert np.isclose(out["a"].min(), 0.0) and np.isclose(out["a"].max(), 1.0)
    assert np.isclose(out["b"].min(), 0.0) and np.isclose(out["b"].max(), 1.0)


# upstream: datawrangler calls pd.concat(copy=...), deprecated in pandas 4
@pytest.mark.filterwarnings(
    'ignore:The copy keyword is deprecated:pandas.errors.Pandas4Warning')
def test_zscore_zero_mean_unit_std():
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.rand(50, 3), columns=list("abc"))
    out = ZScore(axis=0).fit_transform(df)
    assert np.allclose(out.mean(axis=0).to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(out.std(axis=0, ddof=1).to_numpy(), 1.0, atol=1e-6)
