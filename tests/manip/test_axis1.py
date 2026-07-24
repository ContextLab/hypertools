import numpy as np
import pandas as pd
import pytest

from hypertools.manip import ZScore, Normalize, Smooth


def _make_df():
    return pd.DataFrame(np.random.RandomState(2).rand(5, 30))


# upstream: datawrangler calls pd.concat(copy=...), deprecated in pandas 4.
# the filter names the DeprecationWarning BASE class on purpose:
# pandas.errors.Pandas4Warning subclasses it but does not exist on
# pandas 2.x, and pytest aborts (UsageError, exit 4) on filter
# categories it cannot import
@pytest.mark.filterwarnings(
    'ignore:The copy keyword is deprecated:DeprecationWarning')
def test_zscore_axis1_zero_mean_unit_std_per_row():
    df = _make_df()
    out = ZScore(axis=1).fit_transform(df)
    assert out.shape == df.shape
    # each ROW should have ~zero mean and ~unit std across columns
    assert np.allclose(out.mean(axis=1).to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(out.std(axis=1, ddof=1).to_numpy(), 1.0, atol=1e-6)


# upstream: datawrangler calls pd.concat(copy=...), deprecated in pandas 4.
# the filter names the DeprecationWarning BASE class on purpose:
# pandas.errors.Pandas4Warning subclasses it but does not exist on
# pandas 2.x, and pytest aborts (UsageError, exit 4) on filter
# categories it cannot import
@pytest.mark.filterwarnings(
    'ignore:The copy keyword is deprecated:DeprecationWarning')
def test_normalize_axis1_scales_each_row_to_unit_range():
    df = _make_df()
    out = Normalize(min=0, max=1, axis=1).fit_transform(df)
    assert out.shape == df.shape
    # each ROW should be scaled into [0, 1], with min 0 and max 1 achieved per row
    row_min = out.min(axis=1).to_numpy()
    row_max = out.max(axis=1).to_numpy()
    assert np.all(out.to_numpy() >= -1e-9)
    assert np.all(out.to_numpy() <= 1 + 1e-9)
    assert np.allclose(row_min, 0.0, atol=1e-9)
    assert np.allclose(row_max, 1.0, atol=1e-9)


def test_smooth_axis1_preserves_shape_no_error():
    df = _make_df()
    out = Smooth(axis=1, kernel_width=11, order=3).fit_transform(df)
    assert out.shape == df.shape
    assert not out.isna().any().any()
    # maintain_bounds=True (default) should keep values within each row's original range
    row_min = df.min(axis=1).to_numpy()
    row_max = df.max(axis=1).to_numpy()
    assert np.all(out.to_numpy() >= row_min[:, None] - 1e-9)
    assert np.all(out.to_numpy() <= row_max[:, None] + 1e-9)
