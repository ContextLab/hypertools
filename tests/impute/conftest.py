import numpy as np
import pandas as pd


def make_df_with_nans(n=60, ncols=4, seed=0, n_missing=15):
    """A DataFrame with `n_missing` scattered NaNs (never a fully-missing
    row), plus the original (complete) DataFrame and the boolean mask of
    which entries were nulled out."""
    rng = np.random.RandomState(seed)
    x = rng.rand(n, ncols)
    original = pd.DataFrame(x, columns=[f'c{i}' for i in range(ncols)])

    rows = rng.choice(n, size=n_missing, replace=False)
    cols = rng.randint(0, ncols, size=n_missing)
    mask = np.zeros((n, ncols), dtype=bool)
    for r, c in zip(rows, cols):
        mask[r, c] = True

    missing_arr = x.copy()
    missing_arr[mask] = np.nan
    missing = pd.DataFrame(missing_arr, columns=original.columns)
    return original, missing, mask
