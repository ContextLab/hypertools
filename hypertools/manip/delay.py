"""Time-delay (Takens) embedding manipulator (GH #285): `Delay(tau=1, dims=2,
drop_edges=True)`.

Convention (stated explicitly here, since "delay embedding" is used with
different column/row orientations across the literature): for a column `x`
and an output row currently labeled `t` (the row's OWN index -- the
DataFrame's original index, or position for a bare array), the output
columns hold ``x[t - (dims - 1) * tau], ..., x[t - tau], x[t]`` in that
LEFT-TO-RIGHT order -- i.e. the LAST generated column for each input column
is the undelayed value itself (lag 0), and earlier columns hold
progressively larger lags. This ordering (oldest-to-newest, left-to-right)
is chosen because it is exactly what
``docs/tutorials/modern_sklearn_dynamics.ipynb`` builds by hand via
``np.column_stack([x[i * tau:i * tau + n] for i in range(dims)])`` --
`Delay(tau=5, dims=20).fit_transform(x)` reproduces that array exactly
(see `tests/test_manip_delay.py`).

`predict/kalman.py`'s internal delay embedding (`_companion_transition`)
uses the OPPOSITE column order (newest-first: ``x[i - lags:i][::-1]``) for
its own regression bookkeeping; that is a private internal detail of the
Kalman forecaster, not touched or relied on here.
"""
import datawrangler as dw
import numpy as np
import pandas as pd

from .common import Manipulator


def fitter(data, **kwargs):
    """Record the `Delay` manipulator's parameters.

    Delay embedding is stateless: nothing is estimated from the fit-time
    data itself, only `tau`/`dims`/`drop_edges` are recorded.

    Parameters
    ----------
    data : DataFrame or list of DataFrame
        Data being fit on (unused beyond marking the manipulator fitted).
    **kwargs
        `tau`, `dims`, `drop_edges` : the `Delay` constructor parameters,
        passed through unchanged.

    Returns
    -------
    dict
        `{'tau', 'dims', 'drop_edges'}`.
    """
    return {'tau': kwargs['tau'], 'dims': kwargs['dims'], 'drop_edges': kwargs['drop_edges']}


def _delay_embed_dataframe(data, tau, dims, drop_edges):
    """Delay-embed ONE DataFrame (every column, independently).

    For each input column `c`, generates `dims` output columns
    `f'{c}_lag{lag}'` for `lag` in `(dims - 1) * tau, ..., tau, 0`
    (left-to-right), where the `lag`-th column at output row `t` holds
    `c`'s value `lag` rows earlier (`NaN` if that row does not exist).
    """
    n_rows = data.shape[0]
    max_lag = tau * (dims - 1)
    if drop_edges and max_lag >= n_rows:
        raise ValueError(
            f'Delay(tau={tau}, dims={dims}) needs at least {max_lag + 1} '
            'row(s) (tau * (dims - 1) + 1) to produce any output with '
            f'drop_edges=True; got {n_rows} row(s). Use a smaller dims/tau, '
            'or drop_edges=False to pad the too-short rows with NaN '
            'instead of dropping them.')

    lags = [(dims - 1 - i) * tau for i in range(dims)]
    out_columns = {}
    for c in data.columns:
        values = np.asarray(data[c], dtype=float)
        for lag in lags:
            shifted = np.full(n_rows, np.nan)
            if lag == 0:
                shifted[:] = values
            elif lag < n_rows:
                shifted[lag:] = values[:n_rows - lag]
            out_columns[f'{c}_lag{lag}'] = shifted

    embedded = pd.DataFrame(out_columns, index=data.index)
    if drop_edges:
        embedded = embedded.iloc[max_lag:]
    return embedded


def _transform(data, **kwargs):
    """Apply delay embedding PER DATASET (mirroring `Smooth`/`Resample`):
    lists and stacked (multiindex) DataFrames are embedded one dataset at
    a time, so no dataset's history bleeds into another's."""
    if dw.zoo.is_multiindex_dataframe(data):
        return dw.stack([_transform(d, **kwargs) for d in dw.unstack(data)])
    if isinstance(data, list):
        return [_transform(d, **kwargs) for d in data]
    if not isinstance(data, pd.DataFrame):
        # e.g. a bare array passed between hypertools.Pipeline steps
        data = pd.DataFrame(data)
    return _delay_embed_dataframe(data, kwargs['tau'], kwargs['dims'], kwargs['drop_edges'])


def transformer(data, **kwargs):
    """Delay-embed `data` for the `Delay` manipulator.

    Parameters
    ----------
    data : DataFrame, multiindex DataFrame, or list of DataFrame
        Data to delay-embed.
    **kwargs
        `tau`, `dims`, `drop_edges` : parameters from `fitter`.

    Returns
    -------
    The delay-embedded data, in the same list/multiindex structure as
    `data` (a plain DataFrame for a single dataset), with `dims` output
    columns per input column (see the module docstring for the exact
    column order/naming) and, when `drop_edges=True`, `tau * (dims - 1)`
    fewer rows (the leading rows lacking enough history to fill every
    lag); `drop_edges=False` keeps every row, with `NaN` in place of any
    lag that reaches before the start of the data.

    Raises
    ------
    ValueError
        If `drop_edges=True` and there are not enough rows to produce
        even one fully-populated output row.
    """
    return _transform(data, **kwargs)


class Delay(Manipulator):
    """Time-delay (Takens) embedding: stack each column with lagged copies
    of itself.

    For a column `x` and output row `t`, the generated columns hold
    ``x[t - (dims - 1) * tau], ..., x[t - tau], x[t]`` left-to-right (the
    LAST column is always the undelayed value) -- see the module docstring
    for why this order was chosen (it matches
    ``docs/tutorials/modern_sklearn_dynamics.ipynb``'s hand-built
    ``np.column_stack`` embedding exactly).

    Parameters
    ----------
    tau : int
        Delay length in samples between consecutive lags (default: 1).
        Must be a positive integer.

    dims : int
        Embedding dimension: how many lagged copies of each column to
        generate, including the undelayed copy itself (default: 2). Must
        be a positive integer; `dims=1` is a no-op rename (one column per
        input column, `_lag0`, values unchanged).

    drop_edges : bool
        If True (default), drop the leading `tau * (dims - 1)` rows that
        do not have enough history to fill every lag. If False, keep
        every row, with `NaN` for any lag reaching before the start of
        the data.

    Notes
    -----
    Each input column is embedded independently (multi-column input
    produces `dims` output columns per input column, grouped by input
    column in input order; see `_delay_embed_dataframe`). Delay embedding
    is applied PER DATASET: each element of a list input is embedded
    independently, so history never bleeds across dataset boundaries.

    Raises
    ------
    ValueError
        If `tau`/`dims` are not positive integers, or if `drop_edges=True`
        and the data has too few rows to produce any output row.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from hypertools.manip import Delay
    >>> df = pd.DataFrame({'x': np.arange(10, dtype=float)})
    >>> out = Delay(tau=2, dims=3).fit_transform(df)
    >>> list(out.columns)
    ['x_lag4', 'x_lag2', 'x_lag0']
    >>> out.to_numpy()
    array([[0., 2., 4.],
           [1., 3., 5.],
           [2., 4., 6.],
           [3., 5., 7.],
           [4., 6., 8.],
           [5., 7., 9.]])
    """

    # transform re-derives everything from the data being transformed (no
    # fit-time statistics are replayed), so reusing a fitted Delay on new
    # data is always well-defined.
    _stateless_transform = True

    # noinspection PyShadowingBuiltins
    def __init__(self, tau=1, dims=2, drop_edges=True):
        if isinstance(tau, bool) or not isinstance(tau, (int, np.integer)) or tau < 1:
            raise ValueError(f'tau must be a positive integer; got {tau!r}')
        if isinstance(dims, bool) or not isinstance(dims, (int, np.integer)) or dims < 1:
            raise ValueError(f'dims must be a positive integer; got {dims!r}')
        tau = int(tau)
        dims = int(dims)
        drop_edges = bool(drop_edges)
        required = ['tau', 'dims', 'drop_edges']
        super().__init__(fitter=fitter, transformer=transformer, data=None, tau=tau, dims=dims,
                         drop_edges=drop_edges, required=required)

        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.tau = tau
        self.dims = dims
        self.drop_edges = drop_edges
        self.required = required
