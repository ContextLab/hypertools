#!/usr/bin/env python
"""Knock holes in data, on purpose (GH #285).

Every missing-data demo needs damaged data to fill back in, and each one
wrote its own knock-out by hand: a seeded `rng.choice` over flat cell
indices, a `rng.random(shape) < 0.05` mask, an occlusion band of fully
missing rows, and -- in the projectile tutorial -- a written-out workaround
for a genuine pandas trap:

    ``.to_numpy()`` can return a read-only, Fortran-ordered array; a plain
    ``.ravel()`` on that silently returns a *copy* (since it must reorder to
    C order), so writes through it would vanish.

`damage` owns that trap so no caller has to.
"""

import numpy as np
import pandas as pd

__all__ = ['damage']


def _as_float_values(x, name='x'):
    """A writable, C-contiguous float copy of `x`'s values.

    This is the pandas trap named in the module docstring: `.to_numpy()` may
    hand back a read-only, Fortran-ordered array, so the copy (and the C
    order) is what makes a write through it land. Copying here is also what
    leaves the caller's data untouched.
    """
    raw = x.to_numpy() if isinstance(x, (pd.DataFrame, pd.Series)) \
        else np.asarray(x)
    if raw.dtype.kind not in 'fiub':
        raise TypeError(
            f"{name} must hold numeric values (damage marks cells missing "
            f"with NaN); got dtype {raw.dtype!r}.")
    if raw.ndim not in (1, 2):
        raise ValueError(
            f"{name} must be 1- or 2-dimensional (observations x features); "
            f"got {raw.ndim} dimension(s).")
    # `copy=True, order='C'` is the whole point: writable and C-contiguous
    # whatever `.to_numpy()` handed back.
    return np.array(raw, dtype=float, copy=True, order='C')


def _resolve_rows(rows, row_frac, n_rows, rng):
    """The positional row indices to blank out entirely."""
    if rows is not None and row_frac is not None:
        raise ValueError(
            "pass rows= (which rows) or row_frac= (how many, chosen at "
            "random), not both.")

    if rows is None and row_frac is None:
        return np.empty(0, dtype=int)

    if row_frac is not None:
        if not 0 <= row_frac <= 1:
            raise ValueError(
                f"row_frac= must be between 0 and 1; got {row_frac!r}.")
        count = int(round(row_frac * n_rows))
        return np.sort(rng.choice(n_rows, size=count, replace=False))

    if isinstance(rows, slice):
        return np.arange(n_rows)[rows]

    if isinstance(rows, (int, np.integer)) and not isinstance(rows, bool):
        count = int(rows)
        if count < 0:
            raise ValueError(
                f"rows= as a count must be non-negative; got {rows!r}.")
        if count > n_rows:
            raise IndexError(
                f"rows={rows} asks for more rows than the data has "
                f"({n_rows}). Pass a sequence of positions to name "
                "particular rows.")
        return np.sort(rng.choice(n_rows, size=count, replace=False))

    index = np.asarray(rows)
    if index.dtype == bool:
        if index.shape != (n_rows,):
            raise IndexError(
                f"boolean rows= must have one entry per row ({n_rows}); got "
                f"{index.shape[0] if index.ndim else 0}.")
        return np.flatnonzero(index)
    if index.size == 0:
        return np.empty(0, dtype=int)
    if index.dtype.kind not in 'iu':
        raise TypeError(
            "rows= must be an int (how many, at random), a slice, or a "
            f"sequence of row positions; got dtype {index.dtype!r}.")
    if index.min() < -n_rows or index.max() >= n_rows:
        raise IndexError(
            f"rows= contains a position outside the data's {n_rows} row(s): "
            f"{[int(i) for i in index[(index < -n_rows) | (index >= n_rows)]]}.")
    return np.unique(index % n_rows)


def _scatter(values, blanked_rows, frac, rng):
    """Flat indices of the cells the scattered draw takes out.

    Candidates are the cells that are *observed* (already-NaN cells are not
    damaged twice) and not in an entirely blanked row -- the same "only
    among the still-intact cells" rule the projectile tutorial wrote by
    hand. Acceptance is streamed over a permutation of the candidates so
    that a scattered draw never empties a row: with two or more features,
    every row keeps at least one observed cell unless `rows=` asked for it.
    """
    if frac == 0:
        return np.empty(0, dtype=int)

    n_rows, n_cols = values.shape
    candidate = np.isfinite(values)
    candidate[blanked_rows] = False
    flat = np.flatnonzero(candidate.ravel())
    if flat.size == 0:
        return np.empty(0, dtype=int)

    n_take = int(round(frac * flat.size))
    if n_take == 0:
        return np.empty(0, dtype=int)

    order = rng.permutation(flat)
    if n_cols < 2:
        # A single feature: a missing cell IS a missing row, and there is no
        # "rest of the row" to preserve, so the guarantee is vacuous and the
        # draw is taken as asked.
        return np.sort(order[:n_take])

    remaining = candidate.sum(axis=1)
    taken = []
    for cell in order:
        row = cell // n_cols
        if remaining[row] <= 1:
            continue
        remaining[row] -= 1
        taken.append(cell)
        if len(taken) == n_take:
            break
    return np.sort(np.asarray(taken, dtype=int))


def _damage_one(x, frac, rows, row_frac, rng, name='x'):
    """Damage one dataset; returns ``(damaged_copy, mask)``."""
    values = _as_float_values(x, name=name)
    # A 1-D dataset is n observations of a single feature. `reshape` on a
    # fresh C-contiguous copy is a view, so writes through `two_d` land in
    # `values`.
    two_d = values.reshape(len(values), -1)

    blanked_rows = _resolve_rows(rows, row_frac, len(two_d), rng)
    mask = np.zeros(two_d.shape, dtype=bool)
    if blanked_rows.size:
        mask[blanked_rows, :] = np.isfinite(two_d[blanked_rows, :])

    scattered = _scatter(two_d, blanked_rows, frac, rng)
    if scattered.size:
        mask.flat[scattered] = True

    two_d[mask] = np.nan
    mask = mask.reshape(values.shape)

    if isinstance(x, pd.DataFrame):
        return (pd.DataFrame(values, index=x.index, columns=x.columns),
                pd.DataFrame(mask, index=x.index, columns=x.columns))
    if isinstance(x, pd.Series):
        return (pd.Series(values, index=x.index, name=x.name),
                pd.Series(mask, index=x.index, name=x.name))
    return values, mask


def damage(x, frac=0.05, rows=None, row_frac=None, seed=None,
           return_mask=False):
    """Set a fraction of cells (and/or whole rows) to NaN, reproducibly.

    Returns a damaged *copy*: the input is never modified. Damage comes in
    two independent flavours, and a call may use either or both:

    - **scattered** cells (`frac`), which is what an imputer conditioned on
      the other columns of the same row (PPCA) can fill; and
    - **whole rows** (`rows`, `row_frac`) -- an occlusion band -- which it
      cannot, because such a row has no observed cell left to condition on.

    A scattered draw never empties a row: with two or more features, every
    row keeps at least one observed cell unless `rows=`/`row_frac=` asked
    for it.

    Parameters
    ----------
    x : numpy.ndarray, pandas.DataFrame, pandas.Series, or list of them
        The data to damage. A list/tuple damages each element with its own
        independent draw, all from the one seeded generator, so the whole
        call is reproducible. A DataFrame keeps its index and columns; its
        dtypes become float, because NaN needs one.

    frac : float, optional
        Fraction of the *still-intact* cells to set to NaN (default 0.05),
        chosen without replacement. Cells that are already NaN, and cells in
        rows blanked by `rows=`/`row_frac=`, are not candidates. The count
        is ``round(frac * n_candidates)``. ``frac=0`` scatters nothing.

    rows : int, slice, or sequence of int/bool, optional
        Rows to blank out entirely. An **int** means "this many rows, chosen
        at random"; a slice or a sequence of positions (or a boolean mask
        with one entry per row) names particular rows. Negative positions
        count from the end. A position outside the data raises `IndexError`.

    row_frac : float, optional
        Blank out this fraction of the rows, chosen at random
        (``round(row_frac * n_rows)`` of them). Mutually exclusive with
        `rows=`.

    seed : int, numpy.random.Generator, or None, optional
        Seed (or generator) for every draw. Pass one to make a figure
        reproducible.

    return_mask : bool, optional
        When True, return ``(damaged, mask)``: a boolean array (or
        DataFrame, matching the input) that is True exactly where *this
        call* set a cell to NaN. Cells that arrived already NaN are False.

    Returns
    -------
    damaged : same type as `x`
        The damaged copy (float dtype), or a list of them for a list input.
    mask : same type as `x`, only when ``return_mask=True``
        True where this call set a NaN.

    Notes
    -----
    With a single feature there is no "rest of the row" to preserve, so
    `frac` damages cells directly and the no-empty-row guarantee is vacuous.

    `damage` owns a pandas trap the tutorials wrote out by hand:
    ``DataFrame.to_numpy()`` can return a read-only, Fortran-ordered array,
    and ``.ravel()`` on such an array silently returns a *copy*, so writes
    through it vanish. Values are copied into a writable, C-contiguous float
    array before anything is written.

    Examples
    --------
    Scattered cells, reproducibly:

    >>> import numpy as np
    >>> from hypertools.tools import damage
    >>> full = np.arange(40.).reshape(10, 4)
    >>> holey = damage(full, frac=0.1, seed=0)
    >>> int(np.isnan(holey).sum())
    4
    >>> bool(np.isnan(full).any())      # the original is untouched
    False
    >>> np.array_equal(holey, damage(full, frac=0.1, seed=0), equal_nan=True)
    True

    No row is ever emptied by the scattered draw alone:

    >>> bool(np.isnan(damage(full, frac=0.9, seed=1)).all(axis=1).any())
    False

    An occlusion band plus scattered noise, on a DataFrame:

    >>> import pandas as pd
    >>> frame = pd.DataFrame(full, columns=list('abcd'))
    >>> gapped, mask = damage(frame, frac=0.1, rows=slice(2, 5), seed=0,
    ...                       return_mask=True)
    >>> list(gapped.columns), int(gapped.isna().all(axis=1).sum())
    (['a', 'b', 'c', 'd'], 3)
    >>> bool((mask.to_numpy() == gapped.isna().to_numpy()).all())
    True

    See Also
    --------
    hypertools.impute : fill the holes back in.
    hypertools.tools.missing_inds : which rows contain a NaN.
    """
    if not 0 <= frac <= 1:
        raise ValueError(f"frac= must be between 0 and 1; got {frac!r}.")

    rng = seed if isinstance(seed, np.random.Generator) \
        else np.random.default_rng(seed)

    if isinstance(x, (list, tuple)):
        pairs = [_damage_one(item, frac, rows, row_frac, rng,
                             name=f'x[{i}]')
                 for i, item in enumerate(x)]
        damaged = [pair[0] for pair in pairs]
        if return_mask:
            return damaged, [pair[1] for pair in pairs]
        return damaged

    damaged, mask = _damage_one(x, frac, rows, row_frac, rng)
    if return_mask:
        return damaged, mask
    return damaged
