#!/usr/bin/env python
"""Axis-agnostic hierarchy grouping, shared by `hyp.plot` and `hyp.predict`.

This module lives under `core/` rather than `plot/` on purpose: grouping a
frame by its outer index levels is not a rendering concern, and
`hypertools.predict` must never import from `hypertools.plot` (the 1.0
package split put shared machinery under `core/`).

Three DIFFERENT rules exist deliberately:

- `group_columns`            -- COLUMN hierarchy, plot AND predict: the
                                innermost level is the FEATURE axis,
                                everything above it groups.
- `group_rows_for_forecast`  -- ROW hierarchy, predict only: the innermost
                                level is the TIME axis, everything above it
                                groups, and the innermost level SURVIVES as
                                each group's index.
- ROW hierarchy, plot        -- keeps its historical rule (one leaf per
                                unique full index tuple) in
                                `hypertools.plot.multiindex.expand_multiindex`,
                                untouched.

ONE INVARIANT spans both helpers here: every leaf they return is
NON-HIERARCHICAL on the axis it was grouped along -- flat columns for
`group_columns`, a flat index for `group_rows_for_forecast`. Hierarchy is
detected by `nlevels >= 2`, and `hyp.predict` recurses into each group, so a
leaf that still carried its grouping levels would be regrouped forever. Note
`expand_multiindex` deliberately does NOT satisfy this (its leaves keep the
full row MultiIndex and re-expand to themselves -- a fixed point), which is
why its leaves must never be fed back into a hierarchy-detecting entry point.
Grouping never mutates the caller's frame.

See docs/hierarchy.rst for the user-facing comparison table.
"""

import pandas as pd

#: Stand-in for ANY missing hierarchy label during comparison and indexing.
#: Module-private and never user-visible: it exists only so that missing
#: labels hash and compare as one another's equals. Original label values
#: are always what get stored in keys and rendered into legend labels.
_MISSING = object()


def _canonical_label(value):
    """Map one hierarchy label to a hashable, NA-aware comparison key.

    `dropna=False` keeps a group whose hierarchy LABEL is missing, but that
    group then has to be built and styled as ONE group, and ordinary
    equality cannot do it: ``NaN != NaN``. Measured on the pandas here,
    `np.nan`, `None` and `pd.NA` in a MultiIndex level all normalise to
    plain `float('nan')`, and `groupby` mints a SEPARATE nan object per
    group key -- so a dict keyed on raw labels sees two groups where there
    is one, producing duplicate mean traces, duplicate palette entries and
    duplicate legend entries. (`pd.NaT` is a singleton, so identity
    short-circuiting hides the problem for that spelling alone, and
    `expand_multiindex` avoids it only because `df.index` hands back the
    same object each time -- neither is something to rely on.)

    Every missing spelling canonicalises to the same sentinel, which matches
    what the grouping layer already did: pandas had already merged them into
    one group before these keys were built.

    Use this for COMPARISON and INDEXING only. Callers keep the original
    value for `FinalTraces.keys`, `unique_top` and legend labels.
    """
    try:
        if bool(pd.isna(value)):
            return _MISSING
    except (TypeError, ValueError):
        # array-likes make `pd.isna` return an array, whose truth value is
        # ambiguous; such a label is not missing, it is just not scalar.
        pass
    return value


def _canonical_key(key):
    """Canonicalise every level of a hierarchy key tuple (see above)."""
    return tuple(_canonical_label(value) for value in key)


def is_hierarchical(obj, axes='both'):
    """True when `obj` is a DataFrame carrying a MultiIndex on `axes`."""
    if not isinstance(obj, pd.DataFrame):
        return False
    if axes == 'rows':
        return obj.index.nlevels >= 2
    if axes == 'columns':
        return obj.columns.nlevels >= 2
    return obj.index.nlevels >= 2 or obj.columns.nlevels >= 2


def reject_dual_axis(df):
    """Refuse frames carrying a hierarchy on BOTH axes.

    Which hierarchy should win is genuinely ambiguous. Before 1.1 such a
    frame followed the ROW path and its column hierarchy was silently
    ignored; 1.1 declines to guess. This is an intentional compatibility
    change (see CHANGELOG 1.1.0, "Changed / validation").
    """
    if (isinstance(df, pd.DataFrame)
            and df.index.nlevels >= 2 and df.columns.nlevels >= 2):
        raise ValueError(
            "x has both a row and a column MultiIndex. hypertools 1.1 does "
            "not define which hierarchy takes precedence. Flatten one axis "
            "(e.g. df.reset_index(drop=True), or "
            "df.columns = df.columns.map('_'.join)) and try again.")


def reject_hierarchical_in_list(x, caller, axes='columns'):
    """Refuse a hierarchical DataFrame nested inside a list/tuple.

    Hierarchy expansion is defined for a BARE frame only: the hierarchy
    determines the whole trace/group list, which cannot be reconciled with
    a caller-supplied list of datasets.

    `axes` is DELIBERATELY asymmetric between the two callers (CHANGELOG
    1.1.0, "Changed / validation"):

    - ``'columns'`` (``hyp.plot``): reject a COLUMN hierarchy only. Before
      1.1 it flattened to a single line, silently -- nothing pinned it, so
      rejecting it is purely additive. A ROW hierarchy in a list keeps its
      documented warn-and-flatten path ("MultiIndex grouping is only
      applied..."), which `tests/test_multiindex.py:453` pins and 1.1 does
      not change.
    - ``'both'`` (``hyp.predict``): reject either axis. There is nothing to
      preserve: a row-hierarchical frame in a list raises `TypeError:
      cannot perform __sub__ with this index type: MultiIndex` deep inside
      pandas today, and a column-hierarchical one silently forecasts the
      flattened frame.
    """
    if not isinstance(x, (list, tuple)):
        return
    if axes not in ('columns', 'both'):
        raise ValueError(f"axes= must be 'columns' or 'both'; got {axes!r}")
    for i, element in enumerate(x):
        if not isinstance(element, pd.DataFrame):
            continue
        row_hier = element.index.nlevels >= 2
        col_hier = element.columns.nlevels >= 2
        if col_hier or (row_hier and axes == 'both'):
            axis = ('row' if row_hier else 'column')
            raise ValueError(
                f"{caller} received a list whose element {i} is a DataFrame "
                f"with a {axis} MultiIndex. Hierarchy expansion is defined "
                "for a BARE DataFrame only, because the hierarchy determines "
                "the entire group list. Pass the frame on its own "
                f"({caller}(df, ...)), or flatten it first "
                "(df.reset_index(drop=True), or "
                "df.columns = df.columns.map('_'.join)).")


def group_columns(df):
    """Group a column-hierarchical frame into one leaf per group.

    The innermost column level is the FEATURE axis; every level above it is
    the grouping hierarchy. Returns ``(leaves, meta)`` with `meta` shaped
    exactly like `expand_multiindex`'s (plus ``'axis'``), so the style layer
    consumes either without branching.

    Each leaf is FLATTENED onto the feature axis (Contract 11): its columns
    are the innermost level's values, carrying that level's name. Keeping the
    caller's full tuples would leave the leaf hierarchical, so `hyp.predict`'s
    per-group recursion would re-detect it and regroup without bound.

    Duplicate flattened labels are permitted and are handled POSITIONALLY --
    two share classes of one issuer, or a repeated sensor name, are legitimate
    inputs and nothing downstream is name-addressed. Group labels come from
    `meta['leaf_keys']`, never from a leaf's columns.
    """
    reject_dual_axis(df)
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 2:
        raise ValueError(
            "group_columns requires a column MultiIndex with 2 or more "
            f"levels; got {df.columns.nlevels} level(s).")

    group_levels = list(range(df.columns.nlevels - 1))
    feature_name = df.columns.names[-1]
    leaves, leaf_keys = [], []
    # `df.groupby(..., axis=1)` was REMOVED in pandas 3 (TypeError on the
    # 3.0.3 in this venv), so group the transpose and transpose each group
    # back. sort=False preserves first-appearance order, which `leaf_keys`
    # and the palette both depend on. dropna=False keeps groups whose
    # hierarchy LABEL is missing -- measured on pandas 3.0.3, the default
    # silently turns 3 groups into 2.
    for key, sub in df.T.groupby(level=group_levels, sort=False, dropna=False):
        # COPY FIRST, then flatten. `sub.T` may be a VIEW onto the caller's
        # frame depending on the pandas version and copy-on-write state, so
        # assigning `.columns` to it directly risks silently rewriting the
        # input's columns. (Equivalent and also acceptable: build it outright,
        # `pd.DataFrame(sub.T.to_numpy(), index=sub.columns, columns=flat)`.)
        leaf = sub.T.copy()
        # Contract 11: flatten onto the FEATURE axis. Without this the leaf
        # keeps the full ('Market', 'Sector', 'Ticker') tuples, contradicting
        # the feature-axis rule and making hyp.predict recurse without bound
        # (Revision note (v6) D1). Duplicates in the flattened labels are
        # fine -- see the docstring.
        leaf.columns = leaf.columns.get_level_values(-1)
        leaf.columns.name = feature_name
        leaves.append(leaf)
        leaf_keys.append(key if isinstance(key, tuple) else (key,))

    return leaves, {
        'n_levels': len(group_levels),
        'leaf_keys': leaf_keys,
        'level_names': list(df.columns.names[:-1]),
        'axis': 'columns',
    }


def group_rows_for_forecast(df):
    """Group a row-hierarchical frame for forecasting, KEEPING the time axis.

    The innermost row level is the TIME/observation axis, so grouping uses
    every level above it: a ``(Sector, day)`` index yields one group per
    SECTOR, each a full time series **still indexed by ``day``**. Only the
    grouping levels are dropped -- `reset_index(drop=True)` would discard
    the timestamps `hyp.predict` needs for a datetime-like `t` (see
    `hypertools/predict/common.py`).

    This is intentionally NOT `expand_multiindex`, whose plotting rule is one
    leaf per unique full tuple -- that rule makes every ``(Sector, day)``
    pair its own one-row leaf, which cannot be forecast, and whose leaves
    keep the full row MultiIndex (measured: re-expanding one returns itself).

    `droplevel` is what makes this helper satisfy Contract 11 on the row axis
    AND keep the datetime promise at the same time: only the grouping levels
    go, so the surviving index is FLAT and still carries its own name and
    dtype. No `RangeIndex` fallback is used -- for forecasting the innermost
    level IS the time axis, and replacing a non-monotonic or duplicated one
    with positions would hide exactly what the warning/rejection below is for.
    """
    reject_dual_axis(df)
    if df.index.nlevels < 2:
        raise ValueError(
            "group_rows_for_forecast requires a row MultiIndex with 2 or "
            f"more levels; got {df.index.nlevels} level(s).")

    group_levels = list(range(df.index.nlevels - 1))
    groups, keys = [], []
    for key, sub in df.groupby(level=group_levels, sort=False, dropna=False):
        # droplevel, NOT reset_index: the innermost level survives as this
        # group's index, carrying its name and dtype (verified: a datetime
        # innermost level comes back as a DatetimeIndex named 'date').
        groups.append(sub.droplevel(group_levels))
        keys.append(key if isinstance(key, tuple) else (key,))
    return groups, keys
