#!/usr/bin/env python
"""Build a column-hierarchical DataFrame (GH #285).

Since 1.1 `hypertools.plot` and `hypertools.predict` read a column
`MultiIndex` as a *hierarchy*: the innermost level is the feature axis and
every level above it groups (see ``docs/hierarchy.rst``). Building such a
frame is a two-step incantation --

    frame = pd.concat({('listeners', f'subject {i + 1}'): df
                       for i, df in enumerate(subjects)}, axis=1)
    frame.columns.names = ['Group', 'Subject', 'Feature']

-- repeated by hand in the hierarchy tutorial and open-coded (as lists of
arrays plus a separately computed group mean) in half a dozen other places.
`stack` is that incantation, with the checks the hierarchy layer would
otherwise make later.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ['stack']

#: Named aggregators for `stack`'s ``aggregate=``. Each is called as
#: ``func(members, axis=0)`` on a (members x rows x features) array, which
#: is also the contract a callable must satisfy.
_AGGREGATORS = {'mean': np.mean, 'median': np.median}


def _default_columns(width):
    return [f'feature {j + 1}' for j in range(width)]


def _as_leaf_frame(obj, key):
    """One dataset -> a flat-columned DataFrame."""
    where = f"frames{''.join(f'[{k!r}]' for k in key)}"
    if isinstance(obj, pd.DataFrame):
        if obj.columns.nlevels > 1:
            raise ValueError(
                f"{where} already has a column MultiIndex. stack builds the "
                "hierarchy; its inputs must be flat frames or arrays.")
        return obj
    if isinstance(obj, pd.Series):
        name = obj.name if obj.name is not None else 'feature 1'
        return obj.to_frame(name=name)
    values = np.asarray(obj)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(
            f"{where} must be 1- or 2-dimensional (observations x "
            f"features); got {values.ndim} dimension(s).")
    return pd.DataFrame(values, columns=_default_columns(values.shape[1]))


def _flatten(node, prefix, names, out):
    """Walk the nested container, collecting ``(key_tuple, dataset)``."""
    if isinstance(node, Mapping):
        for label, child in node.items():
            _flatten(child, prefix + (label,), names, out)
        return
    if isinstance(node, (list, tuple)) or (
            isinstance(node, Sequence) and not isinstance(node, (str, bytes))):
        labels = names
        if labels is None:
            labels = [f'dataset {i + 1}' for i in range(len(node))]
        elif len(labels) != len(node):
            raise ValueError(
                f"names= has {len(labels)} name(s) but the group at "
                f"{prefix or '(top level)'} holds {len(node)} dataset(s). "
                "names= labels the members of EVERY positional (list) "
                "group, so each must be that long.")
        for label, child in zip(labels, node):
            _flatten(child, prefix + (label,), names, out)
        return
    out.append((prefix, node))


def _aggregator(aggregate):
    """``aggregate=`` -> ``(function, label)``."""
    if isinstance(aggregate, str):
        if aggregate not in _AGGREGATORS:
            raise ValueError(
                f"aggregate= must be one of {tuple(_AGGREGATORS)}, a "
                f"callable, or None; got {aggregate!r}.")
        return _AGGREGATORS[aggregate], aggregate
    if callable(aggregate):
        return aggregate, getattr(aggregate, '__name__', 'aggregate')
    raise TypeError(
        f"aggregate= must be one of {tuple(_AGGREGATORS)}, a callable, or "
        f"None; got {type(aggregate).__name__}.")


def _aggregate_groups(keys, leaves, depth, aggregate):
    """Per-group aggregates, deepest grouping level first.

    The order mirrors `hypertools.plot.hierarchy.build_hierarchy_traces`,
    which appends its derived means from the deepest non-leaf level up to
    the top so that top-level traces come last. A prefix of length ``k`` is
    padded out to a full ``depth``-tuple with the aggregate's own label, so
    every column still carries ``depth + 1`` levels.
    """
    func, label = _aggregator(aggregate)
    existing = set(keys)
    agg_keys, agg_leaves = [], []
    for prefix_len in range(depth - 1, -1, -1):
        seen = []
        for key in keys:
            prefix = key[:prefix_len]
            if prefix not in seen:
                seen.append(prefix)
        for prefix in seen:
            members = [leaf for key, leaf in zip(keys, leaves)
                       if key[:prefix_len] == prefix]
            new_key = prefix + (label,) * (depth - prefix_len)
            if new_key in existing or new_key in agg_keys:
                raise ValueError(
                    f"aggregate={label!r} would add the column group "
                    f"{new_key}, which already exists. Rename that group, "
                    "or aggregate with a differently named callable.")
            stacked = np.stack([leaf.to_numpy() for leaf in members], axis=0)
            values = np.asarray(func(stacked, axis=0))
            if values.shape != members[0].shape:
                raise ValueError(
                    f"aggregate= returned an array of shape {values.shape}; "
                    f"expected {members[0].shape} (it is called as "
                    "aggregate(members, axis=0) on a (members x rows x "
                    "features) array).")
            agg_keys.append(new_key)
            agg_leaves.append(pd.DataFrame(values, index=members[0].index,
                                           columns=members[0].columns))
    return agg_keys, agg_leaves


def stack(frames, names=None, level_names=None, aggregate=None):
    """Stack datasets side by side into a column-hierarchical DataFrame.

    The result is the frame `hypertools.plot` and `hypertools.predict`
    already understand: its **innermost** column level is the feature axis
    and every level above it groups, so ``hyp.plot(stack(...))`` draws one
    trajectory per group plus one derived mean per grouping level above the
    leaves, styled by depth.

    Parameters
    ----------
    frames : dict or list/tuple
        The datasets, nested as deeply as the hierarchy is tall. A dict's
        keys label its groups; a list's members are labelled by `names=` (or
        ``'dataset 1'``, ``'dataset 2'``, ... ). Nesting dicts nests levels:
        ``{'listeners': {'subject 1': df, ...}, 'speakers': {...}}`` gives
        a ``(Group, Subject, Feature)`` hierarchy. Every leaf may be a
        DataFrame, a Series or an array; every branch must be the same depth.

    names : list of str, optional
        Labels for the members of each positional (list/tuple) group. One
        per member; the same names apply to every such group, which is what
        lets ``{'listeners': subjects, 'speakers': subjects}`` share one
        subject naming. Dict groups are labelled by their own keys.

    level_names : list of str, optional
        ``columns.names`` for the result: one name per grouping level plus
        one for the feature level, so ``len(level_names) == depth + 1``.
        Defaults to ``['level 0', ..., 'feature']``.

    aggregate : {'mean', 'median'}, callable or None, optional
        When given, append the aggregate of every group as an extra column
        group, at every level: a ``(Group, Subject, Feature)`` frame gains
        one ``(group, 'mean')`` per group and one ``('mean', 'mean')`` over
        all of them. A callable is called as ``aggregate(members, axis=0)``
        on a (members x rows x features) array.

        Aggregates are **data**, not styling. `hypertools.plot` already
        derives a mean trace per grouping level above the leaves, so pass
        `aggregate=` when you need the aggregate itself -- to forecast it,
        to save it, or to draw it from a *single*-level hierarchy, where
        there is no level above the leaves and so no derived mean. Plotting
        an aggregated multi-level frame draws each mean twice, once as data
        and once as the derived trace (which then also averages over the
        appended group).

    Returns
    -------
    frame : pandas.DataFrame
        ``n_observations`` rows by (datasets x features) columns, with a
        ``depth + 1``-level column MultiIndex.

    Raises
    ------
    ValueError
        If branches differ in depth, if the datasets differ in length, or if
        they do not name the same features.

    Notes
    -----
    Feature correspondence is by NAME, the same rule
    `hypertools.core.hierarchy.group_columns` applies later: every dataset
    must carry the same feature labels, and datasets whose labels are merely
    in a different order are permuted into the first dataset's order (when
    those labels are unique). Arrays are given the shared default labels
    ``'feature 1'``, ``'feature 2'``, ... so they always correspond.

    All datasets must have the same number of rows -- a column hierarchy
    stacks them side by side against one index. That index is the datasets'
    own if every one of them is a DataFrame carrying the same index, and a
    ``RangeIndex`` otherwise.

    Examples
    --------
    Three subjects in one group, from arrays:

    >>> import numpy as np
    >>> from hypertools.tools import stack
    >>> subjects = [np.zeros((100, 5)) + i for i in range(3)]
    >>> frame = stack({'listeners': subjects},
    ...               names=[f'subject {i + 1}' for i in range(3)],
    ...               level_names=['Group', 'Subject', 'Feature'])
    >>> frame.shape
    (100, 15)
    >>> frame.columns.names
    FrozenList(['Group', 'Subject', 'Feature'])
    >>> frame.columns[0]
    ('listeners', 'subject 1', 'feature 1')

    That frame plots as three leaves plus their mean::

        fig = hyp.plot(frame)

    A single-level hierarchy has no derived mean, so ask for one as data:

    >>> sectors = stack({'Tech': np.zeros((10, 3)),
    ...                  'Energy': np.ones((10, 3))},
    ...                 level_names=['Sector', 'Measure'],
    ...                 aggregate='mean')
    >>> sectors.columns.get_level_values('Sector').unique().tolist()
    ['Tech', 'Energy', 'mean']
    >>> float(sectors[('mean', 'feature 1')].iloc[0])
    0.5

    See Also
    --------
    hypertools.plot : reads the column hierarchy this builds.
    hypertools.predict : groups by the same rule.
    """
    if not isinstance(frames, (Mapping, list, tuple)):
        raise TypeError(
            "stack takes a dict (or nested dicts) or a list/tuple of "
            f"datasets; got {type(frames).__name__}.")

    collected = []
    _flatten(frames, (), names, collected)
    if not collected:
        raise ValueError("stack got no datasets to stack.")

    keys = [key for key, _ in collected]
    depths = {len(key) for key in keys}
    if len(depths) > 1:
        raise ValueError(
            "every branch of frames must be the same depth; got depths "
            f"{sorted(depths)} (e.g. {keys[0]} and "
            f"{next(k for k in keys if len(k) != len(keys[0]))}).")
    depth = len(keys[0])

    leaves = [_as_leaf_frame(obj, key) for key, obj in collected]

    lengths = {len(leaf) for leaf in leaves}
    if len(lengths) > 1:
        raise ValueError(
            "a column hierarchy stacks datasets side by side, so they must "
            f"all have the same number of rows; got {sorted(lengths)}.")

    reference = list(leaves[0].columns)
    unique_labels = len(set(reference)) == len(reference)
    for i, (key, leaf) in enumerate(zip(keys, leaves)):
        labels = list(leaf.columns)
        if labels == reference:
            continue
        if unique_labels and sorted(map(str, labels)) == sorted(
                map(str, reference)) and set(labels) == set(reference):
            # the same features in another order: permute into the first
            # dataset's order, exactly as `group_columns` would later.
            leaves[i] = leaf[reference]
            continue
        raise ValueError(
            f"every dataset must name the same features (correspondence is "
            f"by name): {key} has columns {labels}, but {keys[0]} has "
            f"{reference}.")

    index = leaves[0].index
    if not all(isinstance(obj, pd.DataFrame) for _, obj in collected) or \
            not all(leaf.index.equals(index) for leaf in leaves):
        index = pd.RangeIndex(len(leaves[0]))
    leaves = [pd.DataFrame(leaf.to_numpy(), index=index,
                           columns=leaf.columns) for leaf in leaves]

    if aggregate is not None:
        agg_keys, agg_leaves = _aggregate_groups(keys, leaves, depth,
                                                 aggregate)
        keys = keys + agg_keys
        leaves = leaves + agg_leaves

    if level_names is None:
        level_names = [f'level {i}' for i in range(depth)] + ['feature']
    else:
        level_names = list(level_names)
        if len(level_names) != depth + 1:
            raise ValueError(
                f"level_names= needs one name per column level: {depth} "
                f"grouping level(s) plus the feature level = {depth + 1}; "
                f"got {len(level_names)} ({level_names}).")

    # `pd.concat(..., keys=)` rather than a dict, so the group order is the
    # order they were given in (which the palette and the legend follow).
    frame = pd.concat(leaves, axis=1, keys=keys)
    frame.columns.names = level_names
    return frame
