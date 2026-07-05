#!/usr/bin/env python
"""Row-MultiIndex DataFrame expansion for `hypertools.plot.plot` (GH #95).

When a DataFrame with a row MultiIndex (``df.index.nlevels >= 2``) is passed
to `hyp.plot`, it is expanded -- BEFORE the format_data/analyze/reduce
pipeline runs -- into one "leaf" dataset per unique full index combination
(`expand_multiindex`). After that pipeline has transformed the leaves (so
everything below lives in the reduced/plotted space), `build_multiindex_styles`
computes one additional "mean" trajectory per unique value-combination of
each non-leaf level, and returns the per-dataset color/linewidth/alpha/
linestyle/label overrides both drawing backends already understand (the same
`kwargs_list` machinery used by `color=`/`linewidth=`/etc.).

Levels are numbered 0 (top/outermost) .. L-1 (leaf/deepest), where L is
``df.index.nlevels``. For a trace whose deepest represented level is
``level_idx`` (``L - 1`` for a leaf; ``k`` for a mean over the prefix
``levels[0:k+1]``):

- ``linewidth = 1 + (L - 1 - level_idx)`` -- i.e. 1 plus the number of levels
  averaged over. A leaf (``level_idx = L - 1``) always gets linewidth 1; a
  level-0 (top-level) mean gets linewidth ``L`` (the thickest).
- ``alpha = min(1.0, 1 / (level_idx + 1) + 0.2)`` -- leaves are the most
  transparent, top-level means are fully opaque (1.0), with intermediate
  levels smoothly in between.
- ``color`` is assigned purely by the trace's TOP-level (level-0) index
  value, from `palette`, in order of that value's first appearance -- every
  leaf and every mean sharing the same top-level value shares one color.
- ``label``: only the TOP-level mean (``level_idx == 0``) carries a real
  legend label (``str(top_value)``); every other trace (all leaves, and any
  intermediate-level means) gets ``'_nolegend_'``.

Example (2 levels, (cond, subj)): leaves are (cond, subj) pairs, lw=1,
alpha=0.7; cond-means (the only non-leaf level, which is also the top level)
get lw=2, alpha=1.0, and carry the legend label.

Example (3 levels, (grp, cond, subj)): leaves lw=1, alpha=1/3+0.2=0.5333;
(grp, cond)-means lw=2, alpha=0.7; grp-means (top level) lw=3, alpha=1.0 and
carry the legend label.
"""

import warnings

import numpy as np

from .colors import get_palette_colors


def expand_multiindex(df):
    """Expand a DataFrame with a row MultiIndex into one leaf DataFrame per
    unique full index combination (in order of first appearance).

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame whose row index is a `pandas.MultiIndex` with 2 or more
        levels.

    Returns
    -------
    leaf_dfs : list of pandas.DataFrame
        One per unique index tuple, each holding exactly the rows matching
        that tuple, in their original relative order.
    meta : dict
        ``{'n_levels': L, 'leaf_keys': [tuple, ...], 'level_names': [...]}``
        -- consumed by `build_multiindex_styles` once `leaf_dfs` has been
        run through the format_data/analyze/reduce pipeline.
    """
    index = df.index
    n_levels = index.nlevels
    if n_levels < 2:
        raise ValueError(
            "expand_multiindex requires a row MultiIndex with 2 or more "
            f"levels; got {n_levels} level(s)."
        )

    seen = []
    for key in index:
        if key not in seen:
            seen.append(key)

    leaf_dfs = [df.loc[index.isin([key])] for key in seen]

    meta = {
        'n_levels': n_levels,
        'leaf_keys': seen,
        'level_names': list(index.names),
    }
    return leaf_dfs, meta


def build_multiindex_styles(leaf_arrays, meta, palette='hls', linestyle=None,
                            linestyles=None):
    """Compute per-level mean trajectories + per-dataset style overrides.

    Parameters
    ----------
    leaf_arrays : list of numpy.ndarray
        The TRANSFORMED (post normalize/reduce/align) leaf trajectories, in
        the same order as ``meta['leaf_keys']``.
    meta : dict
        The metadata returned by `expand_multiindex`.
    palette : str or list
        Passed to `hypertools.plot.colors.get_palette_colors`; resolves one
        color per unique TOP-level index value.
    linestyle, linestyles : str, list, or None
        If a list/tuple is given (`linestyles` takes priority, mirroring
        `hyp.plot`'s own alias handling), its length MUST equal the number
        of unique top-level index values -- each top-level group then uses
        its entry for every one of its traces (leaves and means alike).
        Raises ``ValueError`` on a length mismatch. A scalar (or None) is
        left untouched -- the caller's existing scalar-broadcast handles it.

    Returns
    -------
    arrays : list of numpy.ndarray
        ``leaf_arrays`` followed by one mean array per non-leaf level
        grouping, ordered from the DEEPEST non-leaf level up to the top
        level (so the top-level means -- the thickest, most opaque, only
        legend-labeled traces -- come last).
    style : dict
        ``{'colors': [...], 'linewidths': [...], 'alphas': [...],
        'labels': [...], 'linestyles': [...] or None, 'n_top': int,
        'unique_top': [...]}``, one entry per array in ``arrays`` (aligned
        by position).
    """
    leaf_keys = meta['leaf_keys']
    n_levels = meta['n_levels']
    n_leaves = len(leaf_keys)
    if len(leaf_arrays) != n_leaves:
        raise ValueError(
            f"build_multiindex_styles got {len(leaf_arrays)} leaf array(s) "
            f"but expected {n_leaves} (one per unique MultiIndex "
            "combination)."
        )

    top_vals = [key[0] for key in leaf_keys]
    unique_top = list(dict.fromkeys(top_vals))
    n_top = len(unique_top)
    top_index_of = {val: i for i, val in enumerate(unique_top)}

    palette_colors = get_palette_colors(palette, n_top)
    color_of_top = [tuple(float(c) for c in palette_colors[i])
                     for i in range(n_top)]

    # linestyle(s): an explicit list/tuple must have length == n_top (one
    # style per top-level group, applied to every trace in that group). A
    # scalar (or None) is left alone -- the existing scalar-broadcast in
    # plot.py already handles that case.
    resolved_linestyle = linestyles if linestyles is not None else linestyle
    per_top_linestyle = None
    if isinstance(resolved_linestyle, (list, tuple)):
        if len(resolved_linestyle) != n_top:
            raise ValueError(
                f"linestyle(s) has {len(resolved_linestyle)} entries but "
                f"there are {n_top} unique top-level MultiIndex value(s) "
                f"({unique_top!r}); pass exactly one linestyle per "
                "top-level group."
            )
        per_top_linestyle = list(resolved_linestyle)

    arrays = list(leaf_arrays)
    colors, linewidths, alphas, labels = [], [], [], []
    linestyles_out = [] if per_top_linestyle is not None else None

    def _append_style(level_idx, top_val, label):
        linewidths.append(float(1 + (n_levels - 1 - level_idx)))
        alphas.append(float(min(1.0, 1.0 / (level_idx + 1) + 0.2)))
        colors.append(color_of_top[top_index_of[top_val]])
        labels.append(label)
        if linestyles_out is not None:
            linestyles_out.append(per_top_linestyle[top_index_of[top_val]])

    # leaves: deepest level (level_idx = n_levels - 1), never legend-labeled
    for key in leaf_keys:
        _append_style(n_levels - 1, key[0], '_nolegend_')

    # level-k means, k = n_levels - 2 down to 0 (deepest non-leaf level up
    # to the top level) -- appended in that order so top-level means (the
    # thickest, most opaque, only legend-labeled traces) come last.
    for k in range(n_levels - 2, -1, -1):
        prefix_members = {}
        prefix_order = []
        for i, key in enumerate(leaf_keys):
            prefix = key[:k + 1]
            if prefix not in prefix_members:
                prefix_members[prefix] = []
                prefix_order.append(prefix)
            prefix_members[prefix].append(i)

        for prefix in prefix_order:
            member_idx = prefix_members[prefix]
            member_arrays = [np.asarray(arrays[i]) for i in member_idx]
            lengths = [a.shape[0] for a in member_arrays]
            min_len = min(lengths)
            if len(set(lengths)) > 1:
                group_name = prefix[0] if len(prefix) == 1 else prefix
                warnings.warn(
                    f"MultiIndex group {group_name!r} has members of "
                    f"unequal length ({lengths}); averaging over the "
                    f"overlapping prefix of {min_len} row(s)."
                )
            stacked = np.stack([a[:min_len] for a in member_arrays], axis=0)
            arrays.append(np.mean(stacked, axis=0))

            label = str(prefix[0]) if k == 0 else '_nolegend_'
            _append_style(k, prefix[0], label)

    style = {
        'colors': colors,
        'linewidths': linewidths,
        'alphas': alphas,
        'labels': labels,
        'linestyles': linestyles_out,
        'n_top': n_top,
        'unique_top': unique_top,
    }
    return arrays, style
