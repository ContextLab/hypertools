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
  intermediate-level means) gets ``'_nolegend_'``. The ONE exception is
  ``n_levels == 1`` -- see the one-level example below.

Example (2 levels, (cond, subj)): leaves are (cond, subj) pairs, lw=1,
alpha=0.7; cond-means (the only non-leaf level, which is also the top level)
get lw=2, alpha=1.0, and carry the legend label.

Example (3 levels, (grp, cond, subj)): leaves lw=1, alpha=1/3+0.2=0.5333;
(grp, cond)-means lw=2, alpha=0.7; grp-means (top level) lw=3, alpha=1.0 and
carry the legend label.

Example (``n_levels == 1``, reachable from a two-level COLUMN hierarchy such
as (Group, Feature), where the innermost level is the feature axis): there is
no non-leaf level, so NO mean is built and every leaf is itself a top-level
group -- lw=1, alpha=1.0, its own colour, and **its own legend label**.
Applying the general rule here would leave every trace ``'_nolegend_'`` and
the legend empty, which is what it did before 1.1 (F11).

Since 1.1 the mean-building and styling halves of this module live in
`hypertools.plot.hierarchy` (`build_hierarchy_traces` / `build_hierarchy_styles`);
`build_multiindex_styles` below is a thin shim over that pair, and grouping
itself lives in `hypertools.core.hierarchy`.
"""

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
    """Deprecated internal shim: build traces, then style them.

    Kept because `tests/test_multiindex.py` imports it directly and pins its
    ``(arrays, style)`` contract. New code calls `build_hierarchy_traces` and
    `build_hierarchy_styles` separately -- see hypertools/plot/hierarchy.py,
    which is the single owner of mean construction, unequal-length
    truncation and the truncation warning.
    """
    from .hierarchy import build_hierarchy_styles, build_hierarchy_traces
    ft = build_hierarchy_traces(leaf_arrays, meta)
    return ft.arrays, build_hierarchy_styles(ft, palette=palette,
                                             linestyle=linestyle,
                                             linestyles=linestyles)
