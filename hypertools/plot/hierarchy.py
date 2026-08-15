#!/usr/bin/env python
"""Final-trace construction for hierarchical plots.

`build_hierarchy_traces` is the SINGLE owner of:
  * per-level mean construction,
  * truncation of unequal-length members to their overlapping prefix,
  * the one aggregated unequal-length warning,
  * co-truncation of any auxiliary per-observation values (hue).

Nothing else in hypertools may append a mean trace. Grouping lives in
`hypertools.core.hierarchy`; styling lives in `build_hierarchy_styles`
below, which consumes this module's METADATA and never sees leaf arrays.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

from .colors import get_palette_colors


@dataclass
class FinalTraces:
    """The complete, ordered list of trajectories a hierarchy will draw.

    Attributes
    ----------
    arrays : list of numpy.ndarray
        Leaves first (in ``meta['leaf_keys']`` order), then the derived means
        deepest-first, so the top-level mean is last. Always plain
        `ndarray`s -- never DataFrames (see `build_hierarchy_traces`).
    keys : list of tuple
        The hierarchy key each trace belongs to: the full leaf key for a
        leaf, the group prefix for a mean.
    level_idx : list of int
        The depth the documented style formulas use: ``n_levels - 1`` for a
        leaf, ``k`` for a mean over levels ``0..k``.
    is_mean : list of bool
        Whether each trace was derived rather than supplied.
    aux : list or None
        Per-observation auxiliary values (hue today), co-truncated with the
        data by the same operation, or None when none were supplied.
    meta : dict
        The grouping metadata this list was built from.
    """

    arrays: list
    keys: list
    level_idx: list
    is_mean: list
    aux: list = None
    meta: dict = field(default_factory=dict)

    def assert_consistent(self, **named_sequences):
        """Raise naming any sequence whose length != len(self.arrays)."""
        n = len(self.arrays)
        for name, seq in named_sequences.items():
            if seq is not None and len(seq) != n:
                raise ValueError(
                    f"hierarchy trace/{name} mismatch: {n} traces but "
                    f"{len(seq)} {name}. Every per-trace sequence must be "
                    "built from the same FinalTraces "
                    "(hypertools/plot/hierarchy.py).")


def build_hierarchy_traces(leaf_arrays, meta, aux=None):
    """Build the final trace list for a hierarchy: leaves, then per-level means.

    This is the ONLY place a mean trajectory is constructed. Before 1.1 the
    styling function did it as a side effect, so any second builder appended
    every mean twice (F1).

    Parameters
    ----------
    leaf_arrays : list of array-like
        The TRANSFORMED (post normalize/reduce/align) leaf trajectories, in
        the same order as ``meta['leaf_keys']``. Row-hierarchy leaves arrive
        as DataFrames; see the coercion note below.
    meta : dict
        Grouping metadata from `expand_multiindex` or
        `hypertools.core.hierarchy.group_columns`.
    aux : list of array-like, or None
        One per-observation auxiliary array per leaf (hue values today).
        Means get the mean of their members' aux, sliced to the same
        overlap as the data, so an auxiliary value can never drift out of
        step with the trace it describes (Contract 6).

    Returns
    -------
    FinalTraces

    Notes
    -----
    Every leaf is coerced with ``np.asarray`` on the way in, so `arrays`
    holds plain `ndarray`s even when the caller passed DataFrames. That is
    what makes Contract 11 hold BY CONSTRUCTION: `expand_multiindex`'s row
    leaves keep the full row MultiIndex and are a measured fixed point, so
    one surviving into a forecast call would be re-detected as hierarchical
    and regrouped without bound. `np.asarray` drops the index and column
    labels, leaving nothing to re-detect.

    The coercion is deliberately NOT ``np.asarray(leaf).copy()``. The
    contract is narrow -- trace leaves are plain `ndarray`s, and nothing in
    this chain mutates its inputs -- and a blanket copy would double peak
    memory for no gain against it. (Incidentally, on the pandas installed
    here copy-on-write makes `np.asarray(df)` a non-writeable view, so a
    stray write raises; that is version-specific behaviour, not a guarantee.
    Any future code that needs to WRITE into an `arrays` member must copy at
    that point and say why.)

    Member leaves of a prefix group are averaged over their overlapping
    (shortest) length when they are of unequal length. In a 3+-level tree
    the same short leaf belongs to the prefix group at every level above it,
    so rather than warn once per level for one underlying issue, every
    unequal-length group found across every level is reported in a single
    aggregated ``UserWarning`` per call.
    """
    leaf_keys = meta['leaf_keys']
    n_levels = meta['n_levels']
    n_leaves = len(leaf_keys)
    if len(leaf_arrays) != n_leaves:
        raise ValueError(
            f"build_hierarchy_traces got {len(leaf_arrays)} leaf array(s) "
            f"but expected {n_leaves} (one per unique MultiIndex "
            "combination)."
        )

    # Coerce on the way in -- see the Notes above. Do NOT write
    # `arrays = list(leaf_arrays)`: that preserves DataFrames.
    arrays = [np.asarray(leaf) for leaf in leaf_arrays]
    aux_out = None if aux is None else [np.asarray(a) for a in aux]

    keys = list(leaf_keys)
    level_idx = [n_levels - 1] * n_leaves
    is_mean = [False] * n_leaves

    # level-k means, k = n_levels - 2 down to 0 (deepest non-leaf level up to
    # the top level) -- appended in that order so top-level means (the
    # thickest, most opaque, only legend-labeled traces) come last. Members
    # are always LEAVES, never previously derived means.
    _unequal_length_groups = []
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
            member_arrays = [arrays[i] for i in member_idx]
            lengths = [a.shape[0] for a in member_arrays]
            min_len = min(lengths)
            if len(set(lengths)) > 1:
                group_name = prefix[0] if len(prefix) == 1 else prefix
                _unequal_length_groups.append((group_name, lengths, min_len))
            stacked = np.stack([a[:min_len] for a in member_arrays], axis=0)
            arrays.append(np.mean(stacked, axis=0))
            if aux_out is not None:
                # the SAME min_len slice as the data (Contract 6)
                aux_stacked = np.stack([aux_out[i][:min_len]
                                        for i in member_idx], axis=0)
                aux_out.append(np.mean(aux_stacked, axis=0))
            keys.append(prefix)
            level_idx.append(k)
            is_mean.append(True)

    if _unequal_length_groups:
        details = "; ".join(
            f"{group_name!r} has members of unequal length ({lengths}), "
            f"averaged over the overlapping prefix of {min_len} row(s)"
            for group_name, lengths, min_len in _unequal_length_groups
        )
        warnings.warn(
            f"MultiIndex group(s) with unequal-length members: {details}."
        )

    return FinalTraces(arrays=arrays, keys=keys, level_idx=level_idx,
                       is_mean=is_mean, aux=aux_out, meta=meta)


def build_hierarchy_styles(traces, palette='hls', linestyle=None,
                           linestyles=None):
    """Per-trace color/linewidth/alpha/linestyle/label from trace METADATA.

    Consumes a `FinalTraces`' `keys`, `level_idx` and `is_mean` -- never its
    arrays -- so it structurally cannot construct or append a trace.

    Label rule. Only the TOP-level mean (``level_idx == 0`` and `is_mean`)
    carries a legend label. When ``meta['n_levels'] == 1`` there IS no mean:
    each leaf is itself a top-level group, so each carries its own label.
    Without this, a two-level (Group, Feature) column hierarchy drew several
    completely unlabelled traces and an empty legend (F11).

    Parameters
    ----------
    traces : FinalTraces
        The output of `build_hierarchy_traces`.
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
    style : dict
        ``{'colors', 'linewidths', 'alphas', 'labels', 'linestyles',
        'n_top', 'unique_top'}``, one entry per trace (aligned by position).
    """
    n_levels = traces.meta['n_levels']

    top_vals = [key[0] for key in traces.keys]
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

    colors, linewidths, alphas, labels = [], [], [], []
    linestyles_out = [] if per_top_linestyle is not None else None

    for key, level, mean in zip(traces.keys, traces.level_idx, traces.is_mean):
        top_val = key[0]
        linewidths.append(float(1 + (n_levels - 1 - level)))
        alphas.append(float(min(1.0, 1.0 / (level + 1) + 0.2)))
        colors.append(color_of_top[top_index_of[top_val]])
        # Only the TOP-level mean carries a legend label -- except when
        # n_levels == 1, where there IS no mean and each leaf is itself a
        # top-level group. Without that exception a (Group, Feature) column
        # hierarchy drew several completely unlabelled traces (F11).
        top_level = (level == 0) and (mean or n_levels == 1)
        labels.append(str(top_val) if top_level else '_nolegend_')
        if linestyles_out is not None:
            linestyles_out.append(per_top_linestyle[top_index_of[top_val]])

    return {
        'colors': colors,
        'linewidths': linewidths,
        'alphas': alphas,
        'labels': labels,
        'linestyles': linestyles_out,
        'n_top': n_top,
        'unique_top': unique_top,
    }
