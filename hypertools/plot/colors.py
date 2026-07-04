#!/usr/bin/env python
"""Robust color mapping for hypertools.

One pathway that turns *anything describable as color information* into RGB:
categorical labels, continuous values, soft cluster assignments / mixture
proportions, or arbitrary numeric matrices. Ported from the hypertools revamp
design (jeremymanning/hypertools issues #10, #11, #24) and reimplemented
against current numpy/seaborn.
"""

import numpy as np
import pandas as pd


def mat2colors(m, palette='hls', n_bins=100):
    """Map labels, values, or matrices to RGB colors.

    Parameters
    ----------
    m : array-like
        One of:
        - list of categorical labels (strings or mixed hashables): each
          category gets its own color from the palette
        - 1D numeric array: values are binned and mapped through a continuous
          version of the palette
        - 2D numeric array (n_samples x k): each row is treated as a soft
          assignment over k components (e.g. mixture proportions). Rows are
          normalized to sum to 1 and each sample's color is the
          proportion-weighted blend of the k component colors.
    palette : str or list
        Seaborn palette name (or list of RGB tuples) supplying the base
        colors (default: 'hls').
    n_bins : int
        Resolution used when binning continuous 1D values (default: 100).

    Returns
    -------
    colors : numpy.ndarray
        (n_samples, 3) array of RGB values in [0, 1].
    """
    import seaborn as sns

    if isinstance(m, pd.DataFrame):
        m = m.values

    # categorical labels (list of strings, or anything non-numeric)
    if not _is_numeric(m):
        labels = _flatten_if_nested(m)
        categories = list(sorted(set(labels), key=list(labels).index))
        base = _get_palette(palette, len(categories), sns)
        return np.asarray([base[categories.index(label)] for label in labels])

    m = np.asarray(m, dtype=np.float64)

    if m.ndim == 1 or (m.ndim == 2 and m.shape[1] == 1):
        vals = m.ravel()
        base = _get_palette(palette, n_bins, sns)
        edges = np.linspace(np.min(vals), np.max(vals), n_bins + 1)
        ranks = np.clip(np.digitize(vals, edges) - 1, 0, n_bins - 1)
        return np.asarray([base[r] for r in ranks])

    if m.ndim == 2:
        # soft assignments / mixture proportions: blend component colors.
        # Shift rows to be non-negative before normalizing so arbitrary
        # matrices (e.g. embeddings) also produce valid blends.
        weights = m - np.min(m, axis=1, keepdims=True)
        row_sums = weights.sum(axis=1, keepdims=True)
        # rows that sum to zero (uniform after shift) get uniform weights
        weights = np.where(row_sums > 0, weights / np.where(row_sums == 0, 1, row_sums),
                           1.0 / m.shape[1])
        base = np.asarray(_get_palette(palette, m.shape[1], sns))[:, :3]
        return np.clip(weights @ base, 0, 1)

    raise ValueError(f'mat2colors requires 1D or 2D input; got {m.ndim}D')


def colors2groups(colors, res=6):
    """Quantize an (n, 3) RGB array into discrete group ids.

    hypertools' matplotlib renderer draws one trace per group, so
    per-observation colors are approximated by quantizing each RGB channel to
    `res` levels (res=6 -> at most 216 distinct groups) and grouping
    observations that share a quantized color.

    Returns
    -------
    group_ids : list of tuples
        One hashable quantized-RGB tuple per observation (usable as a `hue`
        grouping), and
    group_colors : dict
        Mapping from each group id to the mean true color of its members.
    """
    colors = np.asarray(colors)
    quantized = np.round(colors * (res - 1)) / (res - 1)
    group_ids = [tuple(row) for row in quantized]
    group_colors = {}
    for gid in sorted(set(group_ids), key=group_ids.index):
        members = colors[[i for i, g in enumerate(group_ids) if g == gid]]
        group_colors[gid] = tuple(members.mean(axis=0))
    return group_ids, group_colors


def _is_numeric(m):
    if isinstance(m, np.ndarray):
        return np.issubdtype(m.dtype, np.number)
    try:
        flat = _flatten_if_nested(m)
        return all(isinstance(el, (int, float, np.integer, np.floating))
                   and not isinstance(el, bool) for el in flat)
    except TypeError:
        return False


def _flatten_if_nested(vals):
    if any(isinstance(el, (list, np.ndarray)) for el in vals):
        return [item for el in vals for item in np.atleast_1d(el)]
    return list(vals)


def _get_palette(palette, n_colors, sns):
    if isinstance(palette, str):
        return sns.color_palette(palette, n_colors)
    palette = list(palette)
    if len(palette) < n_colors:
        raise ValueError(f'palette supplies {len(palette)} colors but '
                         f'{n_colors} are required')
    return palette[:n_colors]


# Legacy continuous-color helpers live in _shared.helpers (import *-ed widely);
# re-export them here so plot.colors is the single coloring surface.
from .._shared.helpers import vals2colors, vals2bins  # noqa: F401,E402
