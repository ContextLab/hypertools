#!/usr/bin/env python
"""Robust color mapping for hypertools.

One pathway that turns *anything describable as color information* into RGB:
categorical labels, continuous values, soft cluster assignments / mixture
proportions, or arbitrary numeric matrices. Ported from the hypertools revamp
design (jeremymanning/hypertools issues #10, #11, #24) and reimplemented
against current numpy/seaborn.
"""

import collections.abc
import warnings

import numpy as np
import pandas as pd

# neutral color for observations whose hue value is non-finite (NaN/inf):
# a light gray that reads as "no information" next to any palette, so a
# missing value can never silently masquerade as a real data color
# (release-1.0 audit, F02-001/F24-004)
NAN_COLOR = (0.75, 0.75, 0.75)


def is_missing_label(value):
    """True for every spelling of "no label given": `None`, NaN, `pd.NA`.

    NaN needs saying because it is not equal to itself, so two missing
    labels group as two DIFFERENT categories -- and, since `np.nan` is a
    singleton while `float('nan')` is a fresh object each time, whether they
    did depended on how the caller happened to spell it. Callers normalize
    every one of these to `None`, the sentinel this module and `plot()`
    already use for "unlabeled" (drawn `NAN_COLOR`, no legend entry).
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        # pd.isna returns an ARRAY for array-like input, and raises for some
        # exotic types; neither is a missing scalar
        return False


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
          proportion-weighted blend of the k component colors. If ANY entry
          in the matrix is negative, every row is first shifted to be
          non-negative by subtracting its own minimum (negative "weights"
          have no color meaning) -- note this means a signed matrix's rows
          are colored by their within-row CONTRAST, not their absolute
          values, and a 2-column signed row can only ever produce a pure
          component color or the uniform blend. For arbitrary
          high-dimensional matrices, reduce to 3 columns first (see
          ``plot()``'s ``color_reduce=``).
    palette : str, list of colors, or matplotlib.colors.Colormap
        Base colors: a seaborn/matplotlib palette name, an explicit list of
        colors (hex strings like '#ff0000', named colors like 'red', or
        RGB(A) tuples), or a matplotlib Colormap instance (sampled evenly).
        Default: 'hls'. For CONTINUOUS (1D numeric) input, a list shorter
        than `n_bins` is blended into a smooth gradient using the listed
        colors as anchors (seaborn ``blend_palette`` semantics); for
        categorical/matrix input the list must supply at least one color
        per category/column.
    n_bins : int
        Resolution used when binning continuous 1D values (default: 100).
        Must be a positive integer.

    Returns
    -------
    colors : numpy.ndarray
        (n_samples, 3) array of RGB values in [0, 1].

    Notes
    -----
    Non-finite entries (NaN/inf) never corrupt the mapping of the finite
    values: a 1D value that is non-finite (or a 2D row containing any
    non-finite entry) is colored the neutral gray ``NAN_COLOR`` (0.75,
    0.75, 0.75) and excluded from the value range / blend, and a
    ``UserWarning`` reports how many observations were affected.
    Previously a single NaN silently collapsed the ENTIRE 1D gradient to
    one color (release-1.0 audit, F02-001/F24-004).
    """
    import seaborn as sns

    if isinstance(m, pd.DataFrame):
        m = m.values
    elif isinstance(m, collections.abc.Iterator):
        # generators and other one-shot iterators: materialize so the
        # classification below (which iterates more than once) sees the
        # actual values instead of crashing deep in numpy (F24-010)
        m = list(m)
    if np.isscalar(m) and not isinstance(m, str):
        raise ValueError(
            "mat2colors requires a sequence of labels/values (or a 2D "
            f"matrix with one row per sample); got a scalar: {m!r}")
    if isinstance(m, np.ndarray) and m.ndim == 0:
        raise ValueError(
            "mat2colors requires a sequence of labels/values (or a 2D "
            f"matrix with one row per sample); got a 0-dimensional array: "
            f"{m!r}")
    if len(m) == 0:
        raise ValueError(
            "mat2colors got an empty input; pass at least one label/value "
            "(or one matrix row) per sample.")
    if (not isinstance(n_bins, (int, np.integer))
            or isinstance(n_bins, bool) or n_bins < 1):
        raise ValueError(
            f"n_bins= must be a positive integer; got {n_bins!r}")

    # categorical labels (list of strings, or anything non-numeric)
    if not _is_numeric(m):
        labels = _flatten_if_nested(m)
        categories = list(sorted(set(labels), key=list(labels).index))
        base = _get_palette(palette, len(categories), sns)
        return np.asarray(
            [base[categories.index(label)] for label in labels], dtype=float)

    m = np.asarray(m, dtype=np.float64)

    if m.ndim == 1 or (m.ndim == 2 and m.shape[1] == 1):
        vals = m.ravel()
        finite = np.isfinite(vals)
        colors = np.full((vals.size, 3), NAN_COLOR, dtype=float)
        if finite.any():
            base = np.asarray(
                _continuous_palette(palette, n_bins, sns), dtype=float)[:, :3]
            fvals = vals[finite]
            # edges from the FINITE values only, so a NaN can never
            # poison min/max and collapse the whole gradient (F02-001)
            edges = np.linspace(np.min(fvals), np.max(fvals), n_bins + 1)
            ranks = np.clip(np.digitize(fvals, edges) - 1, 0, n_bins - 1)
            colors[finite] = base[ranks]
        if not finite.all():
            _warn_non_finite(np.count_nonzero(~finite))
        return colors

    if m.ndim == 2:
        # soft assignments / mixture proportions: each row is a weight vector
        # over the k component colors, and the sample's color is that
        # proportion-weighted blend. Rows with any non-finite entry are
        # colored NAN_COLOR (with a warning) instead of silently becoming
        # the uniform blend (F24-004).
        finite_rows = np.isfinite(m).all(axis=1)
        colors = np.full((m.shape[0], 3), NAN_COLOR, dtype=float)
        if finite_rows.any():
            weights = m[finite_rows].astype(np.float64, copy=True)
            if weights.min() < 0:
                # SIGNED matrix (e.g. an arbitrary embedding): shift each
                # row to be non-negative before normalizing -- negative
                # "weights" have no color meaning. (For matrices with >3
                # columns, callers typically reduce to a 3-column RGB matrix
                # first; see plot()'s color_reduce=.)
                weights = weights - np.min(weights, axis=1, keepdims=True)
            # else: NON-NEGATIVE rows (mixture proportions / soft cluster
            # assignments) are used AS-IS so a [0.5, 0.5] row blends the two
            # component colors 50/50. The old code unconditionally subtracted
            # the per-row min BEFORE normalizing, which collapsed every
            # non-tied row onto its argmax vertex (pure component color) --
            # so mixture proportions never actually blended (QC 2026-07).
            row_sums = weights.sum(axis=1, keepdims=True)
            # rows that sum to zero get uniform weights
            weights = np.where(row_sums > 0,
                               weights / np.where(row_sums == 0, 1, row_sums),
                               1.0 / m.shape[1])
            base = np.asarray(
                _get_palette(palette, m.shape[1], sns), dtype=float)[:, :3]
            colors[finite_rows] = np.clip(weights @ base, 0, 1)
        if not finite_rows.all():
            _warn_non_finite(np.count_nonzero(~finite_rows))
        return colors

    raise ValueError(f'mat2colors requires 1D or 2D input; got {m.ndim}D')


def _warn_non_finite(n_bad):
    warnings.warn(
        f"{n_bad} observation(s) have non-finite (NaN/inf) hue/color "
        f"values; they are drawn in a neutral gray {NAN_COLOR} and "
        "excluded from the color mapping (the remaining observations "
        "keep their full color range).",
        UserWarning,
        stacklevel=3,
    )


def colors2groups(colors, res=6):
    """Quantize an (n, 3) RGB array into discrete group ids.

    hypertools' matplotlib renderer draws one trace per group, so
    per-observation colors are approximated by quantizing each RGB channel to
    `res` levels (res=6 -> at most 216 distinct groups) and grouping
    observations that share a quantized color. `res` must be an integer >= 2
    (a single quantization level cannot distinguish any colors).

    Returns
    -------
    group_ids : list of tuples
        One hashable quantized-RGB tuple per observation (usable as a `hue`
        grouping), and
    group_colors : dict
        Mapping from each group id to the mean true color of its members.
    """
    if (not isinstance(res, (int, np.integer)) or isinstance(res, bool)
            or res < 2):
        raise ValueError(
            f"res= must be an integer >= 2 (number of quantization levels "
            f"per RGB channel); got {res!r}")
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


def get_palette_colors(palette, n_colors):
    """Resolve `n_colors` RGB colors from a palette (name, list of colors,
    or matplotlib Colormap).

    Public wrapper around `_get_palette` so callers outside this module
    (e.g. colorbar construction in plot.py/plotly_backend.py) can build a
    color mapping that is GUARANTEED to match what `mat2colors` /
    `sns.set_palette` would produce for the same `palette`/`n_colors` --
    the single source of truth for "what color is group i / value v".
    Returns a (n_colors, 3) float array; n_colors=0 returns an empty
    (0, 3) array.
    """
    import seaborn as sns
    if (not isinstance(n_colors, (int, np.integer))
            or isinstance(n_colors, bool) or n_colors < 0):
        raise ValueError(
            f"n_colors= must be a non-negative integer; got {n_colors!r}")
    if n_colors == 0:
        return np.empty((0, 3), dtype=float)
    return np.asarray(_get_palette(palette, n_colors, sns),
                      dtype=float)[:, :3]


def continuous_colormap(palette, n_bins=100):
    """Build a matplotlib ListedColormap matching `mat2colors`'s continuous
    path (same palette, same default `n_bins`, same cyclic-palette
    trimming), so a colorbar built from this colormap shows EXACTLY the
    colors used for continuously-hued lines/markers."""
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    return ListedColormap(
        np.asarray(_continuous_palette(palette, n_bins, sns),
                   dtype=float)[:, :3])


#: How many anchor colors `palette='image:<path>'` extracts for a CONTINUOUS
#: mapping, which asks `_get_palette` for `n_bins` (100) colors -- clustering
#: an image into 100 groups is both slow and meaningless, so it takes this
#: few and lets the short-list blending (the `continuous` arm of
#: `_get_palette`'s `len(colors) < n_colors` branch) build the gradient. A
#: CATEGORICAL or matrix mapping instead extracts exactly as many colors as it
#: has categories, so the number of groups is NOT capped at this value; see
#: `_image_palette_list`.
IMAGE_PALETTE_N = 6

#: Prefix that marks a `palette=` string as "extract this from an image".
#: Seaborn/matplotlib palette names never contain a colon, so there is no
#: collision; an unmatched name still reaches seaborn and raises its own
#: "is not a valid palette name" error.
IMAGE_PALETTE_PREFIX = 'image:'

#: Below this chroma (max(RGB) - min(RGB)) an image has no colour to be
#: salient ABOUT, so `image_palette` orders by population instead.
_ACHROMATIC_EPS = 0.02


def _image_pixels(image, resize):
    """(n_pixels, 3) float RGB in [0, 1] from a path, PIL image, or array."""
    import os

    from PIL import Image

    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype.kind == 'f':
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        im = Image.fromarray(arr.astype(np.uint8)).convert('RGB')
    elif hasattr(image, 'convert'):          # a PIL.Image.Image
        im = image.convert('RGB')
    else:
        path = os.path.expanduser(os.fspath(image))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"image_palette() could not find an image at {path!r}. It "
                "takes a LOCAL path, a PIL image, or an (H, W, 3) array -- "
                "hypertools never downloads the image for you, so fetch and "
                "cache it yourself first.")
        im = Image.open(path).convert('RGB')
    im.thumbnail((int(resize), int(resize)))
    return np.asarray(im, dtype=np.float64).reshape(-1, 3) / 255.0


def image_palette(image, n_colors=IMAGE_PALETTE_N, resize=200, random_state=0):
    """Extract a color palette from an image, most VISUALLY SALIENT first.

    Parameters
    ----------
    image : str, pathlib.Path, PIL.Image.Image, or numpy array
        A LOCAL image file, an already-open PIL image, or an (H, W, 3) array
        (uint8 0-255, or float 0-1). URLs are deliberately not accepted:
        hypertools does not fetch images, so download and cache the file
        yourself and pass the cached path.
    n_colors : int
        UPPER bound on how many colors to return (default 6). Fewer come
        back when the image has fewer distinct colors, or when two cluster
        centers coincide to 3 decimal places.
    resize : int
        Longest edge the image is thumbnailed to before clustering
        (default 200). Clustering cost is linear in pixel count.
    random_state : int
        Seed for the k-means fit, so repeated calls are identical.

    Returns
    -------
    palette : numpy.ndarray
        (k, 3) float RGB in [0, 1], k <= n_colors, ordered most salient
        first.

    Notes
    -----
    Salience is ``pixel_fraction * chroma``, where
    ``chroma = max(r, g, b) - min(r, g, b)`` measures distance from grey.
    Ordering by pixel fraction ALONE returns a painting's background --
    which is exactly the bug this function exists to avoid. When every
    cluster is achromatic (max chroma < 0.02, i.e. a greyscale image) the
    ordering falls back to pixel fraction, because a grey image has no
    vivid color and "largest" is then the right answer.

    Examples
    --------
    >>> from hypertools.plot.colors import image_palette
    >>> image_palette('starry_night.jpg')[0]        # doctest: +SKIP
    array([0.16, 0.24, 0.55])

    The same extraction is reachable declaratively from any plotting call
    that takes a palette::

        hyp.plot(x, hue=values, palette='image:starry_night.jpg')
    """
    from sklearn.cluster import KMeans

    if (not isinstance(n_colors, (int, np.integer))
            or isinstance(n_colors, bool) or n_colors < 1):
        raise ValueError(
            f"n_colors= must be a positive integer; got {n_colors!r}")
    px = _image_pixels(image, resize)
    if len(px) == 0:
        raise ValueError("image_palette() got an image with no pixels")
    # cap k at the number of DISTINCT colors: asking k-means for more
    # clusters than there are distinct points emits a ConvergenceWarning
    # and returns duplicate centers
    k = int(min(n_colors, len(np.unique(px, axis=0))))
    km = KMeans(n_clusters=k, n_init=4, random_state=random_state).fit(px)
    centers = np.clip(km.cluster_centers_, 0.0, 1.0)
    frac = np.bincount(km.labels_, minlength=k) / len(px)
    chroma = centers.max(axis=1) - centers.min(axis=1)
    score = frac if chroma.max() < _ACHROMATIC_EPS else frac * chroma
    out, seen = [], set()
    for i in np.argsort(-score, kind='stable'):
        key = tuple(np.round(centers[i], 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(centers[i])
    return np.asarray(out, dtype=float)


# palettes that wrap the hue circle end-to-end: sampling them over the full
# cycle makes a continuous mapping's minimum and maximum values visually
# identical (release-1.0 audit, F01-013)
_CYCLIC_PALETTES = ('hls', 'husl')


def _continuous_palette(palette, n_colors, sns):
    """Colors for CONTINUOUS value mapping (`mat2colors`'s 1D path and the
    matching `continuous_colormap`). Cyclic palettes ('hls', 'husl') are
    sampled over ~5/6 of the hue circle instead of the full cycle, so the
    endpoints of a continuous hue (e.g. a time-colored trajectory's start
    and end, or a colorbar's two ends) stay clearly distinguishable
    (release-1.0 audit, F01-013: both mapped to near-identical red).
    Categorical mappings are unaffected -- they use `_get_palette`, whose
    evenly-spaced full-circle samples never place two categories at the
    same hue. List palettes SHORTER than `n_colors` are blended into a
    smooth `n_colors`-step gradient anchored at the listed colors
    (seaborn ``blend_palette`` semantics; F02-006/F24-017)."""
    if isinstance(palette, str) and palette.lower() in _CYCLIC_PALETTES:
        n_ext = int(np.ceil(n_colors * 6 / 5))
        return list(sns.color_palette(palette, n_ext))[:n_colors]
    return _get_palette(palette, n_colors, sns, continuous=True)


def _image_palette_list(source, n_colors, sns, continuous):
    """Colors for a `palette='image:<path>'` string, as a list `_get_palette`
    can then handle exactly like any other color list.

    How many colors are EXTRACTED depends on the mapping. A categorical or
    matrix mapping needs one color per category, so exactly `n_colors`
    anchors are pulled: k-means with k = the number of categories is the
    best k-color summary of that image, and extracting a FIXED count would
    instead cap every plot at that many categories. A CONTINUOUS mapping
    asks for `n_bins` (100) colors, and clustering an image into 100 groups
    is both slow and meaningless, so it takes `IMAGE_PALETTE_N` anchors and
    lets the short-list blending below build the gradient from them.

    An image can hold FEWER distinct colors than there are categories (a
    two-tone image, nine groups). Unlike a user-supplied short list -- which
    raises, because the user can simply pass more colors -- a caller cannot
    add colors to an image, so the anchors are interpolated up to `n_colors`
    with the same ``blend_palette`` semantics the continuous path already
    uses (F02-006/F24-017). Interpolating keeps every category a DIFFERENT
    color and leaves the most salient anchor first; cycling the anchors
    would silently give two categories the same color, which is the
    ambiguity the short-list error exists to prevent. A single-color image
    is the one case interpolation cannot serve, and it raises."""
    colors = [tuple(c) for c in image_palette(
        source, n_colors=IMAGE_PALETTE_N if continuous else max(n_colors, 1))]
    if continuous or len(colors) >= n_colors:
        return colors
    if len(colors) == 1:
        raise ValueError(
            f"palette='{IMAGE_PALETTE_PREFIX}{source}' yielded 1 color but "
            f"{n_colors} are required (one per category/component); that "
            "image has a single dominant color, so pass a more colorful "
            "image, an explicit list of colors, or a palette name")
    return [tuple(np.asarray(c)[:3])
            for c in sns.blend_palette(colors, n_colors)]


def _get_palette(palette, n_colors, sns, continuous=False):
    """Resolve `palette` into a list of >= `n_colors` RGB tuples.

    `palette` may be a seaborn/matplotlib palette NAME, an explicit list
    of colors (any matplotlib color spec: hex strings, named colors,
    RGB(A) tuples -- normalized to RGB floats so every downstream path
    gets the same (r, g, b) form; F02-005/F24-006), or a matplotlib
    `Colormap` (sampled evenly over [0, 1]). A color LIST shorter than
    `n_colors` raises for categorical/matrix use (each category needs its
    own color) but is smoothly blended when `continuous=True` (the 1D
    continuous mapping, where a short list naturally means "gradient
    anchor colors")."""
    from matplotlib.colors import Colormap, to_rgb

    if palette is None:
        raise ValueError(
            "palette= must be a seaborn/matplotlib palette name, a list of "
            "colors, or a matplotlib Colormap; got None")
    if isinstance(palette, str):
        if palette.startswith(IMAGE_PALETTE_PREFIX):
            # resolve to a color LIST and fall through to the list handling
            # below, so a continuous mapping blends the extracted anchors
            # into its gradient exactly as it would any short list
            palette = _image_palette_list(
                palette[len(IMAGE_PALETTE_PREFIX):].strip(),
                n_colors, sns, continuous)
        else:
            return sns.color_palette(palette, n_colors)
    if isinstance(palette, Colormap):
        if n_colors == 1:
            return [tuple(np.asarray(palette(0.5))[:3])]
        return [tuple(np.asarray(c)[:3])
                for c in palette(np.linspace(0.0, 1.0, n_colors))]
    try:
        colors = [to_rgb(c) for c in palette]
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "palette= must be a seaborn/matplotlib palette name, a list of "
            "colors (hex strings, named colors, or RGB(A) tuples), or a "
            f"matplotlib Colormap; could not interpret {palette!r} "
            f"({exc})") from exc
    if len(colors) == 0:
        raise ValueError("palette= was given as an empty list; supply at "
                         "least one color")
    if len(colors) < n_colors:
        if continuous:
            # short list + continuous mapping: treat the listed colors as
            # gradient anchors and blend them into the full-resolution
            # gradient (seaborn blend_palette semantics)
            if len(colors) == 1:
                return [colors[0]] * n_colors
            return [tuple(np.asarray(c)[:3])
                    for c in sns.blend_palette(colors, n_colors)]
        raise ValueError(
            f"palette= supplies {len(colors)} color(s) but {n_colors} are "
            "required (one per category/component); pass at least "
            f"{n_colors} colors, a palette name, or a matplotlib Colormap")
    return colors[:n_colors]


# Legacy continuous-color helpers live in _shared.helpers (import *-ed widely);
# re-export them here so plot.colors is the single coloring surface.
from .._shared.helpers import vals2colors, vals2bins  # noqa: F401,E402
