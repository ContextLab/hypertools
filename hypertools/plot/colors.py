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

#: The palette every entry point here falls back to when none is given (and
#: `plot()`'s own default). Named once so the "categories missing from a
#: `palette=` dict fall back to the default palette" rule in
#: `resolve_category_colors` cannot drift away from what an unspecified
#: `palette=` would have drawn.
DEFAULT_PALETTE = 'hls'


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
    palette : str, list of colors, dict, or matplotlib.colors.Colormap
        Base colors: a seaborn/matplotlib palette name, an explicit list of
        colors (hex strings like '#ff0000', named colors like 'red', or
        RGB(A) tuples), a matplotlib Colormap instance (sampled evenly), or
        -- for CATEGORICAL input only -- a ``{category: color}`` dict (see
        `resolve_category_colors`). Default: 'hls'. For CONTINUOUS (1D
        numeric) input, a list shorter than `n_bins` is blended into a
        smooth gradient using the listed colors as anchors (seaborn
        ``blend_palette`` semantics); for categorical/matrix input the list
        must supply at least one color per category/column.
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
        # `resolve_category_colors` is the ONE place a category order is
        # turned into colors, so a `palette=` dict, a list, a name and a
        # Colormap all mean the same thing here, in `plot()`'s legend, and
        # in either backend
        mapping = resolve_category_colors(palette, categories)
        return np.asarray([mapping[label] for label in labels], dtype=float)

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

#: sRGB relative-luminance weights (Rec. 709, as used by WCAG's contrast
#: definition): ``L = 0.2126 R + 0.7152 G + 0.0722 B`` over RGB in [0, 1].
#: Named once so `luminance` and `image_palette`'s `max_luminance=` /
#: `min_luminance=` bounds cannot disagree about what "too bright" means.
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)

#: Options `palette='image:<path>?<key>=<value>&...'` may carry, and the
#: type each is read as. They are exactly the tunable arguments of
#: `image_palette`, so the declarative string form can reach everything the
#: function call can.
_IMAGE_SPEC_OPTIONS = {'max_luminance': float, 'min_luminance': float,
                       'n_colors': int, 'resize': int, 'random_state': int}


def luminance(colors):
    """Relative luminance of one color, or of a sequence of colors.

    ``L = 0.2126 R + 0.7152 G + 0.0722 B`` (`LUMINANCE_WEIGHTS`) over sRGB
    components in [0, 1]: 0 is black, 1 is white, and the weights are the
    Rec. 709 luminance coefficients, so a saturated yellow scores far
    higher than an equally saturated blue -- which is the point, since what
    this measures is how legible a color is against a WHITE page.

    Accepts anything matplotlib accepts as a color (hex string, named
    color, RGB(A) tuple/array), or a sequence of them; an alpha channel is
    ignored.

    Returns
    -------
    float, or numpy.ndarray of shape (n,)
        A scalar for a single color, one value per entry for a sequence.
    """
    from matplotlib.colors import to_rgb

    weights = np.asarray(LUMINANCE_WEIGHTS, dtype=float)
    if _is_color(colors):
        return float(np.asarray(to_rgb(colors), dtype=float) @ weights)
    try:
        rgb = np.asarray([to_rgb(c) for c in colors], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "luminance() takes a matplotlib color (hex string, named "
            f"color, RGB(A) tuple) or a sequence of them; got {colors!r} "
            f"({exc})") from exc
    return np.asarray(rgb @ weights, dtype=float)


def _parse_image_spec(spec):
    """Split an image palette spec into ``(source, image_palette kwargs)``.

    ``'starry_night.jpg'`` -> ``('starry_night.jpg', {})``, and
    ``'starry_night.jpg?max_luminance=0.6&n_colors=8'`` ->
    ``('starry_night.jpg', {'max_luminance': 0.6, 'n_colors': 8})``.

    A REAL path always wins: if a file exists at the literal text, it is
    used verbatim, so a filename that happens to contain '?' (or '=', or
    '&') is never mangled. Only when no such file exists, and every
    ``key=value`` after the last '?' names one of `_IMAGE_SPEC_OPTIONS`, is
    the tail read as options; anything else is left alone so the caller
    still gets `image_palette`'s "could not find an image at ..." error
    naming the path they actually typed.
    """
    import os

    text = str(spec).strip()
    if '?' not in text or os.path.exists(os.path.expanduser(text)):
        return text, {}
    path, _, query = text.rpartition('?')
    options = {}
    for item in query.split('&'):
        key, sep, value = item.partition('=')
        key = key.strip()
        if not sep or key not in _IMAGE_SPEC_OPTIONS:
            return text, {}
        try:
            options[key] = _IMAGE_SPEC_OPTIONS[key](value.strip())
        except ValueError:
            raise ValueError(
                f"palette='{IMAGE_PALETTE_PREFIX}{text}': could not read "
                f"{key}={value.strip()!r} as a "
                f"{_IMAGE_SPEC_OPTIONS[key].__name__}") from None
    return path, options


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


def image_palette(image, n_colors=IMAGE_PALETTE_N, resize=200,
                  random_state=0, max_luminance=None,
                  min_luminance=None):
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
        Seed for the k-means fit, so repeated calls return the same palette
        (to within ~1e-15 per channel: k-means' threaded reductions sum in
        whatever order the threads finish, so the centers are reproducible
        but not bit-identical -- compare extracted colors with
        ``np.allclose``, never ``==``).
    max_luminance : float or None
        Drop extracted colors BRIGHTER than this (default None: keep all).
        Luminance is ``0.2126 R + 0.7152 G + 0.0722 B`` over RGB in [0, 1]
        (`luminance` / `LUMINANCE_WEIGHTS`). Filtering happens AFTER the
        salience ordering, so ``image_palette(path, max_luminance=0.6)[0]``
        is "the most salient color that is still legible on a white page":
        The Great Wave's two most salient clusters are its cream sky and
        foam (luminance 0.88 and 0.94), and this is what steps past them to
        its Prussian blue.
    min_luminance : float or None
        Drop extracted colors DARKER than this (default None: keep all) --
        the same filter from the other end, for artwork whose salient
        colors are near-black on a dark page.

    Returns
    -------
    palette : numpy.ndarray
        (k, 3) float RGB in [0, 1], k <= n_colors, ordered most salient
        first.

    Raises
    ------
    ValueError
        If a luminance bound excludes EVERY extracted color. The message
        reports the luminances actually measured, because the fix (widen
        the bound, raise `n_colors` so more clusters are extracted, or use
        a different image) depends on which of them it was.

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
    that takes a palette, luminance bounds included::

        hyp.plot(x, hue=values, palette='image:starry_night.jpg')
        hyp.plot(x, hue=values,
                 palette='image:great_wave.jpg?max_luminance=0.6')
    """
    from sklearn.cluster import KMeans

    if (not isinstance(n_colors, (int, np.integer))
            or isinstance(n_colors, bool) or n_colors < 1):
        raise ValueError(
            f"n_colors= must be a positive integer; got {n_colors!r}")
    lo, hi = _luminance_bounds(min_luminance, max_luminance)
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
    if lo > 0.0 or hi < 1.0:
        lums = luminance(np.asarray(out, dtype=float))
        kept = [c for c, value in zip(out, np.atleast_1d(lums))
                if lo <= value <= hi]
        if not kept:
            raise ValueError(
                f"image_palette(): none of the {len(out)} color(s) "
                f"extracted from this image have luminance in "
                f"[{lo:g}, {hi:g}] (measured: "
                f"{[round(float(v), 3) for v in np.atleast_1d(lums)]}). "
                "Widen the bound, raise n_colors= so more clusters are "
                "extracted, or use a different image.")
        out = kept
    return np.asarray(out, dtype=float)


def _luminance_bounds(min_luminance, max_luminance):
    """Validated ``(lo, hi)`` luminance window; `None` means "no bound"."""
    bounds = []
    for name, value, default in (('min_luminance', min_luminance, 0.0),
                                 ('max_luminance', max_luminance, 1.0)):
        if value is None:
            bounds.append(default)
            continue
        if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"{name}= must be a number in [0, 1] (relative luminance, "
                f"0.2126 R + 0.7152 G + 0.0722 B) or None; got {value!r}")
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name}= must be in [0, 1] (relative luminance); "
                f"got {value!r}")
        bounds.append(value)
    lo, hi = bounds
    if lo > hi:
        raise ValueError(
            f"min_luminance={lo!r} is greater than max_luminance={hi!r}, "
            "so no color could ever satisfy both")
    return lo, hi


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
    is the one case interpolation cannot serve, and it raises.

    The spec may carry `image_palette` options after a '?'
    (``'wave.jpg?max_luminance=0.6'``); an explicit ``n_colors`` there
    overrides the count chosen above (see `_parse_image_spec`). A LUMINANCE
    BOUND can only ever discard colors, so a bounded spec extracts at least
    `IMAGE_PALETTE_N` anchors however few are asked for, and the survivors
    are taken most-salient-first: extracting k=1 and then filtering it
    would leave a one-category plot with the image's average color or with
    nothing at all."""
    path, options = _parse_image_spec(source)
    wanted = IMAGE_PALETTE_N if continuous else max(n_colors, 1)
    if options.get('max_luminance') is not None or \
            options.get('min_luminance') is not None:
        wanted = max(wanted, IMAGE_PALETTE_N)
    options.setdefault('n_colors', wanted)
    colors = [tuple(c) for c in image_palette(path, **options)]
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
    if isinstance(palette, collections.abc.Mapping):
        raise ValueError(
            "palette= was given as a dict, which maps hue CATEGORIES to "
            "colors (palette={'Alice': '#E4572E', 'Bob': 'C0'}). Resolving "
            "it needs the category names, so it is supported for a "
            f"CATEGORICAL hue only; this call needs {n_colors} color(s) "
            "with no categories attached (a continuous or matrix hue, a "
            "colorbar, or one color per dataset). Pass a palette name, a "
            "list of colors, or a matplotlib Colormap here.")
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
        if _looks_like_dataset_palettes(palette):
            raise ValueError(
                f"palette={palette!r} is a list of PER-DATASET palettes "
                "(at least one entry is itself a palette, not a color). "
                "plot() resolves that form against the datasets you passed "
                "-- one entry per dataset -- so it cannot be used here, "
                f"where a single palette of {n_colors} color(s) is needed. "
                "Pass one palette name, list of colors, or Colormap.") \
                from exc
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


def _is_color(value):
    """True if matplotlib can read `value` as a single color."""
    from matplotlib.colors import to_rgb
    try:
        to_rgb(value)
    except (ValueError, TypeError):
        return False
    return True


def _names_a_palette(name):
    """True if seaborn/matplotlib know `name` as a palette or colormap."""
    import seaborn as sns
    try:
        sns.color_palette(name, 1)
    except (ValueError, TypeError, KeyError):
        return False
    return True


def _is_palette_spec(value):
    """True if `value` can stand alone as one dataset's `palette=`.

    That is: a color (a one-color palette), an ``'image:<path>'`` string, a
    seaborn/matplotlib palette NAME, a `Colormap`, a `{category: color}`
    dict, or a non-empty sequence of colors.
    """
    from matplotlib.colors import Colormap

    if isinstance(value, (Colormap, collections.abc.Mapping)):
        return True
    if isinstance(value, str):
        return (value.startswith(IMAGE_PALETTE_PREFIX) or _is_color(value)
                or _names_a_palette(value))
    if _is_color(value):
        return True
    if isinstance(value, (list, tuple, np.ndarray)):
        entries = list(value)
        return bool(entries) and all(_is_color(c) for c in entries)
    return False


def _looks_like_dataset_palettes(palette):
    """True if `palette` is a sequence with at least one non-color entry
    that IS a palette -- the shape `dataset_palettes` claims."""
    if isinstance(palette, (str, collections.abc.Mapping)):
        return False
    if not isinstance(palette, (list, tuple, np.ndarray)):
        return False
    entries = list(palette)
    return bool(entries) and any(
        not _is_color(e) and _is_palette_spec(e) for e in entries)


def resolve_category_colors(palette, categories,
                            default_palette=DEFAULT_PALETTE):
    """Ordered ``{category: (r, g, b)}`` for a CATEGORICAL hue.

    The single place a category order becomes colors, so the matplotlib
    renderer, the plotly renderer, the legend and the colorbar all read the
    same mapping instead of each re-deriving one (they used to, and a
    `palette=` dict would have had to be taught to three of them
    separately).

    Parameters
    ----------
    palette : str, list of colors, dict, matplotlib.colors.Colormap
        Everything `mat2colors`/`plot()` document, plus a
        ``{category: color}`` DICT that names the color for each category
        explicitly::

            hyp.plot(x, hue=speakers, palette={'Alice': '#E4572E',
                                               'Bob': '#17BEBB'})

        With a dict the category ORDER stops mattering: a caller no longer
        has to compute first-appearance order by hand so a list palette
        lines up with the right speaker. Categories NOT named in the dict
        fall back to `default_palette` ('hls'), each keeping the color that
        palette would have given it at its own position in `categories` --
        so naming a subset shifts nothing else. Colors are not checked for
        collisions: an explicitly named color may coincide with a fallback
        one, and that is the caller's choice to make.
    categories : sequence
        The categories, in the order they are drawn. Duplicates are
        collapsed (first appearance kept), so a raw list of per-observation
        labels may be passed directly. Entries need only be hashable --
        strings, ints, bools, tuples and enums all work.
    default_palette : str, list of colors, or Colormap
        What categories missing from a `palette=` DICT are colored from
        (default `DEFAULT_PALETTE`). Ignored for every other `palette=`
        form.

    Returns
    -------
    dict
        ``{category: (r, g, b)}`` in `categories` order (a plain dict, so
        insertion order IS the drawn order). Empty in, empty out.

    Raises
    ------
    ValueError
        If the dict names a key that is not one of `categories` (the
        message lists the categories that were seen -- a misspelled or
        stale key is otherwise silently ignored, which is exactly the bug
        an explicit mapping is meant to remove), or if one of its values is
        not a color.
    """
    cats = list(dict.fromkeys(categories))
    if not cats:
        return {}
    if not isinstance(palette, collections.abc.Mapping):
        colors = get_palette_colors(palette, len(cats))
        return {c: tuple(float(v) for v in colors[i])
                for i, c in enumerate(cats)}

    from matplotlib.colors import to_rgb

    # match by the category itself; fall back to matching its str() so a
    # dict keyed {0: 'red'} still reaches an integer hue that plot() has
    # stringified for its legend (and vice versa)
    by_str = {}
    for key in palette:
        by_str.setdefault(str(key), key)
    used, resolved = set(), {}
    for cat in cats:
        key = None
        try:
            if cat in palette:
                key = cat
        except TypeError:            # unhashable category: str() match only
            pass
        if key is None:
            key = by_str.get(str(cat))
        if key is not None:
            used.add(key)
            try:
                resolved[cat] = to_rgb(palette[key])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"palette[{key!r}] = {palette[key]!r} is not a color "
                    "(hex string, named color, or RGB(A) tuple)") from exc
    unknown = [k for k in palette if k not in used]
    if unknown:
        raise ValueError(
            f"palette= names {len(unknown)} categor"
            f"{'y' if len(unknown) == 1 else 'ies'} that this plot does not "
            f"have: {unknown!r}. The categories seen were {cats!r}. Remove "
            "the extra key(s), or check for a typo -- categories you do not "
            f"name are colored from the default palette "
            f"({default_palette!r}).")
    fallback = get_palette_colors(default_palette, len(cats))
    return {c: resolved[c] if c in resolved
            else tuple(float(v) for v in fallback[i])
            for i, c in enumerate(cats)}


def dataset_palettes(palette, n_datasets):
    """Split a PER-DATASET `palette=` list into one palette spec per dataset.

    `palette=` has always meant one palette for the WHOLE plot. This adds a
    second reading -- one palette per dataset,
    ``palette=['image:wave.jpg', 'image:starry_night.jpg', 'viridis']`` --
    without touching the first, under one rule:

        A list/tuple whose every entry is a COLOR is a list of colors, as
        it has always been. A list with at least one entry that is NOT a
        color but IS a palette (an ``'image:<path>'`` string, a palette
        name, a `Colormap`, a `{category: color}` dict, or a nested list of
        colors) is a list of per-dataset palettes.

    Nothing that worked before changes meaning: ``['red', '#00ff00',
    (0, 0, 1)]`` stays one three-color palette, because every entry is a
    color. And an entry of a per-dataset list may still be a single color
    ('red' = a one-color palette for that dataset), because the rule looks
    at the list as a whole, not at each entry.

    Parameters
    ----------
    palette : any `palette=` value
        Only a list/tuple/ndarray can be per-dataset; a string, dict,
        `Colormap` or None never is.
    n_datasets : int
        How many datasets are being drawn.

    Returns
    -------
    list or None
        `n_datasets` palette specs -- each usable anywhere `palette=` is --
        or None when `palette` is a single whole-plot palette, which tells
        the caller to keep doing exactly what it does today. A ONE-entry
        per-dataset list is broadcast to every dataset.

    Raises
    ------
    ValueError
        If the list is per-dataset but has neither 1 nor `n_datasets`
        entries, or if an entry is neither a color nor a resolvable palette
        (a typo'd color name reads as a palette name; the message says so).
    """
    if (palette is None or isinstance(palette, (str, collections.abc.Mapping))
            or not isinstance(palette, (list, tuple, np.ndarray))):
        return None
    entries = list(palette)
    if not entries or all(_is_color(e) for e in entries):
        # empty (its own error, downstream) or the historical color list
        return None
    if (not isinstance(n_datasets, (int, np.integer))
            or isinstance(n_datasets, bool) or n_datasets < 1):
        raise ValueError(
            f"n_datasets= must be a positive integer; got {n_datasets!r}")
    bad = [e for e in entries if not _is_palette_spec(e)]
    if bad:
        raise ValueError(
            f"palette={palette!r} has entries that are neither a color nor "
            f"a palette: {bad!r}. Either every entry must be a color (one "
            "palette of explicit colors for the whole plot) or every entry "
            "must be a palette -- a palette name, 'image:<path>', a "
            "Colormap, a {category: color} dict, or a list of colors -- one "
            "per dataset.")
    if len(entries) == 1:
        return entries * n_datasets
    if len(entries) != n_datasets:
        raise ValueError(
            f"palette= lists {len(entries)} per-dataset palettes but "
            f"{n_datasets} dataset(s) were passed. Give one palette per "
            "dataset (or one to use for all of them). If you meant a "
            "single palette of explicit colors, every entry has to be a "
            "color -- at least one of these is not.")
    return entries


def palette_lead_color(spec):
    """The one color that REPRESENTS a palette: its lead color.

    For ``'image:<path>'`` that is the most visually salient color of the
    image -- `image_palette`'s first entry, from its full default six
    anchors, with any ``?max_luminance=``/``?min_luminance=`` bound in the
    spec applied first. (Asking `get_palette_colors` for ONE color from an
    image instead runs k-means with k=1, which returns the image's AVERAGE
    color: muddy, and never the vivid one salience ordering exists to
    find.) For every other palette form it is the first color the palette
    yields.

    A bare color string ('red') is a one-color palette. A string that names
    BOTH a palette and a color ('gray', 'grey', 'pink') is read as the
    PALETTE, because that is what `palette=` has always meant everywhere
    else, and one spelling of a color must not mean two different things in
    two places.

    Used to give each dataset a single color when `palette=` is a
    per-dataset list (see `dataset_palettes`).
    """
    from matplotlib.colors import to_rgb

    if isinstance(spec, str):
        if spec.startswith(IMAGE_PALETTE_PREFIX):
            source, options = _parse_image_spec(
                spec[len(IMAGE_PALETTE_PREFIX):])
            return tuple(float(v) for v in image_palette(source, **options)[0])
        if not _names_a_palette(spec) and _is_color(spec):
            return tuple(float(v) for v in to_rgb(spec))
    return tuple(float(v) for v in get_palette_colors(spec, 1)[0])


def dataset_colors(palette, n_datasets):
    """One representative RGB per dataset, per-dataset palettes honored.

    ``(n_datasets, 3)`` float array. With a per-dataset `palette=` list
    (see `dataset_palettes`) each dataset gets its own palette's
    `palette_lead_color`; otherwise this is exactly
    ``get_palette_colors(palette, n_datasets)``, i.e. today's colors.

    Note for callers that currently hand `palette` to seaborn directly:
    seaborn CYCLES a color list that is shorter than `n_datasets`, while
    `get_palette_colors` raises. Where that difference matters, call
    `dataset_palettes` and fall back to the existing seaborn call when it
    returns None.
    """
    specs = dataset_palettes(palette, n_datasets)
    if specs is None:
        return get_palette_colors(palette, n_datasets)
    return np.asarray([palette_lead_color(s) for s in specs], dtype=float)


# Legacy continuous-color helpers live in _shared.helpers (import *-ed widely);
# re-export them here so plot.colors is the single coloring surface.
from .._shared.helpers import vals2colors, vals2bins  # noqa: F401,E402
