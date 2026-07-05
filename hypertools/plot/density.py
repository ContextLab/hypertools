#!/usr/bin/env python
"""Shared KDE-density-shading helpers for the ``density=`` plot() kwarg
(GH #108, #191).

This module is backend-agnostic: it validates/normalizes the ``density=``
argument, fits/evaluates the ``scipy.stats.gaussian_kde`` grid shared by both
renderers, and (for matplotlib's 3-D path) wraps the optional
``skimage.measure.marching_cubes`` iso-surface extraction. Both
``matplotlib_backend.py`` and ``plotly_backend.py`` import from here so the
two renderers share one source of truth for defaults, validation, and
degenerate-input handling -- mirroring how ``surface.py`` backs the
``surface=`` kwarg (GH #109).

Unlike ``surface=``, `density=` has no per-dataset list form and no `color`
override key: every dataset's density layer always inherits that dataset's
own drawn color (resolved the same way as `surface_colors`), and the single
validated spec applies to every dataset (or, with ``per_group=False``, to
one pooled layer over all of them).
"""
import warnings

import numpy as np
from scipy.stats import gaussian_kde

__all__ = [
    "VALID_DENSITY_KEYS",
    "DENSITY_DEFAULTS",
    "POOLED_COLOR",
    "HAS_SKIMAGE",
    "normalize_density_arg",
    "broadcast_density",
    "resolve_grid",
    "fit_kde",
    "kde_grid_2d",
    "kde_grid_3d",
    "alpha_colormap",
    "iso_surfaces_3d",
]

VALID_DENSITY_KEYS = {"alpha", "levels", "grid", "per_group"}
DENSITY_DEFAULTS = {
    "alpha": 0.2,
    "levels": 3,
    # `grid=None` means "auto": resolved by `resolve_grid` to 200 for 2-D
    # data or 50 for 3-D data (the brief's 200/50 defaults), since the right
    # resolution genuinely differs by dimensionality and this dict has no
    # way to know `ndims` at validation time.
    "grid": None,
    "per_group": True,
}
# Neutral color used for a single pooled density layer (`per_group=False`)
# that doesn't belong to any one dataset.
POOLED_COLOR = (0.3, 0.3, 0.3)

try:
    from skimage import measure as _skimage_measure
    HAS_SKIMAGE = True
except ImportError:
    _skimage_measure = None
    HAS_SKIMAGE = False


def _validate_density_dict(d):
    """Validate one density spec dict and fill in defaults for missing keys."""
    if not isinstance(d, dict):
        raise ValueError(f"density must be bool, dict, or None; got {d!r}")
    unknown = set(d) - VALID_DENSITY_KEYS
    if unknown:
        raise ValueError(
            f"density dict got unknown key(s) {sorted(unknown)}; valid keys "
            f"are {sorted(VALID_DENSITY_KEYS)}."
        )
    merged = dict(DENSITY_DEFAULTS)
    merged.update(d)
    return merged


def normalize_density_arg(density):
    """Validate ``density=`` eagerly (before the analyze/reduce pipeline
    runs) and return a normalized form.

    Returns
    -------
    None
        `density` was ``None``/``False`` (density shading disabled).
    dict
        `density` was ``True`` or a dict -- applied to every dataset (see
        :func:`broadcast_density`), or pooled into a single layer if the
        resolved ``per_group`` key is ``False``.

    Raises
    ------
    ValueError
        `density` is a dict with an unknown key, or is not one of the
        accepted forms (``None``, a bool, or a dict).
    """
    if density is None or density is False:
        return None
    if density is True:
        return _validate_density_dict({})
    if isinstance(density, dict):
        return _validate_density_dict(density)
    raise ValueError(
        f"density must be None, a bool, or a dict; got {density!r}"
    )


def broadcast_density(normalized, n):
    """Broadcast a :func:`normalize_density_arg`-normalized value to exactly
    `n` per-dataset entries (the same validated dict repeated `n` times, or
    `n` ``None`` entries if density shading is disabled)."""
    if normalized is None:
        return [None] * n
    return [normalized] * n


def resolve_grid(spec, ndims):
    """Resolve `spec['grid']` to a concrete grid resolution: the user's
    explicit value if given, else 200 for 2-D data or 50 for 3-D data."""
    grid = spec.get("grid")
    if grid is not None:
        return int(grid)
    return 200 if ndims == 2 else 50


def fit_kde(points, dataset_label=""):
    """Fit a ``scipy.stats.gaussian_kde`` on `points` (an `(n, d)` array).

    Returns the fitted KDE, or ``None`` (with a ``UserWarning``) if there are
    fewer than 3 points or the data is degenerate (singular covariance --
    e.g. exactly collinear/coplanar/duplicate points) -- callers should skip
    that dataset's density layer rather than crash.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        warnings.warn(
            f"density: dataset{dataset_label} has fewer than 3 points "
            "(need >= 3 to fit a KDE); skipping its density."
        )
        return None
    try:
        return gaussian_kde(points.T)
    except (np.linalg.LinAlgError, ValueError) as exc:
        warnings.warn(
            f"density: dataset{dataset_label} is degenerate ({exc}); "
            "skipping its density."
        )
        return None


def _padded_bounds(points, pad):
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = hi - lo
    span = np.where(span == 0, 1.0, span)
    return lo - pad * span, hi + pad * span


def kde_grid_2d(points, kde, gridsize=200, pad=0.15):
    """Evaluate `kde` on a `gridsize` x `gridsize` grid over `points`' bounds
    (padded by `pad` on each side). Returns ``(xs, ys, Z, extent)`` where
    ``Z[iy, ix]`` is the density at ``(xs[ix], ys[iy])`` (matplotlib
    ``imshow(origin='lower')`` layout) and ``extent`` is
    ``(xmin, xmax, ymin, ymax)``."""
    points = np.asarray(points, dtype=float)
    lo, hi = _padded_bounds(points, pad)
    xs = np.linspace(lo[0], hi[0], gridsize)
    ys = np.linspace(lo[1], hi[1], gridsize)
    X, Y = np.meshgrid(xs, ys)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    extent = (xs[0], xs[-1], ys[0], ys[-1])
    return xs, ys, Z, extent


def kde_grid_3d(points, kde, gridsize=50, pad=0.15):
    """Evaluate `kde` on a `gridsize`^3 grid over `points`' bounds (padded by
    `pad` on each side, ``indexing='ij'``). Returns ``(X, Y, Z, D, lo,
    spacing)``: `X`/`Y`/`Z` are the coordinate grids, `D` the density values
    (same shape), `lo` the grid's lower corner, and `spacing` the per-axis
    grid step (both needed by `skimage.measure.marching_cubes`)."""
    points = np.asarray(points, dtype=float)
    lo, hi = _padded_bounds(points, pad)
    axes_ = [np.linspace(lo[i], hi[i], gridsize) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes_, indexing="ij")
    D = kde(np.vstack([X.ravel(), Y.ravel(), Z.ravel()])).reshape(X.shape)
    spacing = [(hi[i] - lo[i]) / (gridsize - 1) for i in range(3)]
    return X, Y, Z, D, lo, spacing


def alpha_colormap(color, max_alpha, name="hypertools_density"):
    """A ``LinearSegmentedColormap`` ramping from fully transparent to
    `color` at `max_alpha` opacity -- used for the 2-D matplotlib ``imshow``
    density layer (an alpha ramp reads as "subtle glow", unlike
    ``contourf``'s hard per-level boundaries)."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    r, g, b = to_rgb(color)
    return LinearSegmentedColormap.from_list(
        name, [(r, g, b, 0.0), (r, g, b, max_alpha)]
    )


def iso_surfaces_3d(D, lo, spacing, fracs=(0.10, 0.35, 0.65)):
    """Extract one iso-surface mesh per `fracs` entry (each a fraction of
    `D.max()`) via ``skimage.measure.marching_cubes``. Returns a list of
    ``(verts, faces)`` tuples (shorter than `fracs` if a level is degenerate
    at the grid's actual value range). Requires :data:`HAS_SKIMAGE`; callers
    must check that themselves (and fall back to the scatter-fog
    alternative) since raising here would defeat that fallback."""
    if not HAS_SKIMAGE:
        raise RuntimeError(
            "scikit-image is required for 3-D iso-surface density "
            "rendering; call sites should check HAS_SKIMAGE first."
        )
    dmax = D.max()
    meshes = []
    if dmax <= 0:
        return meshes
    for frac in fracs:
        level = frac * dmax
        if level <= D.min() or level >= dmax:
            continue
        try:
            verts, faces, _, _ = _skimage_measure.marching_cubes(
                D, level=level, spacing=spacing)
        except (ValueError, RuntimeError):
            continue
        verts = verts + lo
        meshes.append((verts, faces))
    return meshes
