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
    "DENSITY_BOOST_GAMMA",
    "DENSITY_BOOST_MAX",
    "MAX_VOLUME_OPACITY",
    "normalize_density_arg",
    "broadcast_density",
    "resolve_grid",
    "resolve_iso_fracs_alphas",
    "fit_kde",
    "kde_grid_2d",
    "kde_grid_3d",
    "alpha_colormap",
    "iso_surfaces_3d",
    "bbox_extent",
    "density_alpha_boost",
    "resolve_plotly_volume_params",
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

# Auto-boost constants (GH #108 round 2 -- "invisible when separated"): both
# 3-D renderers (matplotlib iso-surfaces, plotly Volume) tune their BASE
# alpha/opacity for a dataset that fills the whole shared plotting cube. When
# datasets are jointly scaled into that one cube but sit far apart (each
# occupying only a small fraction of it), that same base alpha covers only a
# tiny screen area and reads as invisible even though the geometry is drawn
# correctly. `density_alpha_boost` computes a multiplier -- ~1 (no-op) for a
# scene-filling dataset, ramping up to `DENSITY_BOOST_MAX` for a dataset
# that's small relative to the scene -- that both backends apply on TOP of
# the user's own `alpha=` (which still scales everything linearly, exactly
# as before). `gamma=2` (quadratic in the extent ratio -- visibility of an
# alpha-composited layer tracks the SCREEN AREA it covers, which scales
# with the square of its linear extent, not the extent itself) and
# `max_boost=6` were chosen by rendering two-blob scenes at separations
# 0/5/10 (in units of the blobs' own std) and iterating until the density
# was clearly visible but still subtle at every separation, with the data
# points always the dominant visual element -- see `tests/test_density.py`
# and
# `docs/images/v1.0-seven-features/density_3d_{mpl,plotly}.png`.
DENSITY_BOOST_GAMMA = 2.0
DENSITY_BOOST_MAX = 6.0
# Absolute ceiling for plotly's boosted `go.Volume` opacity: high enough
# that a maximally-boosted (small-in-scene) cluster is unmistakably visible,
# but held well below 1.0 so the volume never reads as a fully opaque blob
# -- the underlying data markers must stay the dominant visual element.
# Lowered from the original 0.95 (R2 fix, maintainer request: plotly's
# volumetric shading read as noticeably heavier/denser than matplotlib's
# airy iso-shells -- see `docs/images/v1.0-seven-features/density_3d_{mpl,
# plotly}.png`) -- retuned by rendering the standard 2-blob scene AND the
# separated (`sep=10`, auto-boost-engaged) scene side by side with the
# matplotlib version and iterating until the glow reads as subtle in both,
# while staying clearly visible in the boosted/separated case.
MAX_VOLUME_OPACITY = 0.75

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
    _validate_density_values(merged)
    return merged


def _validate_density_values(spec):
    """Validate the (already-defaulted) values in a merged density spec
    dict, raising a clear ``ValueError`` for anything out of range. Runs on
    every call (including the all-defaults case) so a bad default would be
    caught too, though the shipped defaults are always valid."""
    alpha = spec["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) \
            or not (0 < alpha <= 1):
        raise ValueError(
            f"density['alpha'] must be a real number in (0, 1]; got {alpha!r}"
        )

    grid = spec["grid"]
    if grid is not None:
        if isinstance(grid, bool) or not isinstance(grid, (int, np.integer)) \
                or grid < 8:
            raise ValueError(
                f"density['grid'] must be an int >= 8 (or None for "
                f"auto-resolution); got {grid!r}"
            )

    levels = spec["levels"]
    if isinstance(levels, bool) or not isinstance(levels, (int, np.integer)) \
            or not (1 <= levels <= 10):
        raise ValueError(
            f"density['levels'] must be an int in [1, 10]; got {levels!r}"
        )

    per_group = spec["per_group"]
    if not isinstance(per_group, (bool, np.bool_)):
        raise ValueError(
            f"density['per_group'] must be a bool; got {per_group!r}"
        )


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

    Notes
    -----
    numpy bools (``np.True_``/``np.False_``) are accepted anywhere a
    Python bool is, matching ``per_group``'s tolerance (release-1.0
    audit, F07-008).
    """
    if isinstance(density, np.bool_):
        density = bool(density)
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
    # explicit degeneracy check (release-1.0 audit, D11-009): relying on
    # gaussian_kde to raise was VALUE-dependent -- for some duplicate-point
    # inputs the covariance is only NUMERICALLY (not exactly) singular and
    # gaussian_kde "succeeds" silently. Rank-check the centered points so
    # duplicate/collinear/coplanar data is always detected, whatever its
    # values; the try/except below stays as a backstop.
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered) < points.shape[1]:
        warnings.warn(
            f"density: dataset{dataset_label} is degenerate (the points "
            f"span fewer than {points.shape[1]} dimensions -- e.g. "
            "duplicate, collinear, or coplanar points); skipping its "
            "density."
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


def bbox_extent(points):
    """Scalar "size" of a point cloud's axis-aligned bounding box: the
    Euclidean norm of its per-axis span (``hi - lo``). Used by
    :func:`density_alpha_boost` to compare one dataset's own extent against
    the overall scene's extent."""
    points = np.asarray(points, dtype=float)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return float(np.linalg.norm(hi - lo))


def density_alpha_boost(dataset_extent, scene_extent,
                        gamma=DENSITY_BOOST_GAMMA, max_boost=DENSITY_BOOST_MAX):
    """Multiplicative boost for a dataset's density alpha/opacity, based on
    how small `dataset_extent` (that dataset's own bounding-box size, e.g.
    from :func:`bbox_extent`) is relative to `scene_extent` (the whole
    plotted scene's bounding-box size).

    ``boost = clamp((scene_extent / dataset_extent) ** gamma, 1, max_boost)``

    A dataset that fills the whole scene (``dataset_extent ~= scene_extent``,
    e.g. a single dataset plotted alone) gets ``boost ~= 1`` -- a no-op, so
    today's tuned base alphas are unchanged. A dataset that's small relative
    to the scene (e.g. one of several widely-separated clusters, jointly
    scaled into the same shared cube) gets boosted up to `max_boost`-fold,
    so its density shading stays visible instead of vanishing into a
    handful of near-transparent pixels/voxels.

    ``effective_alpha = base_alpha * density_alpha_boost(...)`` -- the
    boost multiplies ON TOP of the user's own explicit ``alpha=``; it does
    not replace or override it.

    Returns ``1.0`` (no boost) if either extent is non-positive (e.g. a
    degenerate single-point dataset), rather than dividing by zero.
    """
    if dataset_extent <= 0 or scene_extent <= 0:
        return 1.0
    boost = (scene_extent / dataset_extent) ** gamma
    return float(np.clip(boost, 1.0, max_boost))


def resolve_plotly_volume_params(alpha, levels, boost, max_boost=DENSITY_BOOST_MAX):
    """Resolve boost-scaled ``go.Volume`` rendering parameters (GH #108
    round 2, plotly-only): the KDE grid's padding, `isomin`, `opacityscale`,
    `opacity`, and `surface_count`.

    Boosting `opacity` and `surface_count` alone (see
    :func:`density_alpha_boost`) is NOT enough to fix plotly's invisibility
    problem: hypertools' plotly scatter markers are large, fully-opaque,
    same-colored disks that -- for a dataset small relative to the scene --
    cover almost its entire on-screen footprint, hiding any density
    volume drawn underneath. The only part of the volume that can still
    show is a thin glow peeking out past the markers' edges, and that glow
    is governed by how far out the KDE grid is evaluated (`pad`) and how
    much opacity low density VALUES get (`isomin`, `opacityscale`) -- not
    by the trace's overall `opacity`. This was verified empirically:
    rendering an isolated small, separated cluster showed the glow stayed
    invisible even at `opacity` near :data:`MAX_VOLUME_OPACITY` until `pad`
    and the opacity ramp were also widened.

    All five returned values equal their ORIGINAL, pre-boost constants
    when `boost == 1` (a scene-filling dataset) -- `pad=0.15`,
    `isomin=0.05`, `opacityscale=[[0, 0], [0.3, 0.27], [1, 0.53]]`,
    `opacity=min(2*alpha, 0.4)`, `surface_count=5*levels` -- so nothing
    changes for that (already correctly-tuned) case. As `boost` ramps up
    to `max_boost`, `pad` widens (0.15 -> 0.5), `isomin` and the
    `opacityscale` breakpoint shift left (exposing more of the KDE's outer
    tail) and up (giving that tail more opacity), and `opacity`/
    `surface_count` scale as in :func:`density_alpha_boost`'s docstring.

    The `opacity`/`opacityscale` constants (R2 fix, maintainer request:
    plotly's volumetric shading read noticeably heavier/denser than
    matplotlib's airy iso-shells) were lowered from the original
    `opacity=min(3*alpha, 0.6)` / `opacityscale=[[0, 0], [0.3, 0.4],
    [1, 0.8]]` -- retuned by rendering the standard 2-blob 3-D density
    scene AND the separated (`sep=10`, auto-boost-engaged) scene side by
    side with the matplotlib version and iterating until the plotly glow
    read as subtle/airy in both while remaining clearly visible in the
    boosted case (see
    `docs/images/v1.0-seven-features/density_3d_plotly.png`).

    Returns
    -------
    (pad, isomin, opacityscale, opacity, surface_count)
    """
    t = 0.0 if max_boost <= 1 else float(np.clip(
        (boost - 1) / (max_boost - 1), 0.0, 1.0))
    pad = 0.15 + 0.35 * t
    isomin = 0.05 - 0.04 * t
    mid_x = 0.3 - 0.24 * t
    mid_y = 0.27 + 0.10 * t
    top_y = 0.53 + 0.07 * t
    opacityscale = [[0, 0], [mid_x, mid_y], [1, top_y]]
    base_opacity = min(2.0 * alpha, 0.4)
    opacity = min(base_opacity * boost, MAX_VOLUME_OPACITY)
    surface_count = int(round(5 * levels * boost))
    return pad, isomin, opacityscale, opacity, surface_count


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


def resolve_iso_fracs_alphas(levels):
    """Resolve the `levels` knob (matplotlib 3-D only) to a matched pair of
    iso-surface density-fraction thresholds and base alphas, both length
    `levels`.

    ``levels=3`` (the default) reproduces the original hand-tuned constants
    EXACTLY -- ``fracs=(0.10, 0.35, 0.65)``, ``alphas=(0.03, 0.05, 0.07)`` --
    since ``np.linspace(0.10, 0.65, 3)`` gives ``(0.10, 0.375, 0.65)``, not
    the tuned middle shell at 0.35. Any other `levels` value spaces `levels`
    fracs evenly across the same ``[0.10, 0.65]`` density-fraction range via
    `np.linspace`, with alphas ramping linearly from 0.03 to 0.07 to match.
    """
    if levels == 3:
        return (0.10, 0.35, 0.65), (0.03, 0.05, 0.07)
    fracs = tuple(np.linspace(0.10, 0.65, levels))
    alphas = tuple(np.linspace(0.03, 0.07, levels))
    return fracs, alphas


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
