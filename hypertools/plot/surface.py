#!/usr/bin/env python
"""Shared surface-rendering helpers for the ``surface=`` plot() kwarg (GH #109).

This module is backend-agnostic: it validates/normalizes the ``surface=``
argument, builds the smooth-hull mesh (3-D) or outline (2-D) for a single
dataset (delegating the geometry itself to :mod:`hypertools.plot.meshutil`),
and resolves per-backend lighting keyword arguments. Both
``matplotlib_backend.py`` and ``plotly_backend.py`` import from here so the
two renderers share one source of truth for defaults, validation, and
degenerate-input handling.
"""
import warnings

import numpy as np

from .meshutil import smooth_hull_2d, smooth_hull_3d

__all__ = [
    "VALID_SURFACE_KEYS",
    "SURFACE_DEFAULTS",
    "normalize_surface_arg",
    "broadcast_surface",
    "mpl_lighting_kwargs",
    "build_mesh_3d",
    "build_outline_2d",
    "view_vector",
    "surface_cube_scale",
    "PLOTLY_LIGHTPOSITION",
    "PLOTLY_IDENTITY_LIGHTING",
]

VALID_SURFACE_KEYS = {
    "alpha", "color", "lighting", "smoothing", "pre_inflate", "keep_points",
}
SURFACE_DEFAULTS = {
    "alpha": 0.6,
    "color": None,
    "lighting": {},
    "smoothing": 3,
    "pre_inflate": 1.0,
    "keep_points": True,
}
# Upper bound for `smoothing`: the mesh face count scales as
# ``4 ** smoothing`` relative to the raw hull, so smoothing=6 already means
# ~4096x the hull's face count -- values beyond that are a memory/time
# footgun with no visible smoothness gain (release-1.0 audit, F07-006).
MAX_SMOOTHING = 6

# Task R3 (2026-07-06, maintainer request: "confirm we can control
# coloring, lighting, and shading via parameters passed to plot"): both
# backends shade via the SAME two-light Blinn-Phong model
# (`hypertools.plot.meshutil.blinn_phong_colors`/`blinn_phong_vertex_colors`)
# -- matplotlib per-face, plotly per-vertex (precomputed and handed to
# `go.Mesh3d` as `vertexcolor`, with plotly's OWN lighting engine forced to
# the identity; see `PLOTLY_IDENTITY_LIGHTING` below) -- so `_LIGHTING_KEYS`
# is the single, shared set of keys both `mpl_lighting_kwargs` callers (in
# both `matplotlib_backend.py` and `plotly_backend.py`) resolve overrides
# against; there is no longer a separate plotly-only key set. `roughness`
# and `fresnel` (previously accepted, silently ignored) were removed here:
# they only ever applied to plotly's own per-face lighting engine, which
# GH #109 round 3 (commit fd064aa5) replaced with the precomputed
# vertex-shading above -- accepting keys that provably had zero visual
# effect on either backend violates this exact maintainer request, so they
# now raise the same "unknown lighting key" `ValueError` as any other typo
# instead of being silently swallowed. `lightdir` is new (Task R3): the
# Blinn-Phong machinery already supported an explicit key-light direction
# (`blinn_phong_colors`/`blinn_phong_vertex_colors`'s `lightdir` parameter)
# but no `surface['lighting']` key ever exposed it to callers.
_LIGHTING_KEYS = ("ambient", "diffuse", "fill", "specular", "shininess", "lightdir")
_MPL_LIGHTING_KEYS = _LIGHTING_KEYS  # backwards-compat alias
_MPL_LIGHTING_DEFAULTS = dict(
    ambient=0.45, diffuse=0.55, fill=0.25, specular=0.30, shininess=48,
    lightdir=None,
)
PLOTLY_LIGHTPOSITION = dict(x=2.5, y=-1.5, z=3.0)

# GH #109 round 3: plotly's own go.Mesh3d lighting engine computes shading
# per-FACE from each triangle's own (possibly reversed, for the doubled-
# winding self-culling workaround -- see plotly_backend._mesh3d_trace)
# normal, which renders the reversed winding's copy dark wherever the
# original copy faces the light -- large jagged dark patches on both
# meshes. Fixed by precomputing per-VERTEX Blinn-Phong colors ourselves
# (matching the matplotlib renderer's own lighting model exactly -- see
# `mpl_lighting_kwargs`/`hypertools.plot.meshutil.blinn_phong_vertex_colors`)
# and handing them to Mesh3d as `vertexcolor`, then telling Mesh3d's OWN
# lighting engine to leave those colors alone: identity lighting (ambient=1,
# diffuse=specular=fresnel=0) reproduces the vertex colors verbatim instead
# of re-shading them. Both windings of a doubled face share the same three
# vertex indices, so they always get identical (already-correct) colors.
PLOTLY_IDENTITY_LIGHTING = dict(
    ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0
)


def _validate_surface_dict(d):
    """Validate one surface spec dict and fill in defaults for missing keys."""
    if not isinstance(d, dict):
        raise ValueError(f"surface list items must be bool or dict; got {d!r}")
    unknown = set(d) - VALID_SURFACE_KEYS
    if unknown:
        raise ValueError(
            f"surface dict got unknown key(s) {sorted(unknown)}; valid keys "
            f"are {sorted(VALID_SURFACE_KEYS)}."
        )
    lighting = d.get("lighting", {})
    if not isinstance(lighting, dict):
        raise ValueError(
            f"surface['lighting'] must be a dict; got {lighting!r}"
        )
    allowed_lighting = set(_LIGHTING_KEYS)
    unknown_lighting = set(lighting) - allowed_lighting
    if unknown_lighting:
        raise ValueError(
            f"surface['lighting'] got unknown key(s) {sorted(unknown_lighting)}; "
            f"valid keys are {sorted(allowed_lighting)}."
        )
    if "lightdir" in lighting and lighting["lightdir"] is not None:
        lightdir = np.asarray(lighting["lightdir"], dtype=float)
        if lightdir.shape != (3,) or not np.all(np.isfinite(lightdir)):
            raise ValueError(
                "surface['lighting']['lightdir'] must be a finite 3-vector "
                f"(x, y, z) or None; got {lighting['lightdir']!r}"
            )
        if np.allclose(lightdir, 0.0):
            raise ValueError(
                "surface['lighting']['lightdir'] must not be the zero "
                "vector (no well-defined direction)"
            )
    merged = dict(SURFACE_DEFAULTS)
    merged.update(d)
    merged["lighting"] = lighting
    _validate_surface_values(merged)
    return merged


def _validate_surface_values(spec):
    """Validate the (already-defaulted) values in a merged surface spec
    dict, raising a clear ``ValueError`` naming the key, the constraint,
    and the received value for anything out of range -- mirroring
    ``density``'s ``_validate_density_values`` exactly (release-1.0 audit,
    F07-006: previously a bad ``alpha`` only crashed LATE, deep inside
    matplotlib, ``pre_inflate=0`` leaked a raw scipy ``QhullError``, and
    ``pre_inflate<0`` silently built an inside-out mesh). Runs on every
    call (including the all-defaults case) so a bad default would be
    caught too. ``color`` is deliberately not validated here: any
    matplotlib color spec is legal, and matplotlib's own conversion error
    already names the bad value clearly."""
    alpha = spec["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) \
            or not (0 < alpha <= 1):
        raise ValueError(
            f"surface['alpha'] must be a real number in (0, 1]; got {alpha!r}"
        )

    smoothing = spec["smoothing"]
    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, np.integer)) \
            or not (0 <= smoothing <= MAX_SMOOTHING):
        raise ValueError(
            f"surface['smoothing'] must be an int in [0, {MAX_SMOOTHING}] "
            f"(the mesh face count grows as 4**smoothing); got {smoothing!r}"
        )

    pre_inflate = spec["pre_inflate"]
    if isinstance(pre_inflate, bool) \
            or not isinstance(pre_inflate, (int, float, np.integer, np.floating)) \
            or not np.isfinite(pre_inflate) or pre_inflate <= 0:
        raise ValueError(
            f"surface['pre_inflate'] must be a positive, finite real number "
            f"(a scale factor > 0); got {pre_inflate!r}"
        )

    keep_points = spec["keep_points"]
    if not isinstance(keep_points, (bool, np.bool_)):
        raise ValueError(
            f"surface['keep_points'] must be a bool; got {keep_points!r}"
        )


def normalize_surface_arg(surface):
    """Validate ``surface=`` eagerly (before the analyze/reduce pipeline
    runs) and return a normalized form ready for :func:`broadcast_surface`
    once the final (post cluster/hue-reshape) dataset count is known.

    Returns
    -------
    None
        `surface` was ``None``/``False`` (surfaces disabled).
    dict
        `surface` was ``True`` or a single dict -- broadcast to every
        dataset.
    list of (dict or None)
        `surface` was a list of bool/dict -- one entry per dataset (length
        checked in :func:`broadcast_surface`, once the dataset count is
        finalized).

    Raises
    ------
    ValueError
        `surface` (or any dict/lighting-dict inside it) has an unknown key
        or an out-of-range value (see :func:`_validate_surface_values`),
        or `surface` is not one of the accepted forms.

    Notes
    -----
    numpy bools (``np.True_``/``np.False_``) are accepted anywhere a
    Python bool is -- both at the top level and as list entries --
    matching the tolerance of the dict values (release-1.0 audit,
    F07-008).
    """
    if isinstance(surface, np.bool_):
        surface = bool(surface)
    if surface is None or surface is False:
        return None
    if surface is True:
        return _validate_surface_dict({})
    if isinstance(surface, dict):
        return _validate_surface_dict(surface)
    if isinstance(surface, (list, tuple)):
        out = []
        for item in surface:
            if isinstance(item, np.bool_):
                item = bool(item)
            if item is None or item is False:
                out.append(None)
            elif item is True:
                out.append(_validate_surface_dict({}))
            elif isinstance(item, dict):
                out.append(_validate_surface_dict(item))
            else:
                raise ValueError(
                    f"surface list items must be bool or dict; got {item!r}"
                )
        return list(out)
    raise ValueError(
        "surface must be None, a bool, a dict, or a list of bool/dict; got "
        f"{surface!r}"
    )


def broadcast_surface(normalized, n):
    """Broadcast a :func:`normalize_surface_arg`-normalized value to exactly
    `n` per-dataset entries (each a validated dict, or None to disable that
    dataset's surface)."""
    if normalized is None:
        return [None] * n
    if isinstance(normalized, dict):
        return [normalized] * n
    if len(normalized) != n:
        raise ValueError(
            f"surface list has {len(normalized)} entries but there are "
            f"{n} datasets to plot; pass a single bool/dict to apply it to "
            "every dataset, or a list matching the dataset count."
        )
    return normalized


def mpl_lighting_kwargs(spec):
    """Resolve `spec['lighting']` overrides onto the shared two-light
    Blinn-Phong lighting defaults (Task R3: both matplotlib and plotly
    resolve their shading through this same function -- see
    `hypertools.plot.meshutil.blinn_phong_colors`/`blinn_phong_vertex_colors`
    and `_LIGHTING_KEYS` above -- so both backends honor exactly the same
    ``surface['lighting']`` keys)."""
    kw = dict(_MPL_LIGHTING_DEFAULTS)
    kw.update({k: v for k, v in spec.get("lighting", {}).items()
               if k in _LIGHTING_KEYS})
    return kw


def build_mesh_3d(points, spec, dataset_label="", quiet=False):
    """Build a smooth hull mesh for one dataset's 3-D points, honoring
    `spec['smoothing']` (rounds) / `spec['pre_inflate']`.

    Returns ``(verts, faces)``, or ``None`` (optionally warning) if there
    are fewer than 4 points or they are degenerate/coplanar -- callers
    should skip that dataset's surface rather than crash.

    `quiet=True` suppresses the warning (used for per-frame animation
    windows, where a transiently-too-small visible window is expected
    rather than a genuine user-data problem).
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        if not quiet:
            warnings.warn(
                f"surface: dataset{dataset_label} has fewer than 4 points "
                "(need >= 4 for a 3-D hull); skipping its surface."
            )
        return None
    try:
        return smooth_hull_3d(
            points,
            rounds=int(spec.get("smoothing", 3)),
            pre_inflate=float(spec.get("pre_inflate", 1.0)),
        )
    except ValueError as exc:
        if not quiet:
            warnings.warn(
                f"surface: dataset{dataset_label} is degenerate ({exc}); "
                "skipping its surface."
            )
        return None


def build_outline_2d(points, spec, dataset_label="", quiet=False):
    """Build a smooth closed outline for one dataset's 2-D points.

    Returns an ``(n, 2)`` array, or ``None`` (optionally warning) if there
    are fewer than 3 points or they are collinear/degenerate.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        if not quiet:
            warnings.warn(
                f"surface: dataset{dataset_label} has fewer than 3 points "
                "(need >= 3 for a 2-D hull); skipping its surface."
            )
        return None
    try:
        return smooth_hull_2d(points)
    except ValueError as exc:
        if not quiet:
            warnings.warn(
                f"surface: dataset{dataset_label} is degenerate ({exc}); "
                "skipping its surface."
            )
        return None


def view_vector(elev, azim):
    """Convert matplotlib-style `elev`/`azim` (degrees) into the unit
    "direction from the scene towards the camera" vector that
    :func:`hypertools.plot.meshutil.blinn_phong_colors`/`backface_cull`
    expect."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def surface_cube_scale(mesh_list, default=1.0, margin=1.02):
    """Return the half-width of the smallest axis-aligned cube, centered at
    the origin, that contains every vertex across `mesh_list` (GH #109
    round 2).

    Data is centered/scaled into the [-1, 1] cube upstream (in `plot()`)
    BEFORE surface meshes are built; `build_mesh_3d`/`smooth_hull_3d`'s
    `pre_inflate` (bulges the hull outward before smoothing) plus the
    smoothing's own overshoot -- and, for small/sparse point clouds,
    `_rescale_for_containment`'s further grow-only correction -- can push
    mesh vertices outside that cube, so both backends' axes cube and
    (for matplotlib) axis limits / (for plotly) scene ranges must be
    widened to whatever the ACTUAL built meshes need, not assumed to be 1.

    Parameters
    ----------
    mesh_list : sequence of (verts, faces) tuples, or None entries
        Built meshes (e.g. from :func:`build_mesh_3d`); ``None`` entries
        (no surface spec for that dataset, or too few/degenerate points)
        are skipped.
    default : float, optional
        The cube half-width to return when no mesh vertex exceeds it (the
        library's standard [-1, 1] data cube). Default 1.0.
    margin : float, optional
        Safety factor applied to the observed max |vertex coordinate| when
        it exceeds `default`, so meshes clear the drawn cube rather than
        just touching it. Default 1.02.

    Returns
    -------
    float
        `default` if every mesh vertex already fits within it, otherwise
        ``margin * max(|vertex coordinate|)`` across all meshes.
    """
    max_abs = 0.0
    for mesh in mesh_list:
        if mesh is None:
            continue
        verts = np.asarray(mesh[0], dtype=float)
        if verts.size:
            m = float(np.max(np.abs(verts)))
            if m > max_abs:
                max_abs = m
    if max_abs <= default:
        return float(default)
    return max_abs * margin
