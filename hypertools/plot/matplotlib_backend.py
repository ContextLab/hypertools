#!/usr/bin/env python
"""Low-level matplotlib drawing for `hypertools.plot` (static and
animated figures, explore mode, trails, and morphs).

The sole entry point is `_draw`, called by `hypertools.plot.plot` after
it has resolved data, formats, colors, and animation settings; everything
else here is private helpers. This module was renamed from ``draw.py`` in
HyperTools 1.0 (``draw.py`` remains as a compatibility shim).
"""
import functools
import itertools
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

# matplotlib's own fmt-string parser (returns (linestyle, marker, color)
# with 'None' sentinels where the fmt pins a component off) -- private but
# stable for 15+ years; used so hypertools' marker+line fmt splitting
# resolves color letters and linestyles exactly per the fmt grammar
# (release-1.0 audit, F01-003/F01-009). Guarded so a future matplotlib
# relocation degrades to the historical behavior instead of crashing.
try:
    from matplotlib.axes._base import _process_plot_format
except ImportError:  # pragma: no cover
    _process_plot_format = None
from .animate import HyperFuncAnimation
import matplotlib.patches as patches
from .._shared.helpers import *
from ..core.model import external_stacklevel
from .meshutil import (backface_cull, blinn_phong_colors,
                       vertex_colors_from_points, face_colors_from_vertex_colors)
from .surface import (
    build_mesh_3d,
    build_outline_2d,
    mpl_lighting_kwargs,
    surface_cube_scale,
    view_vector,
)
from .trails import (RunWindow, anim_window_bounds, broadcast_trail_flag,
                     dataset_window_bounds, head_window_frames)
from . import morph as _morph
from .density import (
    DENSITY_DEFAULTS,
    POOLED_COLOR,
    alpha_colormap,
    bbox_extent,
    density_alpha_boost,
    fit_kde,
    iso_surfaces_3d,
    kde_grid_2d,
    kde_grid_3d,
    resolve_grid,
    skimage_measure,
    resolve_iso_fracs_alphas,
)


def _apply_title(ax, text, font=None, title_kwargs=None):
    """Set `ax`'s title, honoring the resolved `font=` and `title_kwargs=`
    (GH #285).

    With `title_kwargs=None` this makes EXACTLY the call hypertools has
    always made (`ax.set_title(text)`, or `ax.set_title(text,
    fontproperties=font)` when a `font=` was resolved), so an un-styled
    title's Text artist is byte-identical to before.

    `fontproperties` is inserted BEFORE the individual font properties,
    because `set_title` applies its kwargs in dict order and a
    `fontproperties` REPLACES the Text's whole FontProperties, size
    included (measured on matplotlib 3.10: `set_title(t, fontsize=31,
    fontproperties=fp)` renders at fp's own 10pt, while the same call with
    the two swapped renders at 31). So `title_kwargs={'size': 20}` beats
    the resolved font's size -- which is what asking for a size means.

    Lives here, in the backend, so `plot.py`'s per-frame title updater and
    the static draw below set titles through one function -- the split
    between them is exactly the bug GH #285 reports (a resolved `font=`
    reached the static title and never a per-segment one).
    """
    if title_kwargs:
        kwargs = {}
        if font is not None and 'fontproperties' not in title_kwargs:
            kwargs['fontproperties'] = font
        kwargs.update(title_kwargs)
        return ax.set_title(text, **kwargs)
    if font is not None:
        return ax.set_title(text, fontproperties=font)
    return ax.set_title(text)


# --------------------------------------------------------------------------
# companion= panels (GH #285): extra 2-D panels laid out beside an animated
# plot and revealed in lockstep with it. `plot.py` validates the specs and
# drives the per-frame reveal; the drawing lives here, beside every other
# matplotlib artist hypertools creates.
# --------------------------------------------------------------------------

#: Vertical (or horizontal) gap, in figure fractions, between a companion
#: panel and the plot it hangs off. Small and fixed: the caller's knobs are
#: `size` (how much of the figure the panel gets) and `pad` (how much of
#: that is left for the panel's own ticks and label).
COMPANION_GAP = 0.02

#: The full, unrevealed series is drawn once underneath in this grey, so the
#: revealed part reads against the shape it is filling in.
COMPANION_GHOST_COLOR = '0.85'


def _companion_xy(data):
    """`(x, y)` for a companion panel's ``data``: one column is y against
    row number, two columns are (x, y)."""
    if data.shape[1] == 1:
        return np.arange(data.shape[0], dtype=float), data[:, 0]
    return data[:, 0], data[:, 1]


def _companion_rolling_mean(y, window):
    """Trailing rolling mean of `y` over `window` rows, NaN until the window
    is full -- the same convention `pandas.Series.rolling(window).mean()`
    uses, so a caller who computed one by hand gets the identical line."""
    out = np.full(y.shape[0], np.nan)
    if window <= y.shape[0]:
        c = np.concatenate([[0.0], np.cumsum(y)])
        out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def _grow_for_companion(fig, spec):
    """Make room for one companion panel by GROWING the figure, keeping
    every existing axes at exactly the absolute size it already had, and
    return the new panel's rect in the new figure's fractions.

    Growing rather than shrinking is the same decision (and the same
    reasoning) as `plot._reserve_animated_3d_title_margin`: an animated 3-D
    axes is deliberately maximised to the full canvas so a rotating zoomed
    cube never clips, and shrinking it to make room would shrink the
    rendered cube with it.
    """
    w_in, h_in = fig.get_size_inches()
    size = spec['size']
    if spec['position'] == 'bottom':
        extra = h_in * size / (1.0 - size)
        new_w, new_h = w_in, h_in + extra
        for axes in fig.axes:
            pos = axes.get_position()
            axes.set_position([pos.x0, (pos.y0 * h_in + extra) / new_h,
                               pos.width, pos.height * h_in / new_h])
    else:
        extra = w_in * size / (1.0 - size)
        new_w, new_h = w_in + extra, h_in
        for axes in fig.axes:
            pos = axes.get_position()
            axes.set_position([pos.x0 * w_in / new_w, pos.y0,
                               pos.width * w_in / new_w, pos.height])
    try:
        fig.set_layout_engine('none')
    except Exception:                                    # noqa: BLE001
        pass
    fig.set_size_inches(new_w, new_h)
    pad = spec['pad']
    if spec['position'] == 'bottom':
        height = size - pad - COMPANION_GAP
        if height <= 0:
            raise ValueError(
                f"companion= panel: size={size} leaves no room for the "
                f"panel once pad={pad} is reserved below it for its ticks "
                "and label; raise size= or lower pad=.")
        return [0.12, pad, 0.80, height]
    width = size - pad - COMPANION_GAP
    if width <= 0:
        raise ValueError(
            f"companion= panel: size={size} leaves no room for the panel "
            f"once pad={pad} is reserved to its left for its ticks and "
            "label; raise size= or lower pad=.")
    return [1.0 - size + pad, 0.15, width, 0.75]


def add_companion_panel(fig, spec, cmap=None, norm=None, font=None):
    """Draw one `companion=` panel and return the state its per-frame
    updater needs (GH #285).

    The panel shows the whole series once, faintly, then reveals it: a line
    (or, with ``hue=``, a `LineCollection` coloured through the plot's own
    colour scale) up to the reveal head, an optional trailing rolling mean,
    and an optional marker on the head itself.
    """
    from matplotlib.collections import LineCollection

    rect = _grow_for_companion(fig, spec)
    pax = fig.add_axes(rect)
    x, y = _companion_xy(spec['data'])
    color = spec['color'] or 'C0'

    pax.plot(x, y, color=COMPANION_GHOST_COLOR, linewidth=0.6)
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    if spec['hue'] is not None:
        # the PLOT's own resolved colour scale when there is one (a
        # continuous `hue=` gives `colorbar_info` a cmap and a norm), so the
        # panel and the trajectory it accompanies read the same value the
        # same way; otherwise a plain linear scale over the panel's own
        # values.
        if cmap is None or norm is None:
            from matplotlib.colors import Normalize
            cmap = plt.get_cmap('viridis') if cmap is None else cmap
            norm = Normalize(float(np.nanmin(spec['hue'])),
                             float(np.nanmax(spec['hue'])))
        revealed = pax.add_collection(
            LineCollection([], cmap=cmap, norm=norm, linewidths=1.2))
        head_colors = [cmap(norm(v)) for v in spec['hue']]
    else:
        revealed = LineCollection([], colors=[color], linewidths=1.2)
        pax.add_collection(revealed)
        head_colors = None
    trend = None
    rolling = None
    if spec['smooth'] is not None:
        rolling = _companion_rolling_mean(y, spec['smooth'])
        (trend,) = pax.plot([], [], color='black', linewidth=1.4)
    head = None
    if spec['marker']:
        # clip_on=False so the marker stays whole at either end of the span
        (head,) = pax.plot([], [], 'o', markersize=7,
                           markeredgecolor='black', color=color,
                           clip_on=False)
    span = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    pax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    pax.set_ylim(float(np.nanmin(y)) - 0.05 * span,
                 float(np.nanmax(y)) + 0.05 * span)
    pax.spines[['top', 'right']].set_visible(False)
    label_kw = {} if font is None else {'fontproperties': font}
    if spec['xlabel'] is not None:
        pax.set_xlabel(spec['xlabel'], **label_kw)
    if spec['ylabel'] is not None:
        pax.set_ylabel(spec['ylabel'], **label_kw)
    if font is not None:
        for text in pax.get_xticklabels() + pax.get_yticklabels():
            text.set_fontproperties(font)
    return dict(axes=pax, x=x, y=y, segments=segments, revealed=revealed,
                hue=spec['hue'], head_colors=head_colors, trend=trend,
                rolling=rolling, head=head, reveal=spec['reveal'],
                n_rows=spec['data'].shape[0])


def update_companion_panel(panel, i):
    """Reveal a companion panel up to (and including) input row `i`.

    Every artist is ASSIGNED on every call, never left untouched -- the
    portable-callback rule in `FrameContext`: matplotlib redelivers the same
    artists each frame, so a skipped assignment leaves the previous frame's
    state on screen.
    """
    i = int(min(max(i, 0), panel['n_rows'] - 1))
    if not panel['reveal']:
        i = panel['n_rows'] - 1
    panel['revealed'].set_segments(panel['segments'][:i])
    if panel['hue'] is not None:
        panel['revealed'].set_array(panel['hue'][1:i + 1])
    if panel['trend'] is not None:
        panel['trend'].set_data(panel['x'][:i + 1], panel['rolling'][:i + 1])
    if panel['head'] is not None:
        panel['head'].set_data([panel['x'][i]], [panel['y'][i]])
        if panel['head_colors'] is not None:
            panel['head'].set_markerfacecolor(panel['head_colors'][i])
    return panel


def _legend_proxy_handles(entries, fmt=None):
    """`Line2D` proxy handles for explicit legend entries (GH #285).

    `entries` is a list of ``(label, color)`` pairs -- from a matrix/
    mixture `hue=`'s palette columns, or from an explicit
    `legend_colors=[(label, color), ...]`. Each is drawn as a short line
    swatch, or as a marker swatch when the plot's own `fmt` draws markers
    only, so the key looks like the artists it names.
    """
    from matplotlib.lines import Line2D
    style = {'linestyle': '-', 'linewidth': 2, 'marker': None}
    first_fmt = fmt[0] if isinstance(fmt, (list, tuple)) and fmt else fmt
    if isinstance(first_fmt, str) and _process_plot_format is not None:
        try:
            linestyle, marker, _ = _process_plot_format(first_fmt)
        except Exception:  # noqa: BLE001 - an unparseable fmt keeps the line
            linestyle, marker = '-', None
        if linestyle in (None, 'None') and marker not in (None, 'None'):
            style = {'linestyle': 'None', 'marker': marker, 'markersize': 8}
    return [Line2D([], [], color=color, label=label, **style)
            for label, color in entries]


def legend_call_kwargs(is_3d=False, zlabel=None, font=None,
                       legend_kwargs=None):
    """The `ax.legend(...)` keyword arguments hypertools draws a legend with.

    Factored out of `_draw` (GH #285) so `plot()` can REBUILD the legend
    with identical placement/styling after it adds a labelled overlay
    (a multi-model `predict=` fan, or `truth=`) -- the legend is built
    inside `_draw`, before those artists exist, and a rebuild that guessed
    at the placement would move the legend on exactly the figures that
    gained an entry.
    """
    # a 3-D zlabel is drawn in the axes' right margin -- exactly where
    # the legend is anchored -- so shift the legend further right when
    # both are requested (release-1.0 audit, F10-005: the zlabel text
    # rendered directly on top of the legend's first entry).
    _legend_x = 1.18 if (zlabel is not None and is_3d) else 1.02
    call = dict(loc='center left', bbox_to_anchor=(_legend_x, 0.5),
                borderaxespad=0.0, frameon=False)
    if font is not None:
        call['prop'] = font
    # `legend_kwargs=` (GH #285) is applied LAST, so a caller's
    # loc=/bbox_to_anchor=/frameon=/fontsize= wins over the defaults
    # above -- which is the whole point of the kwarg.
    if legend_kwargs:
        call.update(legend_kwargs)
    return call


def _recolor_legend_handles(legend, colors):
    """Apply `legend_colors=`'s plain colour list to an already-built
    legend, in entry order (GH #285)."""
    handles = legend.legend_handles
    if len(colors) != len(handles):
        raise ValueError(
            f"legend_colors has {len(colors)} entries but the legend has "
            f"{len(handles)}; pass one color per legend entry, or pass "
            "(label, color) pairs to define the entries outright.")
    for handle, color in zip(handles, colors):
        for setter in ('set_color', 'set_markerfacecolor',
                       'set_markeredgecolor'):
            if hasattr(handle, setter):
                getattr(handle, setter)(color)


def _resolve_surface_color(spec, fallback_rgb):
    """Base RGB for one dataset's surface: `spec['color']` if given,
    otherwise the dataset's own drawn color (`fallback_rgb`)."""
    return mcolors.to_rgb(spec["color"]) if spec["color"] is not None else fallback_rgb


def _draw_one_density_2d(ax, pts, spec, color, label="", clip_unit=True):
    """Draw a single subtle alpha-ramped ``imshow`` KDE layer for one
    dataset (or the pooled cloud), below the data (``zorder=-1``).

    `clip_unit` (GH #285): clip the glow to hypertools' unit frame square,
    which only exists under ``axis_scale='unit'``. Under ``'data'`` the
    drawn coordinates are the data's own and there is no square, so
    clipping to ``[-1, 1]`` would erase the whole layer -- the KDE is left
    unclipped there instead."""
    kde = fit_kde(pts, dataset_label=label)
    if kde is None:
        return
    gridsize = resolve_grid(spec, 2)
    _, _, Z, extent = kde_grid_2d(pts, kde, gridsize=gridsize)
    cmap = alpha_colormap(color, spec["alpha"])
    im = ax.imshow(Z, origin="lower", extent=extent, aspect="auto",
                   cmap=cmap, interpolation="bilinear", zorder=-1)
    im.set_label("_nolegend_")
    if not clip_unit:
        return
    # clip the KDE glow to the drawn frame box: the density grid extends
    # ~15% beyond the data bounds (and the KDE bandwidth blows up for tiny
    # n), so for sparse data the haze flooded the figure margins OUTSIDE
    # hypertools' own frame square (release-1.0 audit,
    # D05-gallery-data-text-009). The 2-D paths always rescale data into
    # the [-1, 1] box and draw the frame via plot_square(scale=1), so the
    # frame rectangle is fixed in data coordinates.
    im.set_clip_path(patches.Rectangle((-1.0, -1.0), 2.0, 2.0,
                                       transform=ax.transData))


def _draw_density_2d(ax, points_list, density, density_colors,
                     clip_unit=True):
    """Draw each dataset's (or, with ``per_group=False``, one pooled) 2-D
    KDE density layer (GH #108/#191)."""
    if density[0] is not None and not density[0].get("per_group", True):
        all_pts = np.vstack([np.asarray(p)[:, :2] for p in points_list])
        _draw_one_density_2d(ax, all_pts, density[0], POOLED_COLOR,
                             label=" (pooled)", clip_unit=clip_unit)
        return
    for i, (pts, spec) in enumerate(zip(points_list, density)):
        if spec is None:
            continue
        _draw_one_density_2d(ax, np.asarray(pts)[:, :2], spec,
                             density_colors[i], label=f" {i}",
                             clip_unit=clip_unit)


def _draw_one_density_3d(ax, pts, spec, color, label="", boost=1.0):
    """Draw a single dataset's (or the pooled cloud's) 3-D density layer:
    nested translucent iso-surfaces via marching cubes if scikit-image is
    available, else a translucent scatter "fog" fallback (GH #108/#191).

    `boost` (see :func:`~.density.density_alpha_boost`) multiplies the
    alpha on top of the user's own `spec['alpha']`, so a dataset that's
    small relative to the whole scene (widely-separated clusters, GH #108
    round 2) stays visible instead of vanishing -- it is ``1.0`` (a no-op)
    for a scene-filling dataset."""
    kde = fit_kde(pts, dataset_label=label)
    if kde is None:
        return
    alpha_scale = spec["alpha"] / DENSITY_DEFAULTS["alpha"] * boost
    if skimage_measure() is not None:     # installs [density3d] on demand
        gridsize = resolve_grid(spec, 3)
        _, _, _, D, lo, spacing = kde_grid_3d(pts, kde, gridsize=gridsize)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        levels = spec.get("levels", DENSITY_DEFAULTS["levels"])
        fracs, base_alphas = resolve_iso_fracs_alphas(levels)
        for (verts, faces), base_alpha in zip(
                iso_surfaces_3d(D, lo, spacing, fracs=fracs), base_alphas):
            coll = Poly3DCollection(
                verts[faces], alpha=min(base_alpha * alpha_scale, 1.0),
                facecolor=color, edgecolor="none", linewidths=0, shade=False,
            )
            coll.set_label("_nolegend_")
            # axes-box slicing fix (see `plot_cube`): unclip so this layer
            # never gets sliced by Axes3D's shrunk-square viewport.
            coll.set_clip_on(False)
            ax.add_collection3d(coll)
    else:
        warnings.warn(
            f"density: scikit-image is not installed and could not be "
            f"installed on demand -- dataset{label}'s 3-D density falls back "
            "to a translucent scatter 'fog' instead of smooth iso-surfaces. "
            "Install it with `pip install \"hypertools[density3d]\"`, or use "
            "backend='plotly' for full volumetric rendering.",
            UserWarning,
            stacklevel=external_stacklevel())
        rng = np.random.default_rng()
        fog = kde.resample(4000, seed=rng).T
        fog_coll = ax.scatter(fog[:, 0], fog[:, 1], fog[:, 2], s=6, c=[color],
                              alpha=min(0.03 * alpha_scale, 1.0),
                              edgecolors="none", depthshade=False,
                              label="_nolegend_")
        fog_coll.set_clip_on(False)


def _draw_density_3d(ax, points_list, density, density_colors):
    """Draw each dataset's (or, with ``per_group=False``, one pooled) 3-D
    KDE density layer (GH #108/#191). Computed ONCE from the full point set
    passed in -- callers are responsible for passing the FULL data (not a
    per-frame animation window): a KDE evaluation is far too slow (~536ms
    @ 50**3) to redo every animation frame.

    Each per-dataset layer's alpha is boosted (GH #108 round 2) by how
    small that dataset's own bounding box is relative to the bounding box
    of the WHOLE scene (all datasets combined) -- see
    :func:`~.density.density_alpha_boost`."""
    if density[0] is not None and not density[0].get("per_group", True):
        all_pts = np.vstack([np.asarray(p)[:, :3] for p in points_list])
        _draw_one_density_3d(ax, all_pts, density[0], POOLED_COLOR,
                             label=" (pooled)", boost=1.0)
        return
    scene_pts = np.vstack([np.asarray(p)[:, :3] for p in points_list])
    scene_extent = bbox_extent(scene_pts)
    for i, (pts, spec) in enumerate(zip(points_list, density)):
        if spec is None:
            continue
        pts3 = np.asarray(pts)[:, :3]
        boost = density_alpha_boost(bbox_extent(pts3), scene_extent)
        _draw_one_density_3d(ax, pts3, spec, density_colors[i], label=f" {i}",
                             boost=boost)


def _shade_and_cull_3d(ax, mesh_list, surface, surface_colors, elev, azim,
                       prior_colls=None, surface_point_colors=None):
    """(Re)build a ``Poly3DCollection`` per dataset from PRECOMPUTED
    ``(verts, faces)`` meshes, shading/culling for the CURRENT `elev`/`azim`,
    removing `prior_colls` first (animation frame swap). Returns the new
    per-dataset collection list (``None`` where that dataset has no surface)
    so the caller can pass it back in as `prior_colls` next frame.

    `surface_point_colors` (QC 2026-07): optional list, one entry per dataset,
    of ``(points, per_point_rgb)`` -- when present for a dataset, each mesh
    face is colored by an inverse-distance-weighted blend of the enclosed
    points' colors (`vertex_colors_from_points`) instead of one flat
    surface color, so a `hue=`'d surface matches the hue of its points."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if prior_colls:
        for c in prior_colls:
            if c is not None:
                c.remove()

    v = view_vector(elev, azim)
    new_colls = []
    for i, (mesh, spec) in enumerate(zip(mesh_list, surface)):
        if spec is None or mesh is None:
            new_colls.append(None)
            continue
        verts, faces = mesh
        spc = surface_point_colors[i] if surface_point_colors else None
        if spc is not None:
            pts, cols = spc
            vcolors = vertex_colors_from_points(verts, pts, cols)
            base_rgb = face_colors_from_vertex_colors(vcolors, faces)
        else:
            base_rgb = _resolve_surface_color(spec, surface_colors[i])
        light_kw = mpl_lighting_kwargs(spec)
        rgba = blinn_phong_colors(verts, faces, base_rgb, v, **light_kw)
        cull = backface_cull(verts, faces, v)
        alpha = spec["alpha"]
        rgba = rgba.copy()
        rgba[:, 3] = alpha
        coll = Poly3DCollection(
            verts[faces[cull]], facecolors=rgba[cull], edgecolors="none",
            linewidths=0, shade=False, antialiaseds=(alpha < 1.0),
        )
        coll.set_label("_nolegend_")
        # axes-box slicing fix (see `plot_cube`): unclip so a surface hull
        # never gets sliced by Axes3D's shrunk-square viewport.
        coll.set_clip_on(False)
        ax.add_collection3d(coll)
        new_colls.append(coll)
    return new_colls


def _build_mesh_list(points_list, surface, quiet=False):
    """Build one `(verts, faces)` mesh (or `None`) per dataset from the
    CURRENT per-dataset points (`points_list`) -- shared by
    `_mesh_and_draw_3d` and the cube-scale computation (GH #109 round 2),
    so the mesh is never built twice for the same points."""
    return [
        build_mesh_3d(np.asarray(pts), spec, dataset_label=f" {i}", quiet=quiet)
        if spec is not None else None
        for i, (pts, spec) in enumerate(zip(points_list, surface))
    ]


def _mesh_and_draw_3d(ax, points_list, surface, surface_colors, elev, azim,
                      prior_colls=None, quiet=False,
                      surface_point_colors=None):
    """Build fresh meshes from the CURRENT per-dataset point windows
    (`points_list`) and delegate to `_shade_and_cull_3d`. Used whenever the
    visible point window changes (static plots; 'parallel'/'serial'
    animation frames).

    `surface_point_colors` (release-1.0 audit, F07-005): optional list, one
    entry per dataset, of ``(points, per_point_rgb)`` WINDOWED to the same
    slice as `points_list` -- forwarded to `_shade_and_cull_3d` so animated
    hue'd surfaces keep the same per-vertex hue coloring static plots use
    (previously animation frames dropped it, rendering a rainbow-hue hull
    flat gray)."""
    mesh_list = _build_mesh_list(points_list, surface, quiet=quiet)
    return _shade_and_cull_3d(ax, mesh_list, surface, surface_colors, elev,
                              azim, prior_colls=prior_colls,
                              surface_point_colors=surface_point_colors)


def _window_surface_point_colors(surface_point_colors, i, start, stop):
    """Slice dataset `i`'s ``(points, per_point_rgb)`` hue-color bundle to
    the ``[start:stop]`` row window an animation frame is drawing -- rows of
    the bundle are aligned 1:1 with the drawn dataset's rows (both come
    from the same post-interpolation `xform`; see `plot.py`'s
    `surface_point_colors` construction) -- or ``None`` if that dataset has
    no per-point hue colors (release-1.0 audit, F07-005)."""
    if not surface_point_colors or i >= len(surface_point_colors):
        return None
    spc = surface_point_colors[i]
    if spc is None:
        return None
    pts, cols = spc
    return (np.asarray(pts)[start:stop], np.asarray(cols)[start:stop])


def _fill_and_draw_2d(ax, points_list, surface, surface_colors):
    """Draw one smooth filled ``ax.fill`` outline per dataset (static 2-D
    only), below the data lines (``zorder=0``)."""
    for i, (pts, spec) in enumerate(zip(points_list, surface)):
        if spec is None:
            continue
        outline = build_outline_2d(np.asarray(pts), spec, dataset_label=f" {i}")
        if outline is None:
            continue
        color = _resolve_surface_color(spec, surface_colors[i])
        patches_ = ax.fill(outline[:, 0], outline[:, 1], color=color,
                           alpha=spec["alpha"], zorder=0)
        for p in patches_:
            p.set_label("_nolegend_")


def _hide_no_keep_points(artists, surface):
    """Hide (rather than remove) the primary line/marker artist for any
    dataset whose surface spec has ``keep_points=False`` -- the artist stays
    in `ax.lines`/`legend` bookkeeping, just invisible."""
    for i, spec in enumerate(surface):
        if spec is not None and not spec.get("keep_points", True):
            if i < len(artists) and artists[i] is not None:
                artists[i].set_visible(False)


def _anim_box_zoom(zoom):
    """``set_box_aspect`` zoom factor for ANIMATED 3D plots.

    Slightly zoomed OUT from the historical ``10 / (9 - zoom)`` mapping so
    the wireframe bounding box keeps a comfortable margin at every rotation
    angle and is never clipped (Jeremy's animated-plot zoom-out request).
    At the default ``zoom=1`` this is ``9 / 8 = 1.125`` (previously 1.25).
    Static plots are unaffected: they use the default box aspect (zoom=1).
    """
    return 9.0 / max(0.5, 9.0 - zoom)


# NOTE: the parallel/'window' head + trail geometry lives in
# `trails.anim_window_bounds` (imported above), not here, because the PLOTLY
# renderer calls that same function on the same arguments. See its docstring
# for the backend divergences a shared callee closed.


def _make_save_dpi_safe(line_ani):
    """Guard every future ``line_ani.save(...)`` call against a real
    matplotlib/canvas-manager interaction that corrupts the rendered pixel
    dimensions at some save dpi values -- this is what made sphinx-gallery's
    thumbnail GIFs (re-saved via ``anim.save(gif_path, dpi≈31)``, at a much
    lower dpi than the figure's own) come out with the cube's bounding box
    sheared and cut off at every rotation, for every animation style.

    Root cause: ``Animation.save()`` builds its `MovieWriter` via
    ``writer.setup(fig, outfile, dpi)``, and `MovieWriter._adjust_frame_size`
    reads ``self.codec`` to decide whether to "correct" the figure size for
    macroblock alignment (via ``Figure.set_size_inches(w, h, forward=True)``)
    -- but at that point ``self.codec`` is still whatever
    ``rcParams['animation.codec']`` defaults to (``'h264'``), because it is
    only set from the OUTPUT FILE'S suffix later, when ``self.output_args``
    is first accessed. So this "h264-only" resize runs for every format,
    including GIF/APNG/SVG, whenever the default writer is used (matplotlib
    picks the default writer -- typically 'ffmpeg' if installed -- for any
    ``anim.save(...)`` call that doesn't pass ``writer=`` explicitly, which
    is exactly how sphinx-gallery and other third-party callers invoke it).

    ``forward=True`` is normally a harmless no-op on a plain (Agg) canvas.
    But hypertools creates animated (and interactive) figures under a REAL
    interactive backend so they can be shown live -- the figure's canvas
    keeps that interactive, OS-backed ``.manager`` for its whole life, even
    after ``show=False`` switches pyplot's *default* backend back (that only
    affects figures created afterward, not this one). When the resize above
    runs, ``manager.resize(...)`` resizes a REAL OS window, and the figure
    size read back afterward gets snapped to that window's (coarser)
    pixel/point grid -- turning the deliberately-EVEN pixel size
    ``_adjust_frame_size`` computed (e.g. 198x148 at dpi=31) into an ODD one
    (e.g. 197x147). Piping ODD-width/height raw RGBA frames through
    ffmpeg's GIF ``palettegen``/``paletteuse`` filter chain visibly corrupts
    every frame (a diagonal shear, as if the box were zoomed in and its
    corner cut off).

    matplotlib's own ``Animation.save()`` already null-guards this exact
    thing (it sets ``fig.canvas.manager = None`` for the duration of the
    save, specifically to prevent a live GUI window from being resized) --
    but it does so too late, AFTER ``writer.setup()`` (and its harmful
    resize) already ran. Pre-empting ``manager = None`` here, before
    matplotlib's own (too-late) guard takes over, makes the ``forward=True``
    resize a no-op for the whole call, regardless of the requested dpi --
    exactly like saving at the figure's own (native) dpi already was, which
    never needed the resize in the first place (its pixel size is already
    an even multiple).
    """
    real_save = line_ani.save

    @functools.wraps(real_save)
    def save(*args, **kwargs):
        """Call the original `line_ani.save`, temporarily clearing the figure's canvas manager.

        Prevents matplotlib's writer-setup dpi-correction resize from
        resizing a real, OS-backed interactive window mid-save (which
        would corrupt the pixel dimensions of the saved animation -- see
        `_make_save_dpi_safe`); the manager is restored afterward
        regardless of whether the save succeeds.
        """
        fig = line_ani._fig
        canvas = getattr(fig, "canvas", None)
        manager = getattr(canvas, "manager", None) if canvas is not None else None
        if canvas is not None:
            canvas.manager = None
        try:
            return real_save(*args, **kwargs)
        finally:
            if canvas is not None:
                canvas.manager = manager

    line_ani.save = save
    return line_ani


def serial_reveal_counts(lengths, num, total_frames):
    """Rows revealed per dataset at frame `num` of a serial animation.

    THE reveal schedule. `update_lines_serial` (3-D), `update_lines_serial_2d`
    and `FrameContext.revealed_counts` all read it, and
    `plot._apply_multicolor_animation` recovers its hue window from it, so the
    formula exists once. Equivalent to the historical inline code
    (`revealed = total_points * num / max(1, total_frames - 1)`; per dataset
    `shown = int(np.clip(revealed - start, 0, n_pts))`).
    """
    total_points = sum(lengths)
    revealed = total_points * num / max(1, total_frames - 1)
    counts, remaining = [], revealed
    for length in lengths:
        counts.append(int(max(0, min(length, remaining))))
        remaining -= length
    return counts


def serial_current_index(counts, lengths):
    """``(index, fraction)`` of the dataset mid-reveal at these counts."""
    done = -1
    for i, (shown, length) in enumerate(zip(counts, lengths)):
        if 0 < shown < length:
            return i, (shown - 1) / max(1, length - 1)
        if shown >= length:
            done = i
    if done < 0:
        return 0, 0.0
    return done, 1.0


def _draw(
    x,
    legend=None,
    title=None,
    labels=False,
    show=True,
    kwargs_list=None,
    fmt=None,
    antialias=True,
    raw_data=None,
    animate=False,
    tail_duration=2,
    focused=None,
    rotations=1,
    zoom=1,
    chemtrails=False,
    precog=False,
    bullettime=False,
    frame_rate=30,
    elev=10,
    azim=-60,
    duration=30,
    explore=False,
    size=None,
    ax=None,
    frame_kwargs=None,
    surface=None,
    surface_colors=None,
    surface_point_colors=None,
    density=None,
    density_colors=None,
    morph_tags=None,
    morph_colors=None,
    morph_samples=None,
    morph_loop=False,
    font=None,
    label_alpha=0.5,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    frame_hooks=None,
    ownership=None,
    title_kwargs=None,
    legend_kwargs=None,
    legend_entries=None,
    legend_colors=None,
    axis_scale='unit',
    xlim=None,
    ylim=None,
    x_date=False,
):
    """
    Draws the plot

    `raw_data` (GH #141): the PRE-interpolation per-dataset points, same
    length as `x`/`fmt`. Used only by the STATIC (non-animated) plot1D/2D/
    3D functions below: a dataset whose `fmt` combines a marker AND a line
    (e.g. 'o-') is drawn as two artists -- a smoothed line from `x` (the
    already-interpolated data) plus markers at the raw sample points from
    `raw_data` -- so markers land on the true data regardless of how dense
    the smoothed line is. Ignored (may be None) for pure line/marker-only
    styles and for every ANIMATED style, which still draw marker+line
    combos as a single artist against the (now also smoothed, since the
    interpolation gate itself was fixed for GH #141) `x` data -- so an
    animated 'o-' plot's line is correctly smoothed, but its markers
    currently render at the interpolated points rather than only the
    original samples; splitting the animated marker/line artists frame-by-
    frame was judged out of scope for this fix.

    `ownership` (a `hypertools.plot.ownership.TraceOwnership`, or None): which
    source dataset each drawn trace came from and which of its rows. When
    given, the parallel/'window' updaters pace every trace of ONE dataset
    from that dataset's single clock (`trails.dataset_window_bounds`), so a
    `hue=`/`cluster=` regrouped trajectory sweeps once in row order instead of
    growing in several places at once. `plot()` passes an `identity` ownership
    for unregrouped figures, where the projection is provably the identity, so
    both cases take one code path; it passes None for anything whose traces do
    not correspond to input datasets (marker-only categorical regrouping
    groups globally by category), and those keep `anim_window_bounds` directly.

    `axis_scale` (GH #285): ``'unit'`` (the historical behaviour) draws the
    hypertools frame square and pins the 2-D axes to ``(-1.1, 1.1)`` --
    `plot()` has already mean-centred and rescaled the data into ``[-1, 1]``
    for it. ``'data'`` draws NO frame square, leaves matplotlib's own ticks
    and spines visible, and takes its limits from `xlim`/`ylim` (which
    `plot()` computes from the full data, forecasts included, so an
    animation's viewport never jumps) or from matplotlib's autoscale when
    both are None. 3-D is always ``'unit'`` (`plot()` refuses the other).

    `xlim`/`ylim` (GH #285): explicit ``(low, high)`` axis limits, honoured
    on every non-``'unit'`` 2-D/1-D path (static and animated). ``None``
    leaves the axis alone.

    `x_date` (GH #285): the x values are matplotlib date numbers
    (`matplotlib.dates.date2num`), as `ndims=1` series mode produces from a
    `DatetimeIndex`; puts a date locator/formatter on the x axis so the
    ticks read as real dates.

    `frame_hooks` (the public `on_frame=` hook, plan 1.1 Task 7): a
    `hypertools.plot.animation_context.FrameHooks` registry, created once by
    `plot()` and threaded in here so every animated updater below can call
    `frame_hooks.record(...)` with whatever it knows about the frame just
    drawn. `None` (the default) when no `on_frame=` was requested, in which
    case every updater's `if frame_hooks is not None:` guard is a no-op.
    `plot()` installs the actual callback dispatch as the outermost wrapper
    of the returned `FuncAnimation._func`, AFTER this function returns (see
    `FrameHooks.dispatch`) -- updaters here only ever record state, never
    invoke callbacks.
    """

    # chemtrails/precog/bullettime (GH #127): normalize to one bool per
    # dataset now, at the top of `_draw`, BEFORE any nested closure below is
    # defined -- `update_lines_parallel`/`animate_plot3D` close over these
    # names, so reassigning them here (rather than deeper inside) is what
    # every closure sees at call time. `plot.py` already broadcasts/
    # validates against the FINAL (post cluster/hue-reshape) dataset count
    # before calling `_draw`, but this call is defensive (mirrors
    # `broadcast_surface`'s pattern) so `_draw` also works when called
    # directly (as some tests do) with a bare bool.
    chemtrails = broadcast_trail_flag(chemtrails, len(x), "chemtrails")
    precog = broadcast_trail_flag(precog, len(x), "precog")
    bullettime = broadcast_trail_flag(bullettime, len(x), "bullettime")

    # antialias (see `plot`'s `antialias=`): DRAW-TIME line smoothing for
    # ANIMATIONS. Each line-styled dataset gets a dense, PCHIP-upsampled copy
    # built ONCE here; the `update_lines_*` callbacks below then draw, for
    # whatever window of original rows a frame would have shown, exactly the
    # corresponding stretch of that smooth curve (`_aa_window`). The
    # underlying `x` rows are deliberately left untouched, so frame pacing
    # (`anim_window_bounds`), per-point labels (`_sync_anim_labels`), surface
    # hulls and marker artists all keep indexing the REAL data -- only the
    # drawn polyline is smoothed. (Static plots are antialiased upstream in
    # `plot.py`, where the densified rows also drive label/marker handling.)
    #
    # Marker-only styles are excluded: `has_line_component` is True only when
    # a linestyle token is present (solid/dashed/dotted, with or without a
    # marker), so an 'o'/'.' plot is never touched and its markers stay on the
    # true samples.
    def _fmt_at(idx):
        if isinstance(fmt, (list, tuple, np.ndarray)):
            return fmt[idx] if idx < len(fmt) else None
        return fmt

    _aa_curves = []
    for _i, _xi in enumerate(x):
        _xi = np.asarray(_xi)
        if antialias and animate and has_line_component(_fmt_at(_i)):
            _aa_curves.append(antialias_line(_xi))
        else:
            _aa_curves.append((_xi, 1))

    def _aa_window(i, a, b, artist=None):
        """The smooth polyline to DRAW for the original-row window ``x[i][a:b]``.

        With antialiasing off (or nothing to upsample) this is exactly
        ``x[i][a:b]``, so the drawn vertices are unchanged.

        When `artist` is given, the ORIGINAL row bounds are recorded on it as
        ``_hyp_row_window``. Downstream renderers that re-draw the same window
        in another form -- notably `plot._apply_multicolor_animation`, which
        re-slices a per-segment-colored collection to match -- read that
        instead of trying to recover the window from the artist's vertex
        count, which antialiasing decouples from the row count.
        """
        if artist is not None:
            artist._hyp_row_window = (a, b)
        dense, step = _aa_curves[i]
        if step == 1:
            return dense[a:b]
        if b <= a:
            return dense[0:0]
        return dense[a * step:(b - 1) * step + 1]

    # handle static plots
    def dispatch_static(x, ax=None):
        """Create (or reuse) an Axes sized for `x`'s dimensionality and draw a static plot.

        Dispatches to `plot1D`/`plot2D`/`plot3D` based on the number of
        columns in `x[0]` (creating a 3D-projection Axes when it is 3).
        """
        shape = x[0].shape[1]
        if shape == 3:
            opts = dict(projection="3d")
        else:
            opts = dict()
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, **opts)
        else:
            fig = ax.figure
        if x[0].ndim == 1 or x[0].shape[-1] == 1:
            return plot1D(x, fig, ax)
        elif x[0].shape[-1] == 2:
            return plot2D(x, fig, ax)
        elif x[0].shape[-1] == 3:
            return plot3D(x, fig, ax)

    # GH #141: marker+line combo styles (e.g. 'o-') are split into a
    # smoothed LINE artist (drawn from the already-interpolated `data[i]`,
    # marker stripped) plus a MARKERS-only artist (drawn from the raw,
    # pre-interpolation `raw_data[i]`, linestyle stripped) -- so the line
    # gets the same smoothing a pure '-' style gets, while markers stay
    # anchored to the true sample points instead of moving onto the dense
    # interpolated line. Pure line-only / marker-only styles (one artist,
    # unchanged) and `fmt=None` are untouched.
    def _plot_possibly_split(ax, coords, raw_coords, i):
        f = fmt[i] if fmt is not None else None
        ikwargs = kwargs_list[i]
        if f is None:
            ax.plot(*coords, **ikwargs)
            return
        # resolve the fmt string with matplotlib's OWN parser so a color
        # letter (the 'r' in 'ro-') is honored exactly per the fmt grammar
        # (release-1.0 audit, F01-003: split combo styles silently dropped
        # it), and so every component can be passed as an explicit kwarg --
        # passing a positional fmt ALONGSIDE a linestyle=/marker= kwarg
        # made matplotlib warn "linestyle is redundantly defined" on every
        # documented linestyle= usage (F01-009/F08-014). An explicit
        # color=/linestyle=/marker= kwarg still wins over the fmt string
        # (the historical behavior).
        if _process_plot_format is not None:
            fmt_ls, fmt_marker, fmt_color = _process_plot_format(f)
        else:  # pragma: no cover - matplotlib moved its private parser
            fmt_ls, fmt_marker = split_marker_line_fmt(f)
            fmt_color = None
        line_token, marker_char = split_marker_line_fmt(f)
        if line_token is not None and marker_char is not None:
            line_kwargs = {k: v for k, v in ikwargs.items() if k != 'marker'}
            line_kwargs.setdefault('linestyle', line_token)
            if fmt_color is not None:
                line_kwargs.setdefault('color', fmt_color)
            line_artist = ax.plot(*coords, **line_kwargs)[0]
            marker_kwargs = {k: v for k, v in ikwargs.items() if k != 'marker'}
            marker_kwargs['label'] = '_nolegend_'
            marker_kwargs['linestyle'] = 'None'
            marker_kwargs['marker'] = marker_char
            # markers share their own line's color (F01-002): without an
            # explicit color each split artist consumed one slot of the
            # palette cycle, so markers never matched their line and two
            # 'o-' datasets rendered with identical color pairs.
            marker_kwargs.setdefault('color', line_artist.get_color())
            marker_coords = raw_coords if raw_data is not None else coords
            ax.plot(*marker_coords, **marker_kwargs)
        elif _process_plot_format is not None:
            plot_kwargs = dict(ikwargs)
            if fmt_ls is not None:
                plot_kwargs.setdefault('linestyle', fmt_ls)
            if fmt_marker is not None:
                plot_kwargs.setdefault('marker', fmt_marker)
            if fmt_color is not None:
                plot_kwargs.setdefault('color', fmt_color)
            ax.plot(*coords, **plot_kwargs)
        else:  # pragma: no cover - matplotlib moved its private parser
            ax.plot(*coords, f, **ikwargs)

    # plot data in 1D
    def plot1D(data, fig, ax):
        """Draw each dataset in `data` as a 1D line/scatter (using column 0) on `ax`."""
        n = len(data)
        for i in range(n):
            raw = raw_data[i] if raw_data is not None else data[i]
            _plot_possibly_split(ax, (data[i][:, 0],), (raw[:, 0],), i)
        return fig, ax, data

    # plot data in 2D
    def plot2D(data, fig, ax):
        """Draw each dataset in `data` as a 2D line/scatter (columns 0-1) on `ax`."""
        n = len(data)
        for i in range(n):
            raw = raw_data[i] if raw_data is not None else data[i]
            _plot_possibly_split(
                ax, (data[i][:, 0], data[i][:, 1]),
                (raw[:, 0], raw[:, 1]), i)
        return fig, ax, data

    # plot data in 3D
    def plot3D(data, fig, ax):
        """Draw each dataset in `data` as a 3D line/scatter (columns 0-2) on `ax`."""
        # NOTE (D4 axes-box-slicing investigation): the static path is NOT
        # subject to the animated path's defect -- only `animate_plot3D`
        # forces `ax.set_position([0, 0, 1, 1])` (full canvas), which is
        # what makes `Axes3D.apply_aspect`'s shrunk-square-viewport
        # mismatch with `clip_on=True`'s clip_box actually slice content.
        # The static path keeps matplotlib's normal (non-stretched) subplot
        # margins, so its clip_box already matches what's drawn -- verified
        # empirically (before/after pixel diff at several elev/azim
        # combinations, including a wide/flat dataset, showed the cube
        # always complete either way). Adding `set_clip_on(False)` here
        # was tried and reverted: it changed `tight_layout()`'s computed
        # axes bbox (an unclipped 3-D line's `get_window_extent()` differs
        # from a clipped one), visibly resizing the static cube for no
        # bug-fixing benefit -- not worth the regression risk for a path
        # that was never actually clipping.
        n = len(data)
        for i in range(n):
            raw = raw_data[i] if raw_data is not None else data[i]
            _plot_possibly_split(
                ax, (data[i][:, 0], data[i][:, 1], data[i][:, 2]),
                (raw[:, 0], raw[:, 1], raw[:, 2]), i)
        return fig, ax, data

    def annotate_plot(data, labels, lengths=None):
        """Create labels in 3d chart
        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            labels (list) - list of labels of shape (numPoints,1)
            lengths (list or None) - per-dataset point counts, used to record
                each label's WITHIN-dataset index so an animation can show a
                label only while its datapoint is in the current frame's window
                (QC 2026-07). `None` -> use the global index (static plots).
        Returns:
            None
        """

        global labels_and_points
        labels_and_points = []

        if lengths is not None:
            within = [j for L in lengths for j in range(int(L))]
        else:
            within = list(range(len(data)))

        if data[0].shape[-1] > 2:
            proj = ax.get_proj()

        for idx, x in enumerate(data):
            if labels[idx] is not None:
                # font (GH #205): an explicitly resolved FontProperties
                # (covering whatever non-ASCII text is being drawn) takes
                # the place of the historical `family="serif"` -- passing
                # BOTH would be ambiguous, and `fontproperties` is what
                # actually needs to carry the CJK-covering font face.
                # With NO resolved font, point labels now INHERIT the
                # rcParams font stack like every other text surface, rather
                # than forcing `family="serif"`. That hardcoded serif both
                # clashed with the sans-serif used everywhere else and, more
                # importantly, resolved through matplotlib's stock serif list
                # instead of hypertools' per-glyph fallback stack -- so a
                # label character the serif faces lacked (e.g. U+2726 '✦')
                # rendered as "tofu" even though an installed font had it
                # (maintainer font review).
                _label_font_kwargs = (
                    dict(fontproperties=font) if font is not None else {}
                )
                if data[0].shape[-1] > 2:
                    x2, y2, _ = proj3d.proj_transform(x[0], x[1], x[2], proj)
                    # `ax.annotate`, NOT `plt.annotate`: pyplot's version
                    # draws on the CURRENT axes, which is not this call's
                    # `ax` whenever the caller supplied one (`ax=`, and so
                    # every panel of a `panels=` grid) -- every label then
                    # landed on whichever axes happened to be current, all
                    # of them on the same one (GH #285). Identical to the
                    # previous call when `ax` IS the current axes, which is
                    # every plot that creates its own figure.
                    label = ax.annotate(
                        labels[idx],
                        xy=(x2, y2),
                        xytext=(-20, 20),
                        textcoords="offset points",
                        ha="right",
                        va="bottom",
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=label_alpha),
                        arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"),
                        **_label_font_kwargs,
                    )
                    label._hyp_point_idx = within[idx]
                    label._hyp_global_idx = idx
                    labels_and_points.append((label, x[0], x[1], x[2]))
                elif data[0].shape[-1] == 2:
                    x2, y2 = x[0], x[1]
                    label = ax.annotate(
                        labels[idx],
                        xy=(x2, y2),
                        xytext=(-20, 20),
                        textcoords="offset points",
                        ha="right",
                        va="bottom",
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=label_alpha),
                        arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"),
                        **_label_font_kwargs,
                    )
                    label.draggable()
                    label._hyp_point_idx = within[idx]
                    label._hyp_global_idx = idx
                    labels_and_points.append((label, x[0], x[1]))
        fig.canvas.draw()

    def update_position(e):
        """Update label positions in chart (2D or 3D)
        Args:
            e (mouse event) - event handle to update on
        Returns:
            None
        """

        # `ax.get_proj()` / `proj3d.proj_transform` only exist on 3D
        # (Axes3D) axes -- a 2D `Axes` lacks `get_proj`. `annotate_plot`
        # stores 4-tuples (label, x, y, z) for 3D vs. 3-tuples
        # (label, x, y) for 2D (mirroring its own
        # `data[0].shape[-1] > 2` branch), so branch the same way here.
        is_3d = hasattr(ax, "get_proj")
        if is_3d:
            proj = ax.get_proj()
        # repositioning the annotations needs a renderer; after plt.close()
        # (matplotlib >= 3.11 resets the canvas) or on a headless base canvas
        # there may be none -- update the coordinates but skip the
        # renderer-dependent reposition rather than crash on a missing attr.
        renderer = getattr(fig.canvas, "renderer", None)
        if renderer is None and hasattr(fig.canvas, "get_renderer"):
            try:
                renderer = fig.canvas.get_renderer()
            except Exception:
                renderer = None
        for entry in labels_and_points:
            if is_3d:
                label, x, y, z = entry
                x2, y2, _ = proj3d.proj_transform(x, y, z, proj)
            else:
                label, x2, y2 = entry
            label.xy = x2, y2
            if renderer is not None:
                label.update_positions(renderer)
            label._visible = True
        fig.canvas.draw()

    def _sync_anim_labels(num, window_frames, all_visible=False, revealed=None,
                          hide_all=False):
        """Per-animation-frame label bookkeeping (QC 2026-07): show each
        per-point label ONLY while its datapoint is currently drawn (previously
        every label was drawn on every frame), and reproject the visible ones
        for the (possibly rotated) camera. The visibility rule depends on the
        animation style:

        * window / parallel: the datapoint is inside the head window
          ``[num - window_frames, num]`` (matched on ``_hyp_point_idx``, the
          within-dataset index, so multi-dataset plots window correctly);
        * serial: the datapoint has been REVEALED, i.e. its global index
          (``_hyp_global_idx``) ``<= revealed`` (serial accumulates points, so
          there is no trailing edge);
        * spin (``all_visible=True``): every point is always drawn;
        * morph (``hide_all=True``): the single traveling cloud does not
          correspond to the original labeled points, so labels are hidden for
          the duration of the morph.
        """
        # `labels_and_points` is a module global set only when annotate_plot ran
        # (i.e. this plot has labels); it may also still hold a PREVIOUS plot's
        # labels, so guard for existence and restrict to THIS axes' labels.
        laps = [e for e in (globals().get('labels_and_points') or [])
                if getattr(e[0], 'axes', None) is ax]
        if not laps:
            return
        is_3d = hasattr(ax, "get_proj")
        proj = ax.get_proj() if is_3d else None
        renderer = getattr(fig.canvas, "renderer", None)
        if renderer is None and hasattr(fig.canvas, "get_renderer"):
            try:
                renderer = fig.canvas.get_renderer()
            except Exception:
                renderer = None
        lo = num - window_frames
        for entry in laps:
            label = entry[0]
            if hide_all:
                visible = False
            elif all_visible:
                visible = True
            elif revealed is not None:
                g = getattr(label, "_hyp_global_idx", None)
                visible = g is None or g <= revealed
            else:
                j = getattr(label, "_hyp_point_idx", None)
                visible = j is None or (lo <= j <= num)
            label.set_visible(visible)
            if visible and is_3d:
                x2, y2, _ = proj3d.proj_transform(entry[1], entry[2], entry[3],
                                                  proj)
                label.xy = (x2, y2)
                if renderer is not None:
                    label.update_positions(renderer)

    def hide_labels(e):
        """Hides labels on button press
        Args:
            e (mouse event) - event handle to update on
        Returns:
            None
        """

        for label in labels_and_points:
            label[0]._visible = False

    def add_labels(x, labels, explore=False):
        """Add labels to graph if available
        Args:
            data (np.ndarray) - Array containing the data points
            labels (list) - List containing labels
        Returns:
            None
        """
        # if explore mode is activated, implement the on hover behavior
        if explore:
            X = np.vstack(x)
            if labels is not None:
                if any(isinstance(el, list) for el in labels):
                    labels = list(itertools.chain(*labels))
                fig.canvas.mpl_connect(
                    "motion_notify_event", lambda event: onMouseMotion(event, X, labels)
                )  # on mouse motion
            else:
                fig.canvas.mpl_connect(
                    "motion_notify_event", lambda event: onMouseMotion(event, X)
                )  # on mouse motion

        elif labels is not None:
            X = np.vstack(x)
            lengths = [np.atleast_2d(np.asarray(d)).shape[0] for d in x]
            if any(isinstance(el, list) for el in labels):
                labels = list(itertools.chain(*labels))
            annotate_plot(X, labels, lengths=lengths)
            fig.canvas.mpl_connect("button_press_event", hide_labels)
            fig.canvas.mpl_connect("button_release_event", update_position)

    ##EXPLORE MODE##
    def distance(point, event):
        """Return distance between mouse position and given data point

        Args:
            point (np.array) -  np.array of shape (3,), with x,y,z in data coords
            event (MouseEvent) - mouse event (which contains mouse position in .x and .xdata)
        Returns:
            distance (np.float64) - distance (in screen coords) between mouse pos and data point
        """
        assert point.shape == (3,), (
            "distance: point.shape is wrong: %s, must be (3,)" % point.shape
        )

        # Project 3d data space to 2d data space
        x2, y2, _ = proj3d.proj_transform(
            point[0], point[1], point[2], ax.get_proj()
        )
        # Convert 2d data space to 2d screen space
        x3, y3 = ax.transData.transform((x2, y2))

        return np.sqrt((x3 - event.x) ** 2 + (y3 - event.y) ** 2)

    def calcClosestDatapoint(X, event):
        """ "Calculate which data point is closest to the mouse position.

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            event (MouseEvent) - mouse event (containing mouse position)
        Returns:
            smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
        """

        distances = [distance(X[i, 0:3], event) for i in range(X.shape[0])]
        return np.argmin(distances)

    def annotate_plot_explore(X, index, labels=False):
        """Create popover label in 3d chart

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            index (int) - index (into points array X) of item which should be printed
            labels (list or False) - list of data point labels (default is False)
        Returns:
            None
        """

        # save clicked points
        if not hasattr(annotate_plot_explore, "clicked"):
            annotate_plot_explore.clicked = []

        # If we have previously displayed another label, remove it first
        if hasattr(annotate_plot_explore, "label"):
            if index not in annotate_plot_explore.clicked:
                annotate_plot_explore.label.remove()

        # Get data point from array of points X, at position index
        x2, y2, _ = proj3d.proj_transform(
            X[index, 0], X[index, 1], X[index, 2], ax.get_proj()
        )

        if isinstance(labels, list):
            label = labels[index]
        else:
            label = (
                "Index "
                + str(index)
                + ": ("
                + "{0:.2f}, ".format(X[index, 0])
                + "{0:.2f}, ".format(X[index, 1])
                + "{0:.2f}".format(X[index, 2])
                + ")"
            )

        _explore_font_kwargs = {} if font is None else dict(fontproperties=font)
        annotate_plot_explore.label = ax.annotate(
            label,
            xy=(x2, y2),
            xytext=(-20, 20),
            textcoords="offset points",
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            **_explore_font_kwargs,
        )
        fig.canvas.draw()

    def onMouseMotion(event, X, labels=False):
        """Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse
        Args:
            event (event) - event triggered when the mous is moved
            X (np.ndarray) - coordinates by datapoints matrix
            labels (list or False) - list of data labels (default is False)
        Returns:
            None
        """

        closestIndex = calcClosestDatapoint(X, event)

        if hasattr(onMouseMotion, "first"):
            pass
        else:
            onMouseMotion.first = False
            onMouseMotion.closestIndex_prev = calcClosestDatapoint(X, event)

        if closestIndex != onMouseMotion.closestIndex_prev:
            if isinstance(labels, list):
                annotate_plot_explore(X, closestIndex, labels)
            else:
                annotate_plot_explore(X, closestIndex)
            # update the FUNCTION ATTRIBUTE (the state actually read
            # above) -- this used to assign a dead local, so the
            # previous-index tracking never advanced past the first
            # closest point (X6-code-org-plot-007)
            onMouseMotion.closestIndex_prev = closestIndex

    def plot_cube(scale, **cube_kwargs):
        """Draw a wireframe cube of half-width `scale` (centered at the origin) on `ax`.

        Draws all six faces via `ax.plot_wireframe` with clipping
        disabled (so wide/flat rotated projections aren't sliced by
        matplotlib's shrunk-viewport aspect handling).

        Parameters
        ----------
        scale : float
            Half-width of the cube.
        **cube_kwargs
            Passed to `ax.plot_wireframe` (with sensible defaults for
            `color`/`linewidth`/`rstride`/`cstride` filled in if absent).

        Returns
        -------
        list
            The six `Line3DCollection` wireframe artists (one per face).
        """
        if cube_kwargs.get('colors') is None:
            cube_kwargs.setdefault("color", "black")
        if cube_kwargs.get('linewidths') is None:
            cube_kwargs.setdefault("linewidth", 1)
        cube_kwargs.setdefault("rstride", 1)
        cube_kwargs.setdefault("cstride", 1)

        cube = {
            "top": ([[-1, 1], [-1, 1]], [[-1, -1], [1, 1]], [[1, 1], [1, 1]]),
            "bottom": ([[-1, 1], [-1, 1]], [[-1, -1], [1, 1]], [[-1, -1], [-1, -1]]),
            "left": ([[-1, -1], [-1, -1]], [[-1, 1], [-1, 1]], [[-1, -1], [1, 1]]),
            "right": ([[1, 1], [1, 1]], [[-1, 1], [-1, 1]], [[-1, -1], [1, 1]]),
            "front": ([[-1, 1], [-1, 1]], [[-1, -1], [-1, -1]], [[-1, -1], [1, 1]]),
            "back": ([[-1, 1], [-1, 1]], [[1, 1], [1, 1]], [[-1, -1], [1, 1]]),
        }

        plane_list = []
        for side in cube:
            (Xs, Ys, Zs) = (
                np.asarray(cube[side][0]) * scale,
                np.asarray(cube[side][1]) * scale,
                np.asarray(cube[side][2]) * scale,
            )
            wf = ax.plot_wireframe(Xs, Ys, Zs, **cube_kwargs)
            # axes-box slicing fix: Axes3D's aspect machinery shrinks the
            # effective viewport (`ax.get_position()`) to a centered square
            # at draw time regardless of `ax.set_position([0, 0, 1, 1])`
            # (see the comment above that call in `animate_plot3D`) --
            # matplotlib's default `clip_on=True` then clips this wireframe
            # to that narrower square, slicing the cube whenever its
            # projection is wider than tall (e.g. at some rotation angles
            # for animated plots, or wide/flat data). Disabling clipping
            # lets it draw across the whole (already properly zoom/limit-
            # sized) canvas instead.
            wf.set_clip_on(False)
            plane_list.append(wf)
        return plane_list

    def plot_square(ax, scale=1, **square_kwargs):
        """Draw a square outline of half-width `scale` (centered at the origin) on `ax`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw the square patch on.
        scale : float, optional
            Half-width of the square (default: 1).
        **square_kwargs
            Passed to `matplotlib.patches.Rectangle`, with `edgecolor`/
            `fill`/`linewidth` defaulted (respecting matplotlib's usual
            abbreviated-argument precedence) if not already given.
        """
        # follow default matplotlib behaviors of giving abbreviated
        # arguments priority of full arguments, and `color` priority
        # over `facecolor` and `edgecolor`
        if square_kwargs.get('color') is None:
            if square_kwargs.get('ec') is None:
                square_kwargs.setdefault("edgecolor", "black")
            if (
                    square_kwargs.get('fc') is None and
                    square_kwargs.get('facecolor') is None
            ):
                square_kwargs.setdefault("fill", False)
        if square_kwargs.get("lw") is None:
            square_kwargs.setdefault("linewidth", 1)

        ax.add_patch(
            patches.Rectangle(
                scale * [-1, -1],
                scale * 2,
                scale * 2,
                **square_kwargs
            )
        )

    def frame_2d(ax):
        """Draw the 2-D frame and set the 2-D axis limits for `axis_scale`.

        ``'unit'`` (the default, and everything drawn before GH #285) draws
        hypertools' frame square and pins both axes to ``(-1.1, 1.1)``,
        because `plot()` has already rescaled the data into ``[-1, 1]``.
        ``'data'`` draws no square and applies `xlim`/`ylim` when `plot()`
        computed (or the caller passed) them, leaving matplotlib's autoscale
        in charge otherwise. Called by BOTH the static 2-D path and
        `animate_plot2D`, so the two cannot drift apart.
        """
        if axis_scale != 'data':
            plot_square(ax, **frame_kwargs)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            return
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    def update_lines_parallel(
        num,
        data_lines,
        lines,
        trail_lines,
        cube_scale,
        tail_duration=2,
        rotations=1,
        zoom=1,
        chemtrails=None,
        precog=None,
        bullettime=None,
        elev=10,
    ):
        """FuncAnimation frame-update callback: redraw the cube and every dataset's 3D trail/head for frame `num`.

        Removes and redraws the frame cube, rotates the view (`ax.view_init`)
        by `azim = rotations * 360 * num / n_frames`, applies the zoom via
        `ax.set_box_aspect`, then updates each dataset's line/trail
        artists in lockstep ("parallel" animation, as opposed to the
        serial/morph variants) according to its chemtrails/precog/
        bullettime flags and `tail_duration`.

        Returns
        -------
        tuple of (list, list)
            `(lines, trail_lines)` -- the updated head-line and trail
            artists, for `blit=True` animation.
        """
        if hasattr(update_lines_parallel, "planes"):
            for plane in update_lines_parallel.planes:
                plane.remove()

        update_lines_parallel.planes = plot_cube(cube_scale, **frame_kwargs)
        # camera: honor the user's azim= as the starting angle (F05-003 --
        # previously parallel/window always started at azimuth 0, unlike
        # 'serial'/'morph'/static plots) and pace the rotation over the
        # FRAME count rather than the first dataset's row count (the two
        # are no longer interchangeable for marker-only datasets, which
        # keep their raw rows -- see `anim_window_bounds`).
        total_frames = max(1, int(round(frame_rate * duration)))
        # ONE clock per source dataset: `hue=`/`cluster=` runs of the same
        # dataset must reveal in row order, not all at once (see
        # `trails.dataset_window_bounds`). Without regrouping this returns
        # exactly what `anim_window_bounds` returned before, frame for frame.
        _windows = None
        if ownership is not None:
            _windows = dataset_window_bounds(
                num, total_frames, ownership,
                [d.shape[0] for d in data_lines], tail_duration)
        azim_now = azim + rotations * (360 * (num / total_frames))
        ax.view_init(elev=elev, azim=azim_now)
        # Axes3D.dist was removed in matplotlib >= 3.8, silently disabling
        # zoom; set_box_aspect(zoom=...) is the supported equivalent. See
        # _anim_box_zoom for the (slightly zoomed-out) animation mapping.
        ax.set_box_aspect(None, zoom=_anim_box_zoom(zoom))

        # zip_longest: marker-only animations (or datasets with no trail
        # flag set at all -- GH #127) have no trail artist (trail is None
        # for those datasets), but head artists still animate. chemtrails/
        # precog/bullettime are per-dataset lists (broadcast/validated in
        # `_draw` above), indexed by `i` -- the SAME semantics as before
        # (bullettime, or chemtrails+precog together, show the full trail;
        # chemtrails alone shows the past window; precog alone shows the
        # future window), just resolved per dataset now instead of once
        # globally.
        windows = []
        window_spcs = []
        # GH #285: the drawn head window per run, published as
        # `FrameContext.revealed_counts` / `.window_bounds` below. Collected
        # in the same loop that slices the artists, so what the context
        # reports and what the frame draws cannot drift apart.
        head_bounds = []
        for i, (line, data, trail) in enumerate(itertools.zip_longest(
                lines, data_lines, trail_lines)):

            # head/trail slicing (release-1.0 audit): every dataset is paced
            # onto the shared frame grid -- see `anim_window_bounds` for the
            # F05-001 (negative chemtrails slice), F05-008 (precog gap),
            # F04-003/F05-012 (shorter/1-point datasets vanishing or driving
            # the frame count) fixes it encodes.
            if _windows is not None:
                win = _windows[i]
            else:
                _s, _e, _ts = anim_window_bounds(
                    num, total_frames, data.shape[0], tail_duration)
                win = RunWindow(_s, _e, _ts, max(0, _e - 1), True,
                                data.shape[0])
            start, end = win.head_start, win.head_end
            head_bounds.append((start, end))

            # antialias: each artist draws the SMOOTH curve spanning the same
            # rows it would otherwise have drawn raw (`_aa_window`).
            n_rows = data.shape[0]
            if trail is not None:
                ct, pc, bt = chemtrails[i], precog[i], bullettime[i]
                trail_seg = None
                if (pc and ct) or bt:
                    trail_seg = _aa_window(i, 0, n_rows, artist=trail)
                elif ct:
                    trail_seg = _aa_window(i, 0, win.past_stop, artist=trail)
                elif pc:
                    # `win.future_start`, never `end - 1`: a run the dataset's
                    # clock has not reached has `end == 0`, and `data[-1:]`
                    # would put one point of a future category on screen.
                    trail_seg = _aa_window(i, win.future_start, n_rows,
                                           artist=trail)
                if trail_seg is not None:
                    trail.set_data(trail_seg[:, 0:2].T)
                    trail.set_3d_properties(trail_seg[:, 2])

            window = data[start:end]            # RAW rows: hull/point colors
            draw_window = _aa_window(i, start, end, artist=line)  # drawn curve
            line.set_data(draw_window[:, 0:2].T)
            line.set_3d_properties(draw_window[:, 2])
            windows.append(window)
            window_spcs.append(_window_surface_point_colors(
                surface_point_colors, i, start, end))

        # surface= (GH #109): recompute each dataset's hull from its CURRENT
        # visible window (same window as the head line above) and the
        # current camera view (backface culling depends on it), keeping the
        # per-vertex hue coloring of the (identically-windowed) points
        # (F07-005)
        if surface is not None:
            prior = getattr(update_lines_parallel, "surface_colls", None)
            update_lines_parallel.surface_colls = _mesh_and_draw_3d(
                ax, windows, surface, surface_colors, elev, azim_now,
                prior_colls=prior, quiet=True,
                surface_point_colors=window_spcs)

        # per-point labels track their datapoint's visibility window (the same
        # [num - tail_duration, num] window the head line uses above)
        _sync_anim_labels(num, tail_duration)
        if frame_hooks is not None:
            frame_hooks.record(
                frame=int(num), n_frames=int(total_frames),
                artists=list(lines) + [t for t in trail_lines if t is not None],
                datasets=list(data_lines), style=animate, order='parallel',
                current_index=None, current_fraction=None,
                revealed_counts=tuple(e for _, e in head_bounds),
                window_bounds=tuple(head_bounds))
        return lines, trail_lines

    def update_lines_spin(
        num, data_lines, lines, cube_scale, rotations=1, zoom=1, elev=10
    ):
        """FuncAnimation frame-update callback: rotate the camera around fully-drawn ('spin') datasets for frame `num`.

        Unlike `update_lines_parallel`, every dataset's FULL trajectory
        is already drawn for every frame -- only the cube, camera
        rotation, and (when `surface=` is set) shading/backface-culling
        update per frame.

        Returns
        -------
        list
            The updated matplotlib line artists, for `blit=True` animation.
        """
        if hasattr(update_lines_spin, "planes"):
            for plane in update_lines_spin.planes:
                plane.remove()

        update_lines_spin.planes = plot_cube(cube_scale, **frame_kwargs)
        # honor the user's azim= as the starting camera angle (F05-003:
        # 'spin' previously always started at azimuth 0, so azim=45 was
        # silently ignored and rotations=0 could not pick a viewing angle).
        # Pace the orbit over the ROUNDED frame count -- the number of frames
        # actually drawn -- exactly as `update_lines_parallel` and the plotly
        # renderer do. Spin was the ONLY path dividing by the raw
        # `frame_rate * duration` product, which differs from the drawn frame
        # count whenever that product is not a whole number: at frame_rate=7,
        # duration=2.5 (18 frames drawn, product 17.5) the last frame landed
        # at 289.71 deg here against plotly's 280.0 -- the same call, a 9.71
        # deg disagreement, and 349.71 deg of travel for a `rotations=1` turn.
        # Frames 0..N-1 are meant to span [0, 360) so a looping animation does
        # not draw the same angle twice; dividing by 17.5 overshot that.
        total_frames = max(1, int(round(frame_rate * duration)))
        azim_now = azim + rotations * (360 * (num / total_frames))
        ax.view_init(elev=elev, azim=azim_now)
        # Axes3D.dist was removed in matplotlib >= 3.8, silently disabling
        # zoom; set_box_aspect(zoom=...) is the supported equivalent. See
        # _anim_box_zoom for the (slightly zoomed-out) animation mapping.
        ax.set_box_aspect(None, zoom=_anim_box_zoom(zoom))

        for i, (line, data) in enumerate(zip(lines, data_lines)):
            # antialias: 'spin' draws the FULL trajectory every frame
            draw_data = _aa_window(i, 0, data.shape[0], artist=line)
            line.set_data(draw_data[:, 0:2].T)
            line.set_3d_properties(draw_data[:, 2])

        # surface= (GH #109): the FULL dataset is static in 'spin' mode
        # (only the camera rotates), so the mesh itself is precomputed once
        # (`update_lines_spin.meshes`, set in animate_plot3D before this
        # runs) -- only shading/backface-culling are recomputed per frame.
        if surface is not None:
            prior = getattr(update_lines_spin, "surface_colls", None)
            update_lines_spin.surface_colls = _shade_and_cull_3d(
                ax, update_lines_spin.meshes, surface, surface_colors, elev,
                azim_now, prior_colls=prior,
                # 'spin' draws the FULL dataset every frame, so the full
                # (unwindowed) per-point hue colors apply as-is (F07-005)
                surface_point_colors=surface_point_colors)

        # 'spin' draws every point every frame, so labels stay visible -- but
        # still reproject them for the rotated camera
        _sync_anim_labels(num, 0, all_visible=True)
        if frame_hooks is not None:
            frame_hooks.record(
                frame=int(num), n_frames=int(total_frames),
                artists=list(lines), datasets=list(data_lines), style=animate,
                order='parallel', current_index=None, current_fraction=None,
                # 'spin' draws every dataset in FULL on every frame (only
                # the camera moves), so every row is revealed from frame 0
                # -- reporting None here left a caller no way to tell that
                # apart from "this backend does not know" (GH #285).
                revealed_counts=tuple(d.shape[0] for d in data_lines),
                window_bounds=tuple((0, d.shape[0]) for d in data_lines))
        return lines

    def update_lines_serial(
        num, data_lines, lines, trail_lines, cube_scale, window_frames=1,
        rotations=1, zoom=1, chemtrails=None, precog=None, bullettime=None,
        elev=10,
    ):
        """Serial animation: datasets appear ONE AT A TIME, each growing
        point-by-point into place while all previous datasets stay fully
        drawn (e.g. conversation turns adding to a shared embedding space).
        Datasets are never connected to each other.

        Trail composition (GH #127 follow-up): when a per-dataset
        chemtrails/precog/bullettime flag is set, the ONE dataset currently
        being revealed ALSO traces out a low-opacity trail relative to its
        OWN reveal, led by a short opaque comet-head near the reveal tip --
        chemtrails fades its revealed-so-far past (``data[:shown]``), precog
        fades its not-yet-revealed future (``data[shown - 1:]``, sharing the
        head's last vertex so there is no one-segment gap, cf.
        `anim_window_bounds`' F05-008), and bullettime (or chemtrails AND
        precog together) fades the WHOLE trajectory. Already-revealed
        datasets stay fully drawn (accumulated history) and future ones stay
        invisible. With NO trail flag set for a dataset (plain 'serial'), its
        whole revealed portion is drawn fully opaque with no trail --
        byte-for-byte the historical behavior."""
        if hasattr(update_lines_serial, "planes"):
            for plane in update_lines_serial.planes:
                plane.remove()
        update_lines_serial.planes = plot_cube(cube_scale, **frame_kwargs)

        total_frames = max(1, int(round(frame_rate * duration)))
        azim_now = azim + rotations * 360.0 * num / total_frames
        ax.view_init(elev=elev, azim=azim_now)
        ax.set_box_aspect(None, zoom=_anim_box_zoom(zoom))

        lengths = [d.shape[0] for d in data_lines]
        total_points = sum(lengths)
        revealed = total_points * num / max(1, total_frames - 1)
        _counts = serial_reveal_counts(lengths, num, total_frames)

        windows = []
        window_spcs = []
        for i, (line, data, trail) in enumerate(itertools.zip_longest(
                lines, data_lines, trail_lines)):
            n_pts = data.shape[0]
            shown = _counts[i]

            ct = chemtrails[i] if chemtrails is not None else False
            pc = precog[i] if precog is not None else False
            bt = bullettime[i] if bullettime is not None else False
            # a dataset composes a trail only if it both HAS a trail artist
            # (created by `_wants_trail`) and a flag set this frame's window.
            has_trail = trail is not None and (ct or pc or bt)

            # antialias: bounds are resolved as ORIGINAL-row index pairs, then
            # `_aa_window` maps each onto the smooth curve for drawing.
            trail_bounds = None
            if not has_trail:
                # plain 'serial' (or a dataset with no trail flag): the whole
                # revealed portion is drawn fully opaque -- UNCHANGED.
                head_bounds = (0, shown)
            elif shown <= 0:
                # not started revealing yet: head + trail both empty.
                head_bounds = (0, 0)
            elif shown >= n_pts:
                # fully revealed: the whole dataset stays drawn as opaque
                # history, trail cleared.
                head_bounds = (0, n_pts)
            else:
                # currently revealing: a short opaque comet-head leads the
                # reveal tip while the rest traces out as a faded trail.
                # `window_frames` is the head length in FRAMES; scale it onto
                # this dataset's SHARE of the serial timeline
                # (`n_pts / total_points`, since the serial sweep packs every
                # dataset's rows into the same frame grid), mirroring
                # `anim_window_bounds`' start = end - 1 - w head sizing.
                w = max(1, int(round(window_frames * n_pts
                                     / max(1, total_points))))
                head_bounds = (max(0, shown - 1 - w), shown)
                if (ct and pc) or bt:
                    trail_bounds = (0, n_pts)              # bullettime: whole
                elif ct:
                    trail_bounds = (0, shown)              # chemtrails: past
                else:
                    trail_bounds = (max(0, shown - 1), n_pts)  # precog: future

            head = _aa_window(i, *head_bounds, artist=line)
            trail_seg = (data[:0] if trail_bounds is None
                         else _aa_window(i, *trail_bounds, artist=trail))
            line.set_data(head[:, 0:2].T)
            line.set_3d_properties(head[:, 2])
            if trail is not None:
                trail.set_data(trail_seg[:, 0:2].T)
                trail.set_3d_properties(trail_seg[:, 2])

            # surface hull follows the full revealed portion (same window as
            # plain serial), independent of the comet-head trimming above
            windows.append(data[:shown])
            window_spcs.append(_window_surface_point_colors(
                surface_point_colors, i, 0, shown))

        # surface= (GH #109): each dataset's hull follows its own currently-
        # revealed portion (same window as its line above), keeping the
        # per-vertex hue coloring of the revealed points (F07-005)
        if surface is not None:
            prior = getattr(update_lines_serial, "surface_colls", None)
            update_lines_serial.surface_colls = _mesh_and_draw_3d(
                ax, windows, surface, surface_colors, elev, azim_now,
                prior_colls=prior, quiet=True,
                surface_point_colors=window_spcs)

        # serial reveals points cumulatively: a label shows once its point has
        # been revealed (global index <= revealed), and stays
        _sync_anim_labels(num, 0, revealed=revealed)
        if frame_hooks is not None:
            _idx, _frac = serial_current_index(_counts, lengths)
            frame_hooks.record(
                frame=int(num), n_frames=int(total_frames),
                artists=list(lines) + [t for t in trail_lines if t is not None],
                datasets=list(data_lines), style='serial', order='serial',
                current_index=_idx, current_fraction=_frac,
                revealed_counts=_counts,
                # a serial reveal is cumulative: every dataset's window
                # starts at row 0 (GH #285).
                window_bounds=tuple((0, c) for c in _counts))
        return lines

    def update_morph(num, morph_state, cube_scale, azimuths, zoom=1, elev=10):
        """animate='morph': one traveling point-cloud artist eases through
        the Hungarian-matched hold/morph schedule (see `hypertools.plot.morph`)
        while any UNTAGGED (static) datasets stay fully drawn, untouched, in
        the background -- only the camera (and, if surfaced, this single
        artist's hull) update every frame."""
        if hasattr(update_morph, "planes"):
            for plane in update_morph.planes:
                plane.remove()
        update_morph.planes = plot_cube(cube_scale, **frame_kwargs)

        azim_now = azimuths[num]
        ax.view_init(elev=elev, azim=azim_now)
        ax.set_box_aspect(None, zoom=_anim_box_zoom(zoom))

        seg_idx, step, n_steps = _morph.frame_to_segment(
            morph_state["frame_counts"], num)
        pts = _morph.morph_positions(morph_state["sampled"], seg_idx, step,
                                     n_steps)
        color = _morph.morph_color(morph_state["colors"], seg_idx, step,
                                   n_steps)

        # full-sample morphs (maintainer request, 2026-07-06 follow-up): on
        # a HOLD frame, the held dataset's own duplicated (padding) points
        # are excluded from the DRAWN artist -- so alpha compositing (e.g.
        # semi-transparent markers) looks exactly like a plain plot of that
        # dataset's true points -- but never from `pts` itself, which stays
        # the FULL n-point cloud for hull-building below (duplicates are
        # exact copies of existing points and so never change a convex
        # hull's shape). On a MORPH frame, nothing is hidden: every point,
        # including both endpoints' duplicates, is shown while traveling.
        hide = _morph.morph_visible_mask(morph_state.get("dup_masks"),
                                         seg_idx)
        draw_pts = pts[~hide] if hide is not None else pts

        artist = morph_state["artist"]
        artist.set_data(draw_pts[:, 0], draw_pts[:, 1])
        artist.set_3d_properties(draw_pts[:, 2])
        artist.set_color(color)
        # GH #284: `alpha=` follows the same hold/morph schedule as color;
        # `None` (no alpha given) leaves the artist's default untouched.
        alpha = _morph.morph_alpha(morph_state.get("alphas"), seg_idx,
                                   step, n_steps)
        if alpha is not None:
            artist.set_alpha(alpha)

        # surface= (GH #109/morph): the traveling cloud's own hull (if it
        # has a surface spec) is rebuilt from its CURRENT interpolated
        # positions every frame (like 'parallel'/'serial'); any static
        # (untagged) dataset's surface was already precomputed once
        # (`morph_state['static_meshes']`, set in animate_plot3D) and is
        # only re-shaded/re-culled here every frame, exactly like 'spin' --
        # this runs whenever ANY dataset requested a surface, not only when
        # the morphing cloud itself has one (a static dataset's surface
        # must keep rendering even if no morph-tagged dataset has a spec).
        if surface is not None:
            frame_meshes = list(morph_state["static_meshes"])
            frame_colors = surface_colors
            if morph_state["surface_spec"] is not None:
                mesh = (build_mesh_3d(pts, morph_state["surface_spec"],
                                      dataset_label=" morph", quiet=True)
                       if pts.shape[0] >= 4 else None)
                frame_meshes[morph_state["mesh_slot"]] = mesh
                frame_colors = list(surface_colors)
                frame_colors[morph_state["mesh_slot"]] = color
            # per-point hue colors apply to the STATIC (untagged) datasets'
            # surfaces only (F07-005) -- the traveling morph cloud's own
            # hull keeps its single interpolated `color` (there is no
            # per-point hue correspondence mid-morph), so its slot (and
            # every morph-tagged slot) is forced to None here.
            frame_spcs = None
            if surface_point_colors:
                frame_spcs = list(surface_point_colors)
                for mi in morph_state["indices"]:
                    if mi < len(frame_spcs):
                        frame_spcs[mi] = None
            prior = getattr(update_morph, "surface_colls", None)
            update_morph.surface_colls = _shade_and_cull_3d(
                ax, frame_meshes, surface, frame_colors, elev, azim_now,
                prior_colls=prior, surface_point_colors=frame_spcs)

        # morph collapses the datasets to one traveling cloud that does not
        # correspond to the original labeled points -> hide per-point labels
        _sync_anim_labels(num, 0, hide_all=True)
        if frame_hooks is not None:
            frame_hooks.record(
                frame=int(num),
                n_frames=int(sum(morph_state["frame_counts"])),
                artists=[morph_state["artist"]],
                datasets=list(morph_state["sampled"]),
                style='morph', order='serial',
                # `seg_idx // 2` is a position WITHIN THE MORPH SEQUENCE
                # (0, 1, 2, ... for the 1st, 2nd, 3rd morph-tagged dataset),
                # not a FINAL dataset index -- those only coincide when
                # every dataset is tagged (scalar animate='morph'). For a
                # partial-tag list (e.g. animate=[None, 'morph', 'morph']),
                # `morph_state["indices"]` (built from `morph_tags` in
                # `animate_plot3D` above) maps sequence position back to the
                # actual dataset index, exactly like the simplify guard in
                # `plot.py` already does -- so this agrees with `title=`'s
                # per-segment lookup, which indexes by FINAL dataset.
                current_index=morph_state["indices"][seg_idx // 2],
                current_fraction=step / max(1, n_steps - 1),
                revealed_counts=None,
                segment_index=seg_idx,
                segment_kind='hold' if seg_idx % 2 == 0 else 'transition')
        return (artist,)

    def dispatch_animate(x, ani_params):
        """Dispatch to `animate_plot3D` or `animate_plot2D` based on `x[0]`'s column count.

        Parameters
        ----------
        x : list of numpy.ndarray
            Datasets to animate.
        ani_params : dict
            Keyword arguments forwarded to the resolved animate function.

        Returns
        -------
        The resolved animate function's return value (e.g. `(fig, ax, x,
        line_ani)`), or `None` if `x[0]` has neither 2 nor 3 columns.
        """
        if x[0].shape[1] == 3:
            return animate_plot3D(x, **ani_params)
        if x[0].shape[1] == 2:
            return animate_plot2D(x, **ani_params)

    def animate_plot3D(
        x,
        tail_duration=2,
        focused=None,
        rotations=1,
        zoom=1,
        chemtrails=None,
        precog=None,
        bullettime=None,
        frame_rate=30,
        elev=10,
        style="parallel",
        morph_tags=None,
        morph_colors=None,
        morph_samples=None,
        morph_loop=False,
    ):
        """Build and run a 3D matplotlib `FuncAnimation` for `x` (parallel/spin/serial/morph/window styles).

        Creates the 3D axes and initial (single-point) line/trail
        artists for every dataset, then wires up the appropriate
        per-frame update callback (`update_lines_parallel`,
        `update_lines_spin`, `update_lines_serial`, or `update_morph`,
        selected by `style`) as a `matplotlib.animation.FuncAnimation`.

        Parameters
        ----------
        x : list of numpy.ndarray
            3-column datasets to animate.
        tail_duration : int, optional
            Number of trailing frames shown for chemtrails/precog/
            bullettime trails (default: 2).
        focused : optional
            Index (or indices) of the dataset(s) to keep visually
            emphasized; forwarded to the frame-window logic.
        rotations : float, optional
            Number of full camera rotations over the animation (default: 1).
        zoom : float, optional
            Camera zoom factor (default: 1).
        chemtrails, precog, bullettime : list of bool, optional
            Per-dataset trail-display flags (see `update_lines_parallel`).
        frame_rate : int, optional
            Animation frame rate in fps (default: 30).
        elev : float, optional
            Camera elevation angle in degrees (default: 10).
        style : {'parallel', 'spin', 'serial', 'morph', 'window'}, optional
            Which animation style/update-callback to use (default: 'parallel').
        morph_loop : bool, optional
            ``loop=True`` on `plot`: close the morph sequence by returning
            to the FIRST cloud, reusing its sampled points (GH #285).
        morph_tags, morph_colors, morph_samples : optional
            Parameters controlling the `style='morph'` traveling
            point-cloud animation (see `hypertools.plot.morph`).

        Returns
        -------
        tuple
            `(fig, ax, x, line_ani)` -- the created Figure, Axes3D,
            original data `x`, and the `FuncAnimation` instance.
        """

        # initialize plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # give the (axis-off) 3D axes the full canvas: with the default
        # subplot margins, the zoomed cube overflows the axes viewport at
        # some rotation angles, clipping cube corners and data
        ax.set_position([0.0, 0.0, 1.0, 1.0])

        # create lines. Trail artists are created whenever ANY dataset draws
        # lines (previously is_line(fmt) required ALL datasets to be lines,
        # which left `trail` undefined -- and the FuncAnimation call below
        # crashed -- when fmts were mixed, e.g. after a single-point dataset's
        # line fmt is converted to '.', or for marker-only animations).
        #
        # GH #127: chemtrails/precog/bullettime are now per-dataset lists
        # (broadcast/validated in `_draw` above), so a dataset only gets its
        # OWN trail artist -- `trail[idx]` is `None` for any dataset with
        # none of the three flags set (rather than an inert stub artist
        # created for every dataset whenever ANY flag was set anywhere).
        # `trail_lines` in `update_lines_parallel` tolerates `None` entries
        # via `itertools.zip_longest` already (marker-only animations relied
        # on this same mechanism before this change).
        #
        # 'spin' (GH #127 follow-up): `update_lines_spin` never accepts (or
        # touches) a trail_lines argument -- 'spin' has no "current position"
        # for a trail to lead/follow (only the camera moves). A trail artist
        # created here for 'spin' would stay frozen at its initial (single-
        # point) state for the whole animation: an invisible/useless stub.
        # 'morph' draws a single traveling cloud (no per-dataset current
        # position) and 'window' is bullettime MINUS its trail by definition,
        # so both likewise skip trails. `plot.py` already warns the caller
        # (and names the ignored flags/dataset indices) for these modes.
        #
        # 'serial' now COMPOSES with the trail flags (chemtrails-serial /
        # precog-serial / bullettime-serial): `update_lines_serial` draws the
        # currently-revealing dataset with a per-dataset trail, so its trail
        # artists ARE created here whenever a flag is set for that dataset.
        def _wants_trail(idx):
            if style in ("spin", "morph", "window"):
                return False
            return chemtrails[idx] or precog[idx] or bullettime[idx]

        # pop linewidth ONCE per dataset (it must not also ride along in
        # **kwargs_list[idx]) and share it between each head line and its
        # trail -- popping inside each comprehension left nothing for the
        # trail's pop, so trails silently ignored the user's linewidth=
        # (X6-code-org-plot-009)
        linewidths = [
            kwargs_list[idx].pop("linewidth", 1)
            if isinstance(kwargs_list[idx], dict) else 1
            for idx in range(len(x))
        ]

        # fold the 0.3 trail-fade factor into whatever alpha the dataset kwargs
        # already carry (default 1.0 -> 0.3, i.e. unchanged for the common
        # no-alpha case). Passing a bare `alpha=0.3` alongside **kwargs_list[idx]
        # collided when MultiIndex expansion assigned a per-trace alpha (faint
        # leaf traces vs. opaque group-mean traces), raising "got multiple
        # values for keyword argument 'alpha'".
        def _trail_kwargs(kw):
            kw = dict(kw) if isinstance(kw, dict) else {}
            kw["alpha"] = 0.3 * kw.pop("alpha", 1.0)
            return kw

        trail = []
        if fmt is not None:
            lines = [
                ax.plot(
                    dat[0:1, 0],
                    dat[0:1, 1],
                    dat[0:1, 2],
                    fmt[idx],
                    linewidth=linewidths[idx],
                    **kwargs_list[idx]
                )[0]
                for idx, dat in enumerate(x)
            ]
            if any(is_line(f) for f in fmt):
                trail = [
                    ax.plot(
                        dat[0:1, 0],
                        dat[0:1, 1],
                        dat[0:1, 2],
                        fmt[idx],
                        linewidth=linewidths[idx],
                        **_trail_kwargs(kwargs_list[idx])
                    )[0] if _wants_trail(idx) else None
                    for idx, dat in enumerate(x)
                ]
        else:
            lines = [
                ax.plot(
                    dat[0:1, 0],
                    dat[0:1, 1],
                    dat[0:1, 2],
                    linewidth=linewidths[idx],
                    **kwargs_list[idx]
                )[0]
                for idx, dat in enumerate(x)
            ]
            if is_line(fmt):
                trail = [
                    ax.plot(
                        dat[0:1, 0],
                        dat[0:1, 1],
                        dat[0:1, 2],
                        linewidth=linewidths[idx],
                        **_trail_kwargs(kwargs_list[idx])
                    )[0] if _wants_trail(idx) else None
                    for idx, dat in enumerate(x)
                ]
        # trails are faint context, not legend-worthy: only the in-focus
        # `lines` should carry legend entries. Otherwise every label appears
        # twice -- once for the moving window, once for its tail. The legend is
        # built once from `lines` (all datasets, created upfront), so it shows
        # the static union of in-focus items and never changes across frames.
        for _trail_line in trail:
            if _trail_line is not None:
                _trail_line.set_label('_nolegend_')

        # axes-box slicing fix (see `plot_cube`): unclip every data/trail
        # line artist -- Axes3D's shrunk-square viewport (`ax.get_position()`
        # after `apply_aspect()` ignores the full-canvas `ax.set_position`
        # call above) otherwise clips these lines exactly like the cube
        # wireframe, at any rotation angle where the projected scene is
        # wider than tall (the maintainer-reported "cut off right side"
        # symptom, most visible with wide/flat trajectories like chemtrails).
        for _artist in itertools.chain(lines, trail):
            if _artist is not None:
                _artist.set_clip_on(False)

        # animate='morph' (Hungarian-matched point-cloud morphs, maintainer
        # request): build the single traveling artist here (before the
        # surface/cube-scale block below, which needs `morph_state` set to
        # wire in the per-frame hull) -- the morph-tagged datasets' own
        # `lines`/`trail` artists (created above, for legend bookkeeping)
        # are hidden for the whole animation since only the ONE shared
        # artist is ever actually drawn/moved.
        morph_state = None
        if style == "morph":
            _tags = morph_tags if morph_tags is not None else [True] * len(x)
            morph_indices = [i for i, tag in enumerate(_tags) if tag]
            clouds = [np.asarray(x[i], dtype=np.float64)[:, :3]
                     for i in morph_indices]
            sampled, dup_masks = _morph.sample_and_match_clouds(
                clouds, morph_samples=morph_samples, loop=morph_loop)
            if morph_loop:
                # `loop=True` returns ONE more cloud than it was given --
                # the closing repeat of cloud 0 (GH #285). Extend the
                # sequence-position -> dataset-index map to match, so the
                # closing hold reports (and titles itself as) dataset
                # `morph_indices[0]`, and every downstream consumer
                # (`ds_colors`, `n_morph_datasets`, `frame_counts`) counts
                # the extra segment pair.
                morph_indices = morph_indices + [morph_indices[0]]
            ds_colors = [
                tuple(morph_colors[i]) if morph_colors is not None
                else (0.2, 0.4, 0.8)
                for i in morph_indices
            ]

            for i in morph_indices:
                lines[i].set_visible(False)
                if i < len(trail) and trail[i] is not None:
                    trail[i].set_visible(False)

            # M4 visual-review fix: any UNTAGGED (static backdrop) dataset
            # is drawn once, in full, right here -- `update_morph` (below)
            # only ever moves the single traveling `morph_artist` and never
            # touches `lines`/`trail`, so an untagged line left at its
            # initial `dat[0:1, ...]` (a single point, same as every other
            # style's pre-animation initialization) would silently stay a
            # 1-point "cloud" for the WHOLE animation -- i.e. never
            # actually render. Mirrors the plotly backend, where an
            # untagged dataset's trace is simply never referenced by any
            # frame's `traces=` list, so it keeps whatever it was drawn
            # with up front (its full data).
            for i in range(len(x)):
                if i in morph_indices:
                    continue
                full = x[i]
                lines[i].set_data(full[:, 0:2].T)
                lines[i].set_3d_properties(full[:, 2])

            mesh_slot = morph_indices[0]
            morph_surface_spec = None
            if surface is not None:
                for i in morph_indices:
                    if i < len(surface) and surface[i] is not None:
                        morph_surface_spec = surface[i]
                        break

            # full-sample morphs (maintainer request, 2026-07-06 follow-up):
            # dataset 0's own duplicated rows (padding it up to the target
            # `n`) are hidden at this initial hold-frame-0 draw, exactly
            # like every other hold frame -- see `update_morph` below and
            # `hypertools.plot.morph.morph_visible_mask`.
            first_pts = sampled[0]
            first_hide = _morph.morph_visible_mask(dup_masks, 0)
            first_draw = (first_pts[~first_hide] if first_hide is not None
                         else first_pts)
            _mkw = (kwargs_list[mesh_slot]
                   if isinstance(kwargs_list[mesh_slot], dict) else {})
            morph_markersize = _mkw.get("markersize") or 1.5
            # GH #284: `alpha=` (scalar, or the per-dataset list) lands in
            # each morph-tagged dataset's kwargs, but those datasets' own
            # `lines` are hidden above -- the ONE visible artist is this
            # cloud, so it takes the held/departing dataset's alpha on the
            # same hold/morph schedule as its color (`_morph.morph_alpha`).
            # `None` throughout (no alpha asked for) leaves it at the
            # matplotlib default, exactly as before.
            ds_alphas = [
                (kwargs_list[i] if isinstance(kwargs_list[i], dict)
                 else {}).get("alpha")
                for i in morph_indices
            ]
            (morph_artist,) = ax.plot(
                first_draw[:, 0], first_draw[:, 1], first_draw[:, 2],
                linestyle="None", marker=".", markersize=morph_markersize,
                color=ds_colors[0],
                alpha=_morph.morph_alpha(ds_alphas, 0, 0, 1),
            )
            morph_artist.set_label("_nolegend_")
            # axes-box slicing fix (see `plot_cube`): unclip the traveling
            # morph point-cloud artist too.
            morph_artist.set_clip_on(False)
            if (morph_surface_spec is not None
                    and not morph_surface_spec.get("keep_points", True)):
                morph_artist.set_visible(False)

            morph_state = dict(
                sampled=sampled, dup_masks=dup_masks, colors=ds_colors,
                alphas=ds_alphas,
                artist=morph_artist, mesh_slot=mesh_slot,
                surface_spec=morph_surface_spec, indices=morph_indices,
            )

        # surface= (GH #109)
        # cube_scale_anim (GH #109 round 2): the axes cube/limits must be
        # wide enough to contain every surface mesh built over the COURSE
        # of the animation, not just the initial [-1, 1] data cube. Per-
        # frame windows (parallel/serial) vary in size, so recomputing an
        # exact per-frame bound would mean rebuilding every dataset's mesh
        # a second time on every frame just to measure it (expensive: a
        # full smooth_hull_3d call, not a cheap lookup). Instead this uses
        # the FULL-DATA mesh's extent as the (fixed, precomputed-once)
        # bound for the whole animation -- cheap, and exactly what 'spin'
        # already needed anyway (its mesh is static for the whole
        # animation). This can, in principle, under-cover a transient,
        # very small early window (few points need proportionally more
        # `_rescale_for_containment` growth than the full dataset does),
        # which is an accepted tradeoff for not rebuilding meshes twice
        # per frame.
        cube_scale_anim = 1
        if surface is not None:
            # keep_points=False: hide (not remove) that dataset's line/trail
            # for the whole animation -- visibility is set once here and
            # persists across every frame update.
            _hide_no_keep_points(lines, surface)
            _hide_no_keep_points(trail, surface)
            # M4 fix (maintainer review, surface=True + animate='morph' on
            # large clouds): a morph-tagged dataset NEVER gets a drawn
            # static mesh -- `update_morph` rebuilds the single traveling
            # hull every frame from the CURRENT (sampled/interpolated)
            # positions (see `static_meshes` below), and its box-sizing
            # bound comes entirely from the M3b sampled+union meshes built
            # below. Building a full-cloud `build_mesh_3d` mesh here for a
            # morph-tagged dataset was therefore ALWAYS pure waste (never
            # drawn, never used once the sampled+union bound existed) --
            # and worse, a correctness/performance cliff on large clouds:
            # `smooth_hull_3d` (its `ConvexHull`/Taubin pipeline, and
            # especially its `points_enclosed` containment check, a
            # `Delaunay` build/query over the input points) scales with the
            # FULL point count, not the usually-much-smaller `morph_samples`
            # cap, so a ~20k-30k point raw cloud could make this one-time
            # sizing call itself slow or memory-heavy. `surface_for_full`
            # nulls out every morph-tagged index so `_build_mesh_list` skips
            # them entirely; static (untagged) datasets are unaffected and
            # still get their normal full-cloud mesh.
            if style == "morph" and morph_state is not None:
                surface_for_full = [
                    None if i in morph_state["indices"] else s
                    for i, s in enumerate(surface)
                ]
            else:
                surface_for_full = surface
            full_meshes = _build_mesh_list(x, surface_for_full, quiet=True)
            sizing_meshes = full_meshes
            if style == "morph" and morph_state is not None:
                # M3b box-containment fix: sizing a morph-tagged dataset
                # from its FULL, differently-ORDERED cloud would not be a
                # safe bound for the per-frame rebuilt mesh even setting
                # aside the cost above -- smooth_hull_3d's underlying
                # ConvexHull/Taubin-smoothing pipeline is not invariant to
                # input row order for hulls with many coplanar/degenerate
                # faces (e.g. a cube's flat sides), so the SAME points in a
                # different order can produce a mesh with a larger extent
                # than the one used to size the cube (verified empirically:
                # a cube-shaped cloud's full-order mesh vs. its Hungarian-
                # reordered `sampled` mesh differ in max |vertex| by more
                # than the fixed 2% margin). On top of that, mid-morph
                # interpolated points are convex combinations of two
                # consecutive `sampled` clouds and so can lie outside either
                # endpoint's OWN hull even though they always lie inside the
                # hull of their UNION. Fix: size the cube once, up front,
                # ONLY from meshes built with the EXACT `sampled` arrays
                # `update_morph` will actually draw (guaranteeing hold-frame
                # containment) plus one mesh built from the union of every
                # sampled cloud (a cheap, strictly-safe bound for every
                # interpolated frame, since every interpolated point is a
                # convex combination of union points) -- cheap regardless of
                # how large the ORIGINAL cloud was, since `sampled` is
                # capped at `morph_samples`.
                spec = morph_state["surface_spec"]
                if spec is not None:
                    sampled = morph_state["sampled"]
                    morph_sizing_meshes = [
                        build_mesh_3d(cloud, spec, dataset_label=" morph",
                                     quiet=True)
                        for cloud in sampled
                    ]
                    union_cloud = np.concatenate(sampled, axis=0)
                    morph_sizing_meshes.append(
                        build_mesh_3d(union_cloud, spec,
                                     dataset_label=" morph-union", quiet=True))
                    sizing_meshes = full_meshes + morph_sizing_meshes
            cube_scale_anim = surface_cube_scale(sizing_meshes)
            if (style == "morph" and morph_state is not None
                    and morph_state["surface_spec"] is not None):
                # full-sample duplication can make the endpoint+union
                # sizing bound above under-cover the worst actual mid-
                # morph frame -- see `_morph.MORPH_SURFACE_SIZING_MARGIN`.
                cube_scale_anim *= _morph.MORPH_SURFACE_SIZING_MARGIN
            if style == "spin":
                # 'spin' keeps the FULL dataset static (only the camera
                # rotates) -- reuse the just-built full-data meshes so
                # update_lines_spin only has to re-shade/re-cull per frame.
                update_lines_spin.meshes = full_meshes
            elif style == "morph" and morph_state is not None:
                # every OTHER morph-tagged slot is forced to None: only the
                # single `mesh_slot` chosen above ever gets a (per-frame
                # rebuilt) mesh -- there is only one traveling cloud, so
                # only one hull. `full_meshes[i]` is already `None` for
                # every morph-tagged `i` (see `surface_for_full` above, and
                # the M4 fix note), so this loop is only strictly needed for
                # `mesh_slot` itself; it is kept as an explicit, self-
                # documenting no-op over the rest for clarity/robustness.
                static_meshes = list(full_meshes)
                for i in morph_state["indices"]:
                    static_meshes[i] = None
                morph_state["static_meshes"] = static_meshes

        # the axes cube is redrawn by `update_lines_*` every frame (camera
        # angle/zoom change), but the LIMITS are set once, up front: sized
        # to `cube_scale_anim` so a surface (if any) never spills past the
        # visible frame, exactly mirroring the static 3-D path above.
        ax.set_xlim3d([-cube_scale_anim, cube_scale_anim])
        ax.set_ylim3d([-cube_scale_anim, cube_scale_anim])
        ax.set_zlim3d([-cube_scale_anim, cube_scale_anim])

        # density= (GH #108/#191): computed ONCE from the FULL dataset `x`
        # (never per-frame -- a KDE evaluation is ~536ms @ 50**3, far over a
        # 33ms frame budget) and drawn as a static background BEFORE the
        # FuncAnimation below is created. It is intentionally never touched
        # by any `update_lines_*` frame-update function, so it renders
        # identically (no shading/view dependence -- `shade=False`) across
        # every rotation angle and animation style.
        if density is not None:
            # animate_plot3D is only ever dispatched for 3-D data (see
            # dispatch_animate above; matplotlib animation has no 2-D path).
            _draw_density_3d(ax, x, density, density_colors)

        # focused=/tail_duration= (round17 #8, GH #275): `focused` governs
        # the OPAQUE "in-focus" head-window boundary for `animate='window'`
        # and for any dataset with a chemtrails/precog/bullettime trail;
        # plain `animate=True`/`'parallel'` with NO trail flag set on any
        # dataset keeps using `tail_duration` alone, unaffected by `focused`
        # (`plot.py`'s docstring/`focused=` resolution documents this as the
        # "ignored for parallel" case). `focused` reaching this function is
        # already fully resolved by `plot.py` (never `None` -- it defaults
        # to `tail_duration`'s own value there), so when it IS used the
        # numeric result is byte-identical to before whenever the caller
        # never passed an explicit `focused=`.
        # one implementation, shared with the plotly backend AND with the
        # forecast reveal schedule `plot()` builds (see `trails`)
        window_frames = head_window_frames(
            frame_rate, tail_duration, focused, style == "window",
            chemtrails, precog, bullettime)

        # get line animation
        if style in ["parallel", True, "window"]:
            # frames == round(frame_rate * duration), the documented frame
            # count, for EVERY dataset mix (release-1.0 audit): line datasets
            # are pre-interpolated onto exactly this grid, and marker-only/
            # 1-point datasets are paced onto it by `anim_window_bounds` --
            # previously frames came from x[0].shape[0] alone, so a longer
            # LATER dataset was silently truncated (F04-003), marker-only
            # animations ignored duration= entirely (F04-005/F05-010), and a
            # 1-point FIRST dataset produced a 1-frame "animation" (F05-012).
            line_ani = HyperFuncAnimation(
                fig,
                update_lines_parallel,
                max(1, int(round(frame_rate * duration))),
                fargs=(
                    x,
                    lines,
                    trail,
                    cube_scale_anim,
                    window_frames,
                    rotations,
                    zoom,
                    chemtrails,
                    precog,
                    bullettime,
                    elev,
                ),
                interval=1000 / frame_rate,
                blit=False,
                repeat=False,
            )
        elif style == "serial":
            line_ani = HyperFuncAnimation(
                fig,
                update_lines_serial,
                max(1, int(round(frame_rate * duration))),
                fargs=(x, lines, trail, cube_scale_anim, window_frames,
                       rotations, zoom, chemtrails, precog, bullettime, elev),
                interval=1000 / frame_rate,
                blit=False,
                repeat=False,
            )
        elif style == "spin":
            line_ani = HyperFuncAnimation(
                fig,
                update_lines_spin,
                max(1, int(round(frame_rate * duration))),
                fargs=(x, lines, cube_scale_anim, rotations, zoom, elev),
                interval=1000 / frame_rate,
                blit=False,
                repeat=False,
            )
        elif style == "morph":
            n_morph_datasets = len(morph_state["indices"])
            total_frames = max(1, int(round(frame_rate * duration)))
            frame_counts, _, azimuths = _morph.morph_schedule(
                n_morph_datasets, total_frames, rotations, azim)
            morph_state["frame_counts"] = frame_counts
            line_ani = HyperFuncAnimation(
                fig,
                update_morph,
                sum(frame_counts),
                fargs=(morph_state, cube_scale_anim, azimuths, zoom, elev),
                interval=1000 / frame_rate,
                blit=False,
                repeat=False,
            )
            # `HyperAnimation.n_segments` reads this. Tagged HERE, beside the
            # `sum(frame_counts)` that becomes `_save_count`, so the segment
            # count and the frame count can never describe different
            # schedules. The 2-D morph branch below carries the same tag --
            # tagging only one makes `n_segments` silently None for half of
            # all morphs, which reads as "not a morph" rather than as an
            # error.
            line_ani._hyp_morph_segments = len(frame_counts)

        return fig, ax, x, line_ani

    # 2-D animations (round17 #9, GH #123): fixed (non-rotating) viewport --
    # every style except 'spin' (meaningless without a 3-D camera to rotate)
    # works exactly like its 3-D counterpart, minus all camera-angle
    # bookkeeping. `rotations=`/`zoom=` are 3-D-camera controls with no 2-D
    # equivalent; `plot.py` already warns (once, before dispatching to
    # either backend) whenever either is set to a non-default value for
    # 2-D data, so nothing further happens with them here -- they are
    # simply never read. surface= tracking a per-frame 2-D window (an
    # outline rebuilt every frame, mirroring `_mesh_and_draw_3d`) was
    # judged out of scope; like the plotly backend (see `plotly_draw`'s
    # `n_surface_traces_2d` gate), surface= is silently a no-op for
    # animated 2-D plots. density= (computed once from the FULL dataset,
    # same as every animated 3-D style already does) IS supported, drawn
    # once as a static background before the FuncAnimation is created.
    def update_lines_parallel_2d(
        num, data_lines, lines, trail_lines, tail_duration=2,
        chemtrails=None, precog=None, bullettime=None,
    ):
        """2D counterpart of `update_lines_parallel` (fixed viewport, no camera/cube).

        Returns
        -------
        tuple of (list, list)
            `(lines, trail_lines)` -- the updated head-line and trail
            artists, for `blit=True` animation.
        """
        total_frames = max(1, int(round(frame_rate * duration)))
        # one clock per source dataset, exactly as in the 3-D updater above
        _windows = None
        if ownership is not None:
            _windows = dataset_window_bounds(
                num, total_frames, ownership,
                [d.shape[0] for d in data_lines], tail_duration)
        head_bounds = []                                    # GH #285
        for i, (line, data, trail) in enumerate(itertools.zip_longest(
                lines, data_lines, trail_lines)):
            # same F05-001/F05-008/F04-003/F05-012 slicing fixes as the 3-D
            # path -- see `anim_window_bounds`.
            if _windows is not None:
                win = _windows[i]
            else:
                _s, _e, _ts = anim_window_bounds(
                    num, total_frames, data.shape[0], tail_duration)
                win = RunWindow(_s, _e, _ts, max(0, _e - 1), True,
                                data.shape[0])
            start, end = win.head_start, win.head_end
            head_bounds.append((start, end))
            # antialias: draw the smooth curve spanning the same rows
            n_rows = data.shape[0]
            if trail is not None:
                ct, pc, bt = chemtrails[i], precog[i], bullettime[i]
                trail_seg = None
                if (pc and ct) or bt:
                    trail_seg = _aa_window(i, 0, n_rows, artist=trail)
                elif ct:
                    trail_seg = _aa_window(i, 0, win.past_stop, artist=trail)
                elif pc:
                    # `win.future_start`, never `end - 1` -- see the 3-D path
                    trail_seg = _aa_window(i, win.future_start, n_rows,
                                           artist=trail)
                if trail_seg is not None:
                    trail.set_data(trail_seg[:, 0], trail_seg[:, 1])

            window = _aa_window(i, start, end, artist=line)
            line.set_data(window[:, 0], window[:, 1])

        _sync_anim_labels(num, tail_duration)
        if frame_hooks is not None:
            frame_hooks.record(
                frame=int(num), n_frames=int(total_frames),
                artists=list(lines) + [t for t in trail_lines if t is not None],
                datasets=list(data_lines), style=animate, order='parallel',
                current_index=None, current_fraction=None,
                revealed_counts=tuple(e for _, e in head_bounds),
                window_bounds=tuple(head_bounds))
        return lines, trail_lines

    def update_lines_serial_2d(num, data_lines, lines, trail_lines,
                               window_frames=1, chemtrails=None, precog=None,
                               bullettime=None):
        """2D counterpart of `update_lines_serial` (fixed viewport, no
        camera/cube) -- including the same chemtrails/precog/bullettime trail
        composition on the currently-revealing dataset (see
        `update_lines_serial`).

        Returns
        -------
        list
            The updated matplotlib line artists, for `blit=True` animation.
        """
        total_frames = max(1, int(round(frame_rate * duration)))
        lengths = [d.shape[0] for d in data_lines]
        total_points = sum(lengths)
        revealed = total_points * num / max(1, total_frames - 1)
        _counts = serial_reveal_counts(lengths, num, total_frames)

        for i, (line, data, trail) in enumerate(itertools.zip_longest(
                lines, data_lines, trail_lines)):
            n_pts = data.shape[0]
            shown = _counts[i]

            ct = chemtrails[i] if chemtrails is not None else False
            pc = precog[i] if precog is not None else False
            bt = bullettime[i] if bullettime is not None else False
            has_trail = trail is not None and (ct or pc or bt)

            # antialias: resolve ORIGINAL-row bounds, draw the smooth curve
            trail_bounds = None
            if not has_trail:
                head_bounds = (0, shown)                  # UNCHANGED
            elif shown <= 0:
                head_bounds = (0, 0)
            elif shown >= n_pts:
                head_bounds = (0, n_pts)
            else:
                w = max(1, int(round(window_frames * n_pts
                                     / max(1, total_points))))
                head_bounds = (max(0, shown - 1 - w), shown)
                if (ct and pc) or bt:
                    trail_bounds = (0, n_pts)              # bullettime: whole
                elif ct:
                    trail_bounds = (0, shown)              # chemtrails: past
                else:
                    trail_bounds = (max(0, shown - 1), n_pts)  # precog: future

            head = _aa_window(i, *head_bounds, artist=line)
            trail_seg = (data[:0] if trail_bounds is None
                         else _aa_window(i, *trail_bounds, artist=trail))
            line.set_data(head[:, 0], head[:, 1])
            if trail is not None:
                trail.set_data(trail_seg[:, 0], trail_seg[:, 1])

        _sync_anim_labels(num, 0, revealed=revealed)
        if frame_hooks is not None:
            _idx, _frac = serial_current_index(_counts, lengths)
            frame_hooks.record(
                frame=int(num), n_frames=int(total_frames),
                artists=list(lines) + [t for t in trail_lines if t is not None],
                datasets=list(data_lines), style='serial', order='serial',
                current_index=_idx, current_fraction=_frac,
                revealed_counts=_counts,
                window_bounds=tuple((0, c) for c in _counts))
        return lines

    def update_morph_2d(num, morph_state):
        """2D counterpart of `update_morph`: move the single traveling morph point-cloud artist for frame `num`.

        Returns
        -------
        tuple of (matplotlib.lines.Line2D,)
            The updated morph artist, for `blit=True` animation.
        """
        seg_idx, step, n_steps = _morph.frame_to_segment(
            morph_state["frame_counts"], num)
        pts = _morph.morph_positions(morph_state["sampled"], seg_idx, step,
                                     n_steps)
        color = _morph.morph_color(morph_state["colors"], seg_idx, step,
                                   n_steps)
        hide = _morph.morph_visible_mask(morph_state.get("dup_masks"),
                                         seg_idx)
        draw_pts = pts[~hide] if hide is not None else pts

        artist = morph_state["artist"]
        artist.set_data(draw_pts[:, 0], draw_pts[:, 1])
        artist.set_color(color)
        alpha = _morph.morph_alpha(morph_state.get("alphas"), seg_idx,
                                   step, n_steps)
        if alpha is not None:
            artist.set_alpha(alpha)

        _sync_anim_labels(num, 0, hide_all=True)
        if frame_hooks is not None:
            frame_hooks.record(
                frame=int(num),
                n_frames=int(sum(morph_state["frame_counts"])),
                artists=[morph_state["artist"]],
                datasets=list(morph_state["sampled"]),
                style='morph', order='serial',
                # see the identical note in `update_morph` (3-D): `seg_idx
                # // 2` is a position within the morph SEQUENCE, which only
                # equals the FINAL dataset index when every dataset is
                # tagged -- `morph_state["indices"]` maps sequence position
                # back to the actual dataset index for partial-tag lists.
                current_index=morph_state["indices"][seg_idx // 2],
                current_fraction=step / max(1, n_steps - 1),
                revealed_counts=None,
                segment_index=seg_idx,
                segment_kind='hold' if seg_idx % 2 == 0 else 'transition')
        return (artist,)

    def animate_plot2D(
        x,
        tail_duration=2,
        focused=None,
        rotations=1,
        zoom=1,
        chemtrails=None,
        precog=None,
        bullettime=None,
        frame_rate=30,
        elev=10,
        style="parallel",
        morph_tags=None,
        morph_colors=None,
        morph_samples=None,
        morph_loop=False,
    ):
        """2D counterpart of `animate_plot3D`: build and run a fixed-viewport `FuncAnimation` (no camera rotation).

        `style='spin'` is not supported (there is no 3D camera to
        rotate) and raises `ValueError`. `rotations=`/`zoom=` are
        accepted for signature parity with `animate_plot3D` but ignored
        (they have no 2D meaning; `plot.py` already warns the caller
        when either is set to a non-default value).

        Parameters
        ----------
        x : list of numpy.ndarray
            2-column datasets to animate.
        tail_duration, focused, chemtrails, precog, bullettime,
        frame_rate, elev, style, morph_tags, morph_colors, morph_samples
            Same meaning as in `animate_plot3D`.

        Returns
        -------
        tuple
            `(fig, ax, x, line_ani)` -- the created Figure, Axes, original
            data `x`, and the `FuncAnimation` instance.

        Raises
        ------
        ValueError
            If `style='spin'`.
        """
        if style == "spin":
            raise ValueError(
                "animate='spin' rotates the 3-D camera and has no meaning "
                "for 2-D data (2-D animations use a fixed, non-rotating "
                "viewport). Use 'parallel'/True, 'serial', 'window', "
                "'chemtrails', 'precog', 'bullettime', or 'morph' instead."
            )

        # initialize plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        def _wants_trail(idx):
            # 'serial' composes with the trail flags (see animate_plot3D's
            # `_wants_trail`); only 'morph'/'window' skip trails in 2-D
            # ('spin' is rejected for 2-D data before this point).
            if style in ("morph", "window"):
                return False
            return chemtrails[idx] or precog[idx] or bullettime[idx]

        # see animate_plot3D: pop linewidth once per dataset so head lines
        # and trails share the user's linewidth= (X6-code-org-plot-009)
        linewidths = [
            kwargs_list[idx].pop("linewidth", 1)
            if isinstance(kwargs_list[idx], dict) else 1
            for idx in range(len(x))
        ]

        # see animate_plot3D: fold the 0.3 trail-fade factor into any alpha the
        # dataset kwargs already carry, so a per-trace alpha from MultiIndex
        # expansion does not collide with a bare `alpha=0.3` ("got multiple
        # values for keyword argument 'alpha'").
        def _trail_kwargs(kw):
            kw = dict(kw) if isinstance(kw, dict) else {}
            kw["alpha"] = 0.3 * kw.pop("alpha", 1.0)
            return kw

        trail = []
        if fmt is not None:
            lines = [
                ax.plot(
                    dat[0:1, 0],
                    dat[0:1, 1],
                    fmt[idx],
                    linewidth=linewidths[idx],
                    **kwargs_list[idx]
                )[0]
                for idx, dat in enumerate(x)
            ]
            if any(is_line(f) for f in fmt):
                trail = [
                    ax.plot(
                        dat[0:1, 0],
                        dat[0:1, 1],
                        fmt[idx],
                        linewidth=linewidths[idx],
                        **_trail_kwargs(kwargs_list[idx])
                    )[0] if _wants_trail(idx) else None
                    for idx, dat in enumerate(x)
                ]
        else:
            lines = [
                ax.plot(
                    dat[0:1, 0],
                    dat[0:1, 1],
                    linewidth=linewidths[idx],
                    **kwargs_list[idx]
                )[0]
                for idx, dat in enumerate(x)
            ]
            if is_line(fmt):
                trail = [
                    ax.plot(
                        dat[0:1, 0],
                        dat[0:1, 1],
                        linewidth=linewidths[idx],
                        **_trail_kwargs(kwargs_list[idx])
                    )[0] if _wants_trail(idx) else None
                    for idx, dat in enumerate(x)
                ]
        for _trail_line in trail:
            if _trail_line is not None:
                _trail_line.set_label('_nolegend_')

        # animate='morph': single traveling point-cloud artist, exactly
        # mirroring `animate_plot3D`'s morph setup but in 2-D.
        morph_state = None
        if style == "morph":
            _tags = morph_tags if morph_tags is not None else [True] * len(x)
            morph_indices = [i for i, tag in enumerate(_tags) if tag]
            clouds = [np.asarray(x[i], dtype=np.float64)[:, :2]
                     for i in morph_indices]
            sampled, dup_masks = _morph.sample_and_match_clouds(
                clouds, morph_samples=morph_samples, loop=morph_loop)
            if morph_loop:
                # see the identical note in `animate_plot3D` above
                morph_indices = morph_indices + [morph_indices[0]]
            ds_colors = [
                tuple(morph_colors[i]) if morph_colors is not None
                else (0.2, 0.4, 0.8)
                for i in morph_indices
            ]

            for i in morph_indices:
                lines[i].set_visible(False)
                if i < len(trail) and trail[i] is not None:
                    trail[i].set_visible(False)

            # any UNTAGGED (static backdrop) dataset is drawn once, in
            # full -- see the identical M4 fix note in `animate_plot3D`.
            for i in range(len(x)):
                if i in morph_indices:
                    continue
                full = x[i]
                lines[i].set_data(full[:, 0], full[:, 1])

            mesh_slot = morph_indices[0]
            first_pts = sampled[0]
            first_hide = _morph.morph_visible_mask(dup_masks, 0)
            first_draw = (first_pts[~first_hide] if first_hide is not None
                         else first_pts)
            _mkw = (kwargs_list[mesh_slot]
                   if isinstance(kwargs_list[mesh_slot], dict) else {})
            morph_markersize = _mkw.get("markersize") or 1.5
            # GH #284: see the identical note in `animate_plot3D`.
            ds_alphas = [
                (kwargs_list[i] if isinstance(kwargs_list[i], dict)
                 else {}).get("alpha")
                for i in morph_indices
            ]
            (morph_artist,) = ax.plot(
                first_draw[:, 0], first_draw[:, 1],
                linestyle="None", marker=".", markersize=morph_markersize,
                color=ds_colors[0],
                alpha=_morph.morph_alpha(ds_alphas, 0, 0, 1),
            )
            morph_artist.set_label("_nolegend_")

            morph_state = dict(
                sampled=sampled, dup_masks=dup_masks, colors=ds_colors,
                alphas=ds_alphas,
                artist=morph_artist, indices=morph_indices,
            )

        # border square + fixed axes limits (matches the static 2-D path).
        # Under axis_scale='data' the limits are FIXED from the full data
        # (computed by `plot()`, forecasts included) rather than autoscaled
        # per frame, so the viewport never jumps mid-animation.
        frame_2d(ax)

        # density= (GH #108/#191): computed ONCE from the FULL dataset,
        # same as every animated 3-D style -- see `animate_plot3D`.
        if density is not None:
            _draw_density_2d(ax, x, density, density_colors,
                             clip_unit=axis_scale != 'data')

        # one implementation, shared with the plotly backend AND with the
        # forecast reveal schedule `plot()` builds (see `trails`)
        window_frames = head_window_frames(
            frame_rate, tail_duration, focused, style == "window",
            chemtrails, precog, bullettime)

        if style in ["parallel", True, "window"]:
            # frames == round(frame_rate * duration) -- see the identical
            # F04-003/F04-005/F05-010/F05-012 note in `animate_plot3D`.
            line_ani = HyperFuncAnimation(
                fig,
                update_lines_parallel_2d,
                max(1, int(round(frame_rate * duration))),
                fargs=(x, lines, trail, window_frames, chemtrails, precog,
                      bullettime),
                interval=1000 / frame_rate,
                blit=False,
                repeat=False,
            )
        elif style == "serial":
            line_ani = HyperFuncAnimation(
                fig,
                update_lines_serial_2d,
                max(1, int(round(frame_rate * duration))),
                fargs=(x, lines, trail, window_frames, chemtrails, precog,
                      bullettime),
                interval=1000 / frame_rate,
                blit=False,
                repeat=False,
            )
        elif style == "morph":
            # 2-D morphs always use even segment timing regardless of
            # `rotations=` -- see the module-level note above this
            # function: `rotations` doubles as a per-segment PACING
            # control for `animate='morph'` in 3-D (not purely a camera
            # control), but is ignored uniformly for every 2-D style for
            # consistency (and `plot.py` has already warned about it).
            n_morph_datasets = len(morph_state["indices"])
            total_frames = max(1, int(round(frame_rate * duration)))
            frame_counts, _, _ = _morph.morph_schedule(
                n_morph_datasets, total_frames, 1, 0)
            morph_state["frame_counts"] = frame_counts
            line_ani = HyperFuncAnimation(
                fig,
                update_morph_2d,
                sum(frame_counts),
                fargs=(morph_state,),
                interval=1000 / frame_rate,
                blit=False,
                repeat=False,
            )
            # the 2-D half of the `n_segments` tag -- see the 3-D branch above
            line_ani._hyp_morph_segments = len(frame_counts)

        return fig, ax, x, line_ani

    # if a single point, but formatted as a line, replace with a point
    for i, (xi, fi) in enumerate(zip(x, fmt)):
        if xi.shape[0] == 1 and fi in ("-", ":", "--"):
            fmt[i] = "."

    if not show:
        # prevents the backend from rendering this plot
        plt.ioff()

    if frame_kwargs is None:
        frame_kwargs = {}

    if animate in [True, "parallel", "spin", "serial", "morph", "window"]:
        # round17 #9 (GH #123): animations now support 2-D as well as 3-D
        # data (`dispatch_animate` above routes to `animate_plot2D` or
        # `animate_plot3D` accordingly); 1-D (and any other dimensionality)
        # still has no animatable trajectory concept. A real ValueError, not
        # an assert (F04-012: asserts vanish under `python -O`, and every
        # sibling animate validation raises ValueError).
        if x[0].shape[1] not in (2, 3):
            raise ValueError(
                "Animations are only supported for 2-D or 3-D plots (got "
                f"{x[0].shape[1]}-D data); pass ndims=2 or ndims=3 (the "
                "default)."
            )

        # animation params
        ani_params = dict(
            tail_duration=tail_duration,
            focused=focused,
            rotations=rotations,
            zoom=zoom,
            chemtrails=chemtrails,
            precog=precog,
            bullettime=bullettime,
            frame_rate=frame_rate,
            elev=elev,
            style=animate,
            morph_tags=morph_tags,
            morph_colors=morph_colors,
            morph_samples=morph_samples,
            morph_loop=morph_loop,
        )

        # dispatch animation
        fig, ax, data, line_ani = dispatch_animate(x, ani_params)

    else:

        # dispatch static
        fig, ax, data = dispatch_static(x, ax)

        # if 3d, plot the cube
        if x[0].shape[1] == 3:

            # surface= (GH #109 round 2): build each dataset's mesh ONCE,
            # from the FULL static data, before the cube is drawn -- the
            # smoothed hull (pre_inflate + smoothing overshoot, further
            # grown by `_rescale_for_containment` for small point clouds)
            # can bulge past the standard [-1, 1] data cube, so the cube
            # and axis limits must be sized to whatever was actually built,
            # not assumed to be 1. Reusing this SAME mesh_list for the
            # shading pass below (rather than rebuilding it) also avoids
            # computing every dataset's mesh twice.
            mesh_list = _build_mesh_list(data, surface) if surface is not None else None
            cube_scale = (surface_cube_scale(mesh_list)
                         if mesh_list is not None else 1)

            # plot cube
            plot_cube(cube_scale, **frame_kwargs)

            # set the axes properties
            ax.set_xlim3d([-cube_scale, cube_scale])
            ax.set_ylim3d([-cube_scale, cube_scale])
            ax.set_zlim3d([-cube_scale, cube_scale])

            # initialize the view
            ax.view_init(elev=elev, azim=azim)

            # surface= (GH #109): smooth lit hull surfaces, one per dataset
            if surface is not None:
                _shade_and_cull_3d(ax, mesh_list, surface, surface_colors,
                                   elev, azim,
                                   surface_point_colors=surface_point_colors)
                _hide_no_keep_points(ax.lines, surface)

            # density= (GH #108/#191): subtle KDE density shading, one
            # layer per dataset (or one pooled layer), below the data
            if density is not None:
                _draw_density_3d(ax, data, density, density_colors)

        elif x[0].shape[1] == 2:

            # plot square + axis limits (see `frame_2d`: axis_scale='data'
            # draws no square and keeps the data's own coordinates)
            frame_2d(ax)

            # surface= (GH #109): smooth filled hull outlines, below the data
            if surface is not None:
                _fill_and_draw_2d(ax, data, surface, surface_colors)
                _hide_no_keep_points(ax.lines, surface)

            # density= (GH #108/#191): subtle KDE density shading, one
            # layer per dataset (or one pooled layer), below the data
            if density is not None:
                _draw_density_2d(ax, data, density, density_colors,
                                 clip_unit=axis_scale != 'data')

        else:
            # 1-D: no frame to draw, but explicit limits still apply (GH
            # #285 -- `ndims=1` series mode uses them for the reveal axis)
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)

        # set line_ani to empty
        line_ani = None

    # remove axes -- hypertools draws its own cube/square frame in place of
    # matplotlib's default axes box, so ticks/spines/(3-D) panes are always
    # hidden. `Axes.set_axis_off()`/`Axes3D.set_axis_off()` remove the
    # WHOLE axis (ticks AND the axis label Text artist together) from the
    # draw list, so xlabel=/ylabel=/zlabel= (round17 #7) would never
    # actually render if drawn while the axis is off (`get_xlabel()` etc.
    # would still return the right string -- the Text's `.get_text()` is
    # set regardless -- but the figure itself would show nothing, verified
    # empirically: 0 changed pixels). When any of the three is given, hide
    # ticks/spines/gridlines/(3-D) panes INDIVIDUALLY instead, leaving the
    # axis unglobally "off" so its label artist(s) still draw; byte-
    # identical to plain `set_axis_off()` otherwise (also verified
    # empirically: 0 changed pixels when no label is requested).
    #
    # axis_scale='data' (GH #285) is the deliberate exception: its whole
    # point is that the drawn coordinates ARE the data's own, so its ticks
    # and spines stay on (matplotlib's defaults) and only the top/right
    # spines are dropped, the way a plain time-series panel is drawn.
    if axis_scale == 'data':
        for _side in ('top', 'right'):
            if _side in ax.spines:
                ax.spines[_side].set_visible(False)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if x_date:
            # the x column holds `date2num` day numbers (see `plot()`'s
            # ndims=1 series mode); without a date converter they would tick
            # as five-digit floats
            ax.xaxis_date()
    elif xlabel is None and ylabel is None and zlabel is None:
        ax.set_axis_off()
    elif hasattr(ax, "get_proj"):
        # 3-D: Axes3D's own `_axis3don` flag gates panes/gridlines/ticks/
        # labels ALL TOGETHER (see `Axes3D.draw`) -- there is no coarser
        # public on/off switch to leave alone, so each axis's individual
        # sub-artists are hidden instead, with `_axis3don` left True so
        # `axis.draw()` (which draws the label) still runs.
        for _axis in ax._axis_map.values():
            _axis.pane.set_visible(False)
            # NOTE: alpha=0 (transparent), not `set_visible(False)` --
            # `Axis3D.get_tightbbox` (called by `plt.tight_layout()` on
            # static plots below) unions the bboxes of the axis line,
            # every tick, and (only if `for_layout_only=False`) the
            # label; with ticks emptied by `set_ticks([])` above AND the
            # line invisible, that union would be over an EMPTY list,
            # which raises `ValueError` inside matplotlib itself. An
            # alpha-0 line still counts as "visible" for bbox purposes
            # (so the union is never empty) while remaining fully
            # transparent -- i.e. absent from the rendered image.
            _axis.line.set_alpha(0)
            _axis.gridlines.set_visible(False)
            _axis.set_ticks([])
        ax.patch.set_visible(False)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if zlabel is not None:
            ax.set_zlabel(zlabel)
    else:
        # 2-D (or 1-D): hide ticks/spines/gridlines individually, leaving
        # `axison` at its default True so the axis label Text artist(s)
        # still draw.
        ax.set_xticks([])
        ax.set_yticks([])
        for _spine in ax.spines.values():
            _spine.set_visible(False)
        ax.grid(False)
        ax.patch.set_visible(False)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        # zlabel on a 2-D/1-D plot is rejected upstream in plot.py
        # (ValueError, before the pipeline even runs) -- zlabel is
        # guaranteed None here.

    # add labels
    add_labels(x, labels, explore=explore)

    # add title
    if title is not None:
        _apply_title(ax, title, font=font, title_kwargs=title_kwargs)

    # add legend: to the RIGHT of the plot, vertically centered on the
    # box (never overlapping the data). `prop=font` (GH #205) applies the
    # SAME resolved font to every legend text entry -- `_fit_right_legend`
    # (plot.py) measures the legend's true extent from these Text artists'
    # own fontproperties, so this also fixes multibyte legend clipping
    # without any change needed there.
    if legend is not None or legend_entries is not None:
        _legend_call = legend_call_kwargs(
            is_3d=hasattr(ax, "get_proj"), zlabel=zlabel, font=font,
            legend_kwargs=legend_kwargs)
        if legend_entries is not None:
            # explicit entries (matrix/mixture hue palette swatches, or
            # `legend_colors=[(label, color), ...]`): proxy handles, since
            # no drawn artist carries these labels.
            ax.legend(handles=_legend_proxy_handles(legend_entries, fmt),
                      **_legend_call)
        else:
            _legend_artist = ax.legend(**_legend_call)
            if legend_colors is not None:
                _recolor_legend_handles(_legend_artist, legend_colors)

    if size is not None:
        fig.set_size_inches(size)

    if line_ani is not None:
        _make_save_dpi_safe(line_ani)

    return fig, ax, data, line_ani
