#!/usr/bin/env python

import functools
import itertools
import warnings

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import matplotlib.animation as animation
import matplotlib.patches as patches
from .._shared.helpers import *
from .meshutil import backface_cull, blinn_phong_colors
from .surface import (
    build_mesh_3d,
    build_outline_2d,
    mpl_lighting_kwargs,
    surface_cube_scale,
    view_vector,
)
from .trails import broadcast_trail_flag
from . import morph as _morph
from .density import (
    DENSITY_DEFAULTS,
    HAS_SKIMAGE,
    POOLED_COLOR,
    alpha_colormap,
    bbox_extent,
    density_alpha_boost,
    fit_kde,
    iso_surfaces_3d,
    kde_grid_2d,
    kde_grid_3d,
    resolve_grid,
    resolve_iso_fracs_alphas,
)


def _resolve_surface_color(spec, fallback_rgb):
    """Base RGB for one dataset's surface: `spec['color']` if given,
    otherwise the dataset's own drawn color (`fallback_rgb`)."""
    return mcolors.to_rgb(spec["color"]) if spec["color"] is not None else fallback_rgb


def _draw_one_density_2d(ax, pts, spec, color, label=""):
    """Draw a single subtle alpha-ramped ``imshow`` KDE layer for one
    dataset (or the pooled cloud), below the data (``zorder=-1``)."""
    kde = fit_kde(pts, dataset_label=label)
    if kde is None:
        return
    gridsize = resolve_grid(spec, 2)
    _, _, Z, extent = kde_grid_2d(pts, kde, gridsize=gridsize)
    cmap = alpha_colormap(color, spec["alpha"])
    im = ax.imshow(Z, origin="lower", extent=extent, aspect="auto",
                   cmap=cmap, interpolation="bilinear", zorder=-1)
    im.set_label("_nolegend_")


def _draw_density_2d(ax, points_list, density, density_colors):
    """Draw each dataset's (or, with ``per_group=False``, one pooled) 2-D
    KDE density layer (GH #108/#191)."""
    if density[0] is not None and not density[0].get("per_group", True):
        all_pts = np.vstack([np.asarray(p)[:, :2] for p in points_list])
        _draw_one_density_2d(ax, all_pts, density[0], POOLED_COLOR,
                             label=" (pooled)")
        return
    for i, (pts, spec) in enumerate(zip(points_list, density)):
        if spec is None:
            continue
        _draw_one_density_2d(ax, np.asarray(pts)[:, :2], spec,
                             density_colors[i], label=f" {i}")


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
    if HAS_SKIMAGE:
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
            ax.add_collection3d(coll)
    else:
        warnings.warn(
            f"density: scikit-image is not installed -- dataset{label}'s "
            "3-D density falls back to a translucent scatter 'fog' instead "
            "of smooth iso-surfaces. Install it with `pip install "
            "hypertools[density3d]`, or use backend='plotly' for full "
            "volumetric rendering.",
            UserWarning,
        )
        rng = np.random.default_rng()
        fog = kde.resample(4000, seed=rng).T
        ax.scatter(fog[:, 0], fog[:, 1], fog[:, 2], s=6, c=[color],
                  alpha=min(0.03 * alpha_scale, 1.0), edgecolors="none",
                  depthshade=False, label="_nolegend_")


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
                       prior_colls=None):
    """(Re)build a ``Poly3DCollection`` per dataset from PRECOMPUTED
    ``(verts, faces)`` meshes, shading/culling for the CURRENT `elev`/`azim`,
    removing `prior_colls` first (animation frame swap). Returns the new
    per-dataset collection list (``None`` where that dataset has no surface)
    so the caller can pass it back in as `prior_colls` next frame."""
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
                      prior_colls=None, quiet=False):
    """Build fresh meshes from the CURRENT per-dataset point windows
    (`points_list`) and delegate to `_shade_and_cull_3d`. Used whenever the
    visible point window changes (static plots; 'parallel'/'serial'
    animation frames)."""
    mesh_list = _build_mesh_list(points_list, surface, quiet=quiet)
    return _shade_and_cull_3d(ax, mesh_list, surface, surface_colors, elev,
                              azim, prior_colls=prior_colls)


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


def _draw(
    x,
    legend=None,
    title=None,
    labels=False,
    show=True,
    kwargs_list=None,
    fmt=None,
    animate=False,
    tail_duration=2,
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
    density=None,
    density_colors=None,
    morph_tags=None,
    morph_colors=None,
    morph_samples=None,
):
    """
    Draws the plot
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

    # handle static plots
    def dispatch_static(x, ax=None):
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

    # plot data in 1D
    def plot1D(data, fig, ax):
        n = len(data)
        for i in range(n):
            ikwargs = kwargs_list[i]
            if fmt is None:
                ax.plot(data[i][:, 0], **ikwargs)
            else:
                ax.plot(data[i][:, 0], fmt[i], **ikwargs)
        return fig, ax, data

    # plot data in 2D
    def plot2D(data, fig, ax):
        n = len(data)
        for i in range(n):
            ikwargs = kwargs_list[i]
            if fmt is None:
                ax.plot(data[i][:, 0], data[i][:, 1], **ikwargs)
            else:
                ax.plot(data[i][:, 0], data[i][:, 1], fmt[i], **ikwargs)
        return fig, ax, data

    # plot data in 3D
    def plot3D(data, fig, ax):
        n = len(data)
        for i in range(n):
            ikwargs = kwargs_list[i]
            if fmt is None:
                ax.plot(data[i][:, 0], data[i][:, 1], data[i][:, 2], **ikwargs)
            else:
                ax.plot(data[i][:, 0], data[i][:, 1], data[i][:, 2], fmt[i], **ikwargs)
        return fig, ax, data

    def annotate_plot(data, labels):
        """Create labels in 3d chart
        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            labels (list) - list of labels of shape (numPoints,1)
        Returns:
            None
        """

        global labels_and_points
        labels_and_points = []

        if data[0].shape[-1] > 2:
            proj = ax.get_proj()

        for idx, x in enumerate(data):
            if labels[idx] is not None:
                if data[0].shape[-1] > 2:
                    x2, y2, _ = proj3d.proj_transform(x[0], x[1], x[2], proj)
                    label = plt.annotate(
                        labels[idx],
                        xy=(x2, y2),
                        xytext=(-20, 20),
                        textcoords="offset points",
                        ha="right",
                        va="bottom",
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.5),
                        arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"),
                        family="serif",
                    )
                    labels_and_points.append((label, x[0], x[1], x[2]))
                elif data[0].shape[-1] == 2:
                    x2, y2 = x[0], x[1]
                    label = plt.annotate(
                        labels[idx],
                        xy=(x2, y2),
                        xytext=(-20, 20),
                        textcoords="offset points",
                        ha="right",
                        va="bottom",
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.5),
                        arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"),
                        family="serif",
                    )
                    label.draggable()
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
                # fig.canvas.mpl_connect('button_press_event', lambda event: onMouseClick(event, X, labels))  # on mouse click
            else:
                fig.canvas.mpl_connect(
                    "motion_notify_event", lambda event: onMouseMotion(event, X)
                )  # on mouse motion
                # fig.canvas.mpl_connect('button_press_event', lambda event: onMouseClick(event, X, labels))  # on mouse click

        elif labels is not None:
            X = np.vstack(x)
            if any(isinstance(el, list) for el in labels):
                labels = list(itertools.chain(*labels))
            annotate_plot(X, labels)
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
            point[0], point[1], point[2], plt.gca().get_proj()
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

        if type(labels) is list:
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

        annotate_plot_explore.label = plt.annotate(
            label,
            xy=(x2, y2),
            xytext=(-20, 20),
            textcoords="offset points",
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
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
            if type(labels) is list:
                annotate_plot_explore(X, closestIndex, labels)
                closestIndex_prev = closestIndex
            else:
                annotate_plot_explore(X, closestIndex)
                closestIndex_prev = closestIndex

    def plot_cube(scale, **cube_kwargs):
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
            plane_list.append(ax.plot_wireframe(Xs, Ys, Zs, **cube_kwargs))
        return plane_list

    def plot_square(ax, scale=1, **square_kwargs):
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

        if hasattr(update_lines_parallel, "planes"):
            for plane in update_lines_parallel.planes:
                plane.remove()

        update_lines_parallel.planes = plot_cube(cube_scale, **frame_kwargs)
        azim_now = rotations * (360 * (num / data_lines[0].shape[0]))
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
        for i, (line, data, trail) in enumerate(itertools.zip_longest(
                lines, data_lines, trail_lines)):

            if trail is not None:
                ct, pc, bt = chemtrails[i], precog[i], bullettime[i]
                if (pc and ct) or bt:
                    trail.set_data(data[:, 0:2].T)
                    trail.set_3d_properties(data[:, 2])
                elif ct:
                    trail.set_data(data[0 : num - tail_duration + 1, 0:2].T)
                    trail.set_3d_properties(data[0 : num - tail_duration + 1, 2])
                elif pc:
                    trail.set_data(data[num + 1 :, 0:2].T)
                    trail.set_3d_properties(data[num + 1 :, 2])

            if num <= tail_duration:
                window = data[0 : num + 1]
            else:
                window = data[num - tail_duration : num + 1]
            line.set_data(window[:, 0:2].T)
            line.set_3d_properties(window[:, 2])
            windows.append(window)

        # surface= (GH #109): recompute each dataset's hull from its CURRENT
        # visible window (same window as the head line above) and the
        # current camera view (backface culling depends on it)
        if surface is not None:
            prior = getattr(update_lines_parallel, "surface_colls", None)
            update_lines_parallel.surface_colls = _mesh_and_draw_3d(
                ax, windows, surface, surface_colors, elev, azim_now,
                prior_colls=prior, quiet=True)

        return lines, trail_lines

    def update_lines_spin(
        num, data_lines, lines, cube_scale, rotations=1, zoom=1, elev=10
    ):

        if hasattr(update_lines_spin, "planes"):
            for plane in update_lines_spin.planes:
                plane.remove()

        update_lines_spin.planes = plot_cube(cube_scale, **frame_kwargs)
        azim_now = rotations * (360 * (num / (frame_rate * duration)))
        ax.view_init(elev=elev, azim=azim_now)
        # Axes3D.dist was removed in matplotlib >= 3.8, silently disabling
        # zoom; set_box_aspect(zoom=...) is the supported equivalent. See
        # _anim_box_zoom for the (slightly zoomed-out) animation mapping.
        ax.set_box_aspect(None, zoom=_anim_box_zoom(zoom))

        for line, data in zip(lines, data_lines):
            line.set_data(data[:, 0:2].T)
            line.set_3d_properties(data[:, 2])

        # surface= (GH #109): the FULL dataset is static in 'spin' mode
        # (only the camera rotates), so the mesh itself is precomputed once
        # (`update_lines_spin.meshes`, set in animate_plot3D before this
        # runs) -- only shading/backface-culling are recomputed per frame.
        if surface is not None:
            prior = getattr(update_lines_spin, "surface_colls", None)
            update_lines_spin.surface_colls = _shade_and_cull_3d(
                ax, update_lines_spin.meshes, surface, surface_colors, elev,
                azim_now, prior_colls=prior)

        return lines

    def update_lines_serial(
        num, data_lines, lines, cube_scale, rotations=1, zoom=1, elev=10
    ):
        """Serial animation: datasets appear ONE AT A TIME, each growing
        point-by-point into place while all previous datasets stay fully
        drawn (e.g. conversation turns adding to a shared embedding space).
        Datasets are never connected to each other."""
        if hasattr(update_lines_serial, "planes"):
            for plane in update_lines_serial.planes:
                plane.remove()
        update_lines_serial.planes = plot_cube(cube_scale, **frame_kwargs)

        total_frames = frame_rate * duration
        ax.view_init(elev=elev,
                     azim=azim + rotations * 360.0 * num / total_frames)
        ax.set_box_aspect(None, zoom=_anim_box_zoom(zoom))

        lengths = [d.shape[0] for d in data_lines]
        total_points = sum(lengths)
        revealed = total_points * num / max(1, total_frames - 1)

        start = 0
        windows = []
        for line, data in zip(lines, data_lines):
            shown = int(np.clip(revealed - start, 0, data.shape[0]))
            window = data[:shown]
            line.set_data(window[:, 0:2].T)
            line.set_3d_properties(window[:, 2])
            windows.append(window)
            start += data.shape[0]

        # surface= (GH #109): each dataset's hull follows its own currently-
        # revealed portion (same window as its line above)
        if surface is not None:
            azim_now = azim + rotations * 360.0 * num / total_frames
            prior = getattr(update_lines_serial, "surface_colls", None)
            update_lines_serial.surface_colls = _mesh_and_draw_3d(
                ax, windows, surface, surface_colors, elev, azim_now,
                prior_colls=prior, quiet=True)

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
            prior = getattr(update_morph, "surface_colls", None)
            update_morph.surface_colls = _shade_and_cull_3d(
                ax, frame_meshes, surface, frame_colors, elev, azim_now,
                prior_colls=prior)

        return (artist,)

    def dispatch_animate(x, ani_params):
        if x[0].shape[1] == 3:
            return animate_plot3D(x, **ani_params)

    def animate_plot3D(
        x,
        tail_duration=2,
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
    ):

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
        # 'spin'/'serial' (GH #127 follow-up): neither `update_lines_spin`
        # nor `update_lines_serial` accepts (or ever touches) a trail_lines
        # argument -- 'spin' has no "current position" for a trail to lead/
        # follow (only the camera moves) and 'serial' already communicates
        # elapsed time via its point-by-point reveal. Trail artists created
        # here for those two styles would therefore stay frozen at their
        # initial (single-point) state for the whole animation: invisible/
        # useless stubs. `plot.py` already warns the caller and names the
        # ignored flags/dataset indices; this just skips ever creating them
        # so `_wants_trail` is forced False for every dataset in these modes.
        def _wants_trail(idx):
            if style in ("spin", "serial", "morph"):
                return False
            return chemtrails[idx] or precog[idx] or bullettime[idx]

        trail = []
        if fmt is not None:
            lines = [
                ax.plot(
                    dat[0:1, 0],
                    dat[0:1, 1],
                    dat[0:1, 2],
                    fmt[idx],
                    linewidth=kwargs_list[idx].pop("linewidth", 1) if isinstance(kwargs_list[idx], dict) else 1,
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
                        alpha=0.3,
                        linewidth=kwargs_list[idx].pop("linewidth", 1) if isinstance(kwargs_list[idx], dict) else 1,
                        **kwargs_list[idx]
                    )[0] if _wants_trail(idx) else None
                    for idx, dat in enumerate(x)
                ]
        else:
            lines = [
                ax.plot(
                    dat[0:1, 0],
                    dat[0:1, 1],
                    dat[0:1, 2],
                    linewidth=kwargs_list[idx].pop("linewidth", 1) if isinstance(kwargs_list[idx], dict) else 1,
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
                        alpha=0.3,
                        linewidth=kwargs_list[idx].pop("linewidth", 1) if isinstance(kwargs_list[idx], dict) else 1,
                        **kwargs_list[idx]
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
                clouds, morph_samples=morph_samples)
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
            (morph_artist,) = ax.plot(
                first_draw[:, 0], first_draw[:, 1], first_draw[:, 2],
                linestyle="None", marker=".", markersize=morph_markersize,
                color=ds_colors[0],
            )
            morph_artist.set_label("_nolegend_")
            if (morph_surface_spec is not None
                    and not morph_surface_spec.get("keep_points", True)):
                morph_artist.set_visible(False)

            morph_state = dict(
                sampled=sampled, dup_masks=dup_masks, colors=ds_colors,
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

        if tail_duration == 0:
            tail_duration = 1
        else:
            tail_duration = int(frame_rate * tail_duration)

        # get line animation
        if style in ["parallel", True]:
            line_ani = animation.FuncAnimation(
                fig,
                update_lines_parallel,
                x[0].shape[0],
                fargs=(
                    x,
                    lines,
                    trail,
                    cube_scale_anim,
                    tail_duration,
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
            line_ani = animation.FuncAnimation(
                fig,
                update_lines_serial,
                frame_rate * duration,
                fargs=(x, lines, cube_scale_anim, rotations, zoom, elev),
                interval=1000 / frame_rate,
                blit=False,
                repeat=False,
            )
        elif style == "spin":
            line_ani = animation.FuncAnimation(
                fig,
                update_lines_spin,
                frame_rate * duration,
                fargs=(x, lines, cube_scale_anim, rotations, zoom, elev),
                interval=1000 / frame_rate,
                blit=False,
                repeat=False,
            )
        elif style == "morph":
            n_morph_datasets = len(morph_state["indices"])
            total_frames = frame_rate * duration
            frame_counts, _, azimuths = _morph.morph_schedule(
                n_morph_datasets, total_frames, rotations, azim)
            morph_state["frame_counts"] = frame_counts
            line_ani = animation.FuncAnimation(
                fig,
                update_morph,
                sum(frame_counts),
                fargs=(morph_state, cube_scale_anim, azimuths, zoom, elev),
                interval=1000 / frame_rate,
                blit=False,
                repeat=False,
            )

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

    if animate in [True, "parallel", "spin", "serial", "morph"]:
        assert (
            x[0].shape[1] == 3
        ), "Animations are currently only supported for 3d plots."

        # animation params
        ani_params = dict(
            tail_duration=tail_duration,
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
                                   elev, azim)
                _hide_no_keep_points(ax.lines, surface)

            # density= (GH #108/#191): subtle KDE density shading, one
            # layer per dataset (or one pooled layer), below the data
            if density is not None:
                _draw_density_3d(ax, data, density, density_colors)

        elif x[0].shape[1] == 2:

            # plot square
            plot_square(ax, **frame_kwargs)

            # set axes
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

            # surface= (GH #109): smooth filled hull outlines, below the data
            if surface is not None:
                _fill_and_draw_2d(ax, data, surface, surface_colors)
                _hide_no_keep_points(ax.lines, surface)

            # density= (GH #108/#191): subtle KDE density shading, one
            # layer per dataset (or one pooled layer), below the data
            if density is not None:
                _draw_density_2d(ax, data, density, density_colors)

        # set line_ani to empty
        line_ani = None

    # remove axes
    ax.set_axis_off()

    # add labels
    add_labels(x, labels, explore=explore)

    # add title
    if title is not None:
        ax.set_title(title)

    # add legend: to the RIGHT of the plot, vertically centered on the
    # box (never overlapping the data)
    if legend is not None:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                  borderaxespad=0.0, frameon=False)

    if size is not None:
        fig.set_size_inches(size)

    if line_ani is not None:
        _make_save_dpi_safe(line_ani)

    return fig, ax, data, line_ani
