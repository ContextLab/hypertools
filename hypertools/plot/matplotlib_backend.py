#!/usr/bin/env python

import itertools

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import matplotlib.animation as animation
import matplotlib.patches as patches
from .._shared.helpers import *
from .meshutil import backface_cull, blinn_phong_colors
from .surface import build_mesh_3d, build_outline_2d, mpl_lighting_kwargs, view_vector


def _resolve_surface_color(spec, fallback_rgb):
    """Base RGB for one dataset's surface: `spec['color']` if given,
    otherwise the dataset's own drawn color (`fallback_rgb`)."""
    return mcolors.to_rgb(spec["color"]) if spec["color"] is not None else fallback_rgb


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


def _mesh_and_draw_3d(ax, points_list, surface, surface_colors, elev, azim,
                      prior_colls=None, quiet=False):
    """Build fresh meshes from the CURRENT per-dataset point windows
    (`points_list`) and delegate to `_shade_and_cull_3d`. Used whenever the
    visible point window changes (static plots; 'parallel'/'serial'
    animation frames)."""
    mesh_list = [
        build_mesh_3d(np.asarray(pts), spec, dataset_label=f" {i}", quiet=quiet)
        if spec is not None else None
        for i, (pts, spec) in enumerate(zip(points_list, surface))
    ]
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
):
    """
    Draws the plot
    """

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
        chemtrails=False,
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

        # zip_longest: marker-only animations have no trail artists (trail
        # is None for those datasets), but head artists still animate
        windows = []
        for line, data, trail in itertools.zip_longest(
                lines, data_lines, trail_lines):

            if trail is not None:
                if (precog and chemtrails) or bullettime:
                    trail.set_data(data[:, 0:2].T)
                    trail.set_3d_properties(data[:, 2])
                elif chemtrails:
                    trail.set_data(data[0 : num - tail_duration + 1, 0:2].T)
                    trail.set_3d_properties(data[0 : num - tail_duration + 1, 2])
                elif precog:
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

    def dispatch_animate(x, ani_params):
        if x[0].shape[1] == 3:
            return animate_plot3D(x, **ani_params)

    def animate_plot3D(
        x,
        tail_duration=2,
        rotations=1,
        zoom=1,
        chemtrails=False,
        frame_rate=30,
        elev=10,
        style="parallel",
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
                    )[0]
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
                    )[0]
                    for idx, dat in enumerate(x)
                ]
        # trails are faint context, not legend-worthy: only the in-focus
        # `lines` should carry legend entries. Otherwise every label appears
        # twice -- once for the moving window, once for its tail. The legend is
        # built once from `lines` (all datasets, created upfront), so it shows
        # the static union of in-focus items and never changes across frames.
        for _trail_line in trail:
            _trail_line.set_label('_nolegend_')

        # surface= (GH #109)
        if surface is not None:
            # keep_points=False: hide (not remove) that dataset's line/trail
            # for the whole animation -- visibility is set once here and
            # persists across every frame update.
            _hide_no_keep_points(lines, surface)
            _hide_no_keep_points(trail, surface)
            if style == "spin":
                # 'spin' keeps the FULL dataset static (only the camera
                # rotates) -- precompute each dataset's mesh once here so
                # update_lines_spin only has to re-shade/re-cull per frame.
                update_lines_spin.meshes = [
                    build_mesh_3d(np.asarray(pts), spec, dataset_label=f" {i}",
                                 quiet=True) if spec is not None else None
                    for i, (pts, spec) in enumerate(zip(x, surface))
                ]

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
                    1,
                    tail_duration,
                    rotations,
                    zoom,
                    chemtrails,
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
                fargs=(x, lines, 1, rotations, zoom, elev),
                interval=1000 / frame_rate,
                blit=False,
                repeat=False,
            )
        elif style == "spin":
            line_ani = animation.FuncAnimation(
                fig,
                update_lines_spin,
                frame_rate * duration,
                fargs=(x, lines, 1, rotations, zoom, elev),
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

    if animate in [True, "parallel", "spin", "serial"]:
        assert (
            x[0].shape[1] == 3
        ), "Animations are currently only supported for 3d plots."

        # animation params
        ani_params = dict(
            tail_duration=tail_duration,
            rotations=rotations,
            zoom=zoom,
            chemtrails=chemtrails,
            frame_rate=frame_rate,
            elev=elev,
            style=animate,
        )

        # dispatch animation
        fig, ax, data, line_ani = dispatch_animate(x, ani_params)

    else:

        # dispatch static
        fig, ax, data = dispatch_static(x, ax)

        # if 3d, plot the cube
        if x[0].shape[1] == 3:

            # set cube scale
            cube_scale = 1

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
                _mesh_and_draw_3d(ax, data, surface, surface_colors, elev, azim)
                _hide_no_keep_points(ax.lines, surface)

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

    return fig, ax, data, line_ani
