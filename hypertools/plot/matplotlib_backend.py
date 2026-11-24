#!/usr/bin/env python
"""Low-level matplotlib drawing for `hypertools.plot` (static and
animated figures, explore mode, trails, and morphs).

The sole entry point is `_draw`, called by `hypertools.plot.plot` after
it has resolved data, formats, colors, and animation settings; everything
else here is private helpers. This module was renamed from ``draw.py`` in
HyperTools 1.0 (``draw.py`` remains as a compatibility shim).
"""
import contextlib
import functools
import itertools
import warnings
import weakref

import matplotlib.artist
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
from .._shared.helpers import UNIT_FRAME_LIMIT, UNIT_FRAME_SCALE
from .._shared.helpers import row_index_x
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


def _label_callout_kwargs(label_alpha, arrowstyle='-', facecolor='white'):
    """`ax.annotate` box and connector styling for an observation label,
    in the colors plotly's annotations use (`plotly_backend`: connector
    rgba(0,0,0,0.6), box edge rgba(0,0,0,0.4), a white box at
    `label_alpha`). Both colors are EXPLICIT: seaborn's whitegrid style
    sets ``patch.edgecolor='w'``, so the defaults drew a white connector
    and a white box edge -- an invisible link that cut white notches
    through the markers (1.1 release review, figure QA). `label_alpha` stays
    the box patch's own ``alpha`` (the documented `label_alpha=` contract),
    which matplotlib applies to its edge too: a dark edge at that opacity
    (0.5 by default, beside plotly's 0.4)."""
    return dict(
        bbox=dict(boxstyle="round,pad=0.5", fc=facecolor, ec='black',
                  lw=0.8, alpha=label_alpha),
        arrowprops=dict(arrowstyle=arrowstyle, connectionstyle="arc3,rad=0",
                        color=(0, 0, 0, 0.6), lw=0.8),
    )


#: where an observation label sits relative to its point, in points: up
#: and to the left, as hypertools has always drawn it
LABEL_OFFSET_PT = 20.0

#: the places a label can take when its box is covered by another
#: label's (`declutter_labels`): the four corners around its point and
#: the four sides, at the usual distance and at two further out
_LABEL_REACHES_PT = (LABEL_OFFSET_PT, 2.25 * LABEL_OFFSET_PT,
                     3.5 * LABEL_OFFSET_PT)
_LABEL_CORNERS = ((-1, 1), (1, 1), (-1, -1), (1, -1))
_LABEL_SIDES = ((0, 1), (0, -1), (-1, 0), (1, 0))

#: the place every label starts from: up and to the left of its point
_LABEL_HOME = (_LABEL_CORNERS[0], LABEL_OFFSET_PT)

#: the clear space, in points, a label is given around its box when it is
#: moved to a new place. A label that is not moved keeps its place until
#: its box actually meets another's, so a label never sits on the border
#: between two places (it would switch on every frame of an animation).
_LABEL_CLEARANCE_PT = 3.0

#: a label that cannot be put clear of the others moves only to a place
#: that covers less than this share of what it covers where it is
_LABEL_MOVE_GAIN = 0.5

#: what covering a legend, a colorbar, a title or a figure text costs a
#: place, against covering the same area of another label
_LABEL_OBSTACLE_WEIGHT = 4.0

#: how many times the labels are gone through on one draw (each time only
#: the labels still covered move)
_LABEL_PASSES = 12

#: above this many visible labels they are left where they are: that many
#: boxes cover the plot whichever way each one points
_LABEL_DECLUTTER_MAX = 300


def _label_box(anchor, size, corner, reach_px, pad_px):
    """The display box (x0, y0, x1, y1), its round frame included, of a
    label of text `size` attached to the point `anchor` on the `corner`
    side (``(-1, 1)`` is up and to the left, ``(0, 1)`` straight up) at
    `reach_px`."""
    (px, py), (w, h), (sx, sy) = anchor, size, corner
    x0 = px + sx * reach_px - (w if sx < 0 else 0.5 * w if sx == 0 else 0.0)
    y0 = py + sy * reach_px - (h if sy < 0 else 0.5 * h if sy == 0 else 0.0)
    return (x0 - pad_px, y0 - pad_px, x0 + w + pad_px, y0 + h + pad_px)


def _covered_area(box, others):
    """The area of `box` (x0, y0, x1, y1) that the boxes `others` (an
    ``(n, 4)`` array) cover, summed over them. `box` may also be an
    ``(m, 4)`` array of boxes: the result is then one area per box."""
    box = np.asarray(box, dtype=float)
    if not len(others):
        return 0.0 if box.ndim == 1 else np.zeros(len(box))
    if box.ndim == 1:
        wide = (np.minimum(box[2], others[:, 2])
                - np.maximum(box[0], others[:, 0]))
        tall = (np.minimum(box[3], others[:, 3])
                - np.maximum(box[1], others[:, 1]))
        area = np.clip(wide, 0, None) * np.clip(tall, 0, None)
        return float(np.nansum(area))
    wide = (np.minimum(box[:, None, 2], others[None, :, 2])
            - np.maximum(box[:, None, 0], others[None, :, 0]))
    tall = (np.minimum(box[:, None, 3], others[None, :, 3])
            - np.maximum(box[:, None, 1], others[None, :, 1]))
    return np.nansum(
        np.clip(wide, 0, None) * np.clip(tall, 0, None), axis=1)


#: the resolution every label is measured at (`_label_size_pt`), and the
#: renderer that measures: hypertools' own figures are made at 100 dpi,
#: so this is what the screen and a default `savefig` draw
_LABEL_METRICS_DPI = 100.0
_LABEL_METRICS_RENDERER = []


def _label_size_pt(label):
    """The width and height of `label`'s text, in points, whatever is
    drawing it.

    Measured with matplotlib's own layout of the text (line breaks, line
    spacing, mathtext) on ONE renderer at ONE resolution
    (`_LABEL_METRICS_DPI`) instead of the renderer of the draw: Agg fits
    glyphs to the pixel grid, so the same text is a slightly different
    size at every dpi (and PDF and SVG measure it a third way), and a
    placement made from those sizes was a different arrangement on the
    screen, in a 300 dpi PNG and in a PDF. At the default 100 dpi the
    size is exactly the one drawn; a text is a little larger there than
    at a higher resolution or in a vector file, so boxes that are clear
    of each other here are clear there."""
    import matplotlib.text as mtext
    # everything the extent of the drawn text depends on: a label turned
    # on its side (`set_rotation(90)`) or switched to plain text
    # (`set_parse_math(False)`) after it was first measured kept its old
    # size, so labels that now covered each other were left where they
    # were (1.2.1 review round 3). A wrapped text is laid out against
    # the figure it is in, so its size is measured every time. (The
    # alignment is not part of it: it moves the text, which is how a
    # label is put at its place, and leaves its size alone.)
    key = (label.get_text(), hash(label.get_fontproperties()),
           label._linespacing, label.get_usetex(), label.get_parse_math(),
           label.get_rotation(), label.get_rotation_mode(),
           label.get_transform_rotates_text())
    if label.get_wrap():
        key = None
    known = getattr(label, '_hyp_size_pt', None)
    if key is not None and known is not None and known[0] == key:
        return known[1]
    if not _LABEL_METRICS_RENDERER:
        from matplotlib.backends.backend_agg import RendererAgg
        _LABEL_METRICS_RENDERER.append(
            RendererAgg(1, 1, _LABEL_METRICS_DPI))
    own_renderer, shown = label._renderer, label.get_visible()
    try:
        # (a hidden text has no extent; visibility is put back below
        # without marking the figure as changed)
        label._visible = True
        box = mtext.Text.get_window_extent(
            label, _LABEL_METRICS_RENDERER[0])
        to_pt = 72.0 / _LABEL_METRICS_DPI
        size = (float(box.width) * to_pt, float(box.height) * to_pt)
    finally:
        label._visible = shown
        label._renderer = own_renderer
    label._hyp_size_pt = (key, size)
    return size


def _label_user_placed(label):
    """Whether `label` is no longer where hypertools put it: moved by the
    reader (2-D labels can be dragged) or by the caller's own code. It is
    theirs from then on."""
    if getattr(label, '_hyp_user_placed', False):
        return True
    ours = getattr(label, '_hyp_offset', None)
    if ours is None:
        return False
    x, y = label.xyann
    if abs(x - ours[0]) > 1e-6 or abs(y - ours[1]) > 1e-6:
        label._hyp_user_placed = True
        return True
    return False


#: how many `plot()` calls are running (`labels_held_while_plotting`)
_PLOTTING = [0]


@contextlib.contextmanager
def labels_held_while_plotting():
    """For the time of one `plot()` call: the labels an axes was just
    given stay at their usual places until the call has finished its
    figure (`reset_label_places` at the end of `plot()` releases them).

    `plot()` draws the figure several times while it lays it out, and
    each of those draws used to arrange the labels for a layout that was
    about to change, only for the arrangement to be undone at the end:
    with 300 labels a quarter of the call's time (1.2.1 review round 2).
    Held, the layout is also measured with the labels where 1.2.0 had
    them. Outside a `plot()` call nothing is ever held, so a figure that
    for any reason left `plot()` unreleased is still arranged when it is
    drawn."""
    _PLOTTING[0] += 1
    try:
        yield
    finally:
        _PLOTTING[0] -= 1


def _label_entries(ax):
    """The ``(annotation, x, y[, z])`` entries of every label hypertools
    has drawn into `ax` and that is still there."""
    return [entry for entry in
            (getattr(ax, '_hyp_labels_and_points', None) or [])
            if entry[0].axes is ax]


def axis_text_boxes_px(ax, renderer=None):
    """The boxes ``(x0, y0, x1, y1)``, in pixels, of every axis label
    and tick label the 3-D axes `ax` draws at its current view: an
    ``(n, 4)`` array (empty for a 2-D axes, whose axis texts lie outside
    the plot, and for a cube that draws none).

    matplotlib decides where these texts go while it draws the axis
    (which edge of the cube a label hangs from changes as the cube
    turns). During a draw of the figure (`renderer` given) the axes has
    just drawn them, so they are read where they are; otherwise the
    three axes alone are drawn onto the figure's renderer first, about
    a hundredth of what drawing the figure costs, which is what lets
    EVERY frame's view of an animation be measured."""
    if not hasattr(ax, 'get_proj') or not getattr(ax, '_axis3don', True):
        return np.zeros((0, 4))
    texts = []
    for axis3d in ax._axis_map.values():
        texts.extend(
            text for text in [axis3d.label] + [
                tick.label1 for tick in axis3d.get_major_ticks()]
            if text.get_visible() and text.get_text().strip())
    if not texts:
        return np.zeros((0, 4))
    if renderer is None:
        renderer = ax.get_figure(root=True)._get_renderer()
        ax.apply_aspect()
        # (what `Axes3D.draw` sets up before it draws its axes)
        ax.M = ax.get_proj()
        ax.invM = np.linalg.inv(ax.M)
        for axis3d in ax._axis_map.values():
            axis3d.draw(renderer)
    boxes = [text.get_window_extent(renderer).extents for text in texts]
    return np.array([box for box in boxes if np.all(np.isfinite(box))],
                    dtype=float).reshape(-1, 4)


def _whole_points(boxes_px, origin, to_pt):
    """Pixel boxes as boxes in points from `origin`, widened to whole
    points."""
    boxes = (np.asarray(boxes_px, dtype=float).reshape(-1, 4)
             - (origin[0], origin[1], origin[0], origin[1])) * to_pt
    boxes[:, :2] = np.floor(boxes[:, :2])
    boxes[:, 2:] = np.ceil(boxes[:, 2:])
    return boxes


def _label_obstacles_pt(ax, renderer, origin, to_pt, axis_text=True):
    """The boxes, in points from the canvas's bottom-left corner, of
    everything in the figure of `ax` that an observation label must not
    be put on: every legend (the axes' own and the figure's, such as the
    one shared legend of a `panels=` grid), every colorbar with its tick
    labels, every panel's title, and every figure text (`suptitle=`,
    `caption=`, a caller's own ``fig.text``). An ``(n, 4)`` array.

    1.2.1 review round 5 added two things. Every OTHER axes of the
    figure (the next panel of a grid, a companion panel): a label that
    reached into the next panel was painted over by that panel's
    background, cut off at its edge. And, with `axis_text`, the axis
    labels and tick labels of a 3-D axes' own cube
    (`axis_text_boxes_px`), which hang where the labels of points near
    the cube's edges want to go.

    1.2.1 review round 4: the labels were only kept off each other, so a
    label moved out of another label's way could land on the legend
    (which the labels of 1.2.0, all at one fixed offset the layout had
    made room for, never did) or on a title.

    Each box is measured where the figure shows it now (no draw is made:
    a legend and a text know their extent from the figure's layout) and
    then widened to whole points, so the small differences between how
    one renderer and another measures the same text almost never change
    which places are free."""
    fig = ax.get_figure(root=True)
    if renderer is None:
        renderer = fig._get_renderer()
    found = []

    def _add(artist, tight=False):
        if artist is None or not artist.get_visible():
            return
        try:
            box = (artist.get_tightbbox(renderer) if tight
                   else artist.get_window_extent(renderer))
        except Exception:  # noqa: BLE001 - an artist that cannot be measured is no obstacle
            return
        if box is None or not np.all(np.isfinite(box.extents)) \
                or box.width <= 0 or box.height <= 0:
            return
        found.append(box.extents)

    for legend in fig.legends:
        _add(legend)
    for text in fig.texts:
        if text.get_text().strip():
            _add(text)
    for other in fig.axes:
        if not other.get_visible():
            continue
        _add(other.get_legend())
        for title in (other.title, getattr(other, '_left_title', None),
                      getattr(other, '_right_title', None)):
            if title is not None and title.get_text().strip():
                _add(title)
        if other is not ax and hasattr(other, '_colorbar'):
            _add(other, tight=True)
        elif other is not ax:
            _add(other)
    if axis_text:
        found.extend(axis_text_boxes_px(ax, renderer))
    if not found:
        return np.zeros((0, 4))
    return _whole_points(found, origin, to_pt)


def _label_geometry(ax, labels):
    """What the labels `labels` of `ax` are placed from, all in POINTS
    from the bottom-left corner of the canvas the figure was made for:
    ``(origin, to_pt, canvas, anchors)``. `origin` is that corner in
    display pixels, `to_pt` the points in a pixel, `canvas` the box
    ``(0, 0, width, height)`` and `anchors` the ``(n, 2)`` positions of
    the labels' points as the axes shows them now."""
    fig = ax.get_figure(root=True)
    to_pt = 72.0 / fig.dpi
    # the canvas the figure was made for: the unit square of the figure
    # (a `bbox_inches='tight'` save moves the origin and crops, and the
    # labels must not be arranged for the crop)
    (left, bottom), (right, top) = fig.transFigure.transform(
        [(0.0, 0.0), (1.0, 1.0)])
    canvas = (0.0, 0.0, round((right - left) * to_pt, 4),
              round((top - bottom) * to_pt, 4))
    if not labels:
        return (left, bottom), to_pt, canvas, np.zeros((0, 2))
    anchors = np.round(
        (ax.transData.transform(np.array([label.xy for label in labels],
                                         dtype=float))
         - (left, bottom)) * to_pt, 4)
    return (left, bottom), to_pt, canvas, anchors


def _label_pads_pt(labels):
    """How far each label's round frame reaches beyond its text, in
    points."""
    return np.array([
        (getattr(frame.get_boxstyle(), 'pad', 0.0) if frame else 0.0)
        * label.get_size()
        for label, frame in ((label, label.get_bbox_patch())
                             for label in labels)], dtype=float)


def _settle_axes_for_labels(ax):
    """With no draw going on, bring `ax` to the state a draw would give
    it (pending autoscaling, the box an aspect ratio leaves), so its
    data transform is the one the picture is drawn with."""
    ax.get_xlim(), ax.get_ylim()
    locator = ax.get_axes_locator()
    ax.apply_aspect(locator(ax, None) if locator else None)


def _project_label_anchors(ax, entries):
    """Point every visible 3-D label of `entries` at its point where the
    camera and the box zoom of `ax` put it NOW (a projection kept from an
    earlier layout or zoom left the arrow pointing at nothing)."""
    if not hasattr(ax, 'get_proj'):
        return
    from mpl_toolkits.mplot3d import proj3d
    proj = ax.get_proj()
    for entry in entries:
        if len(entry) == 4 and entry[0].get_visible():
            x2, y2, _ = proj3d.proj_transform(
                entry[1], entry[2], entry[3], proj)
            entry[0].xy = (x2, y2)


#: every place a label can take, nearest first, and each place's reach
_LABEL_CANDIDATES = tuple((corner, reach) for reach in _LABEL_REACHES_PT
                          for corner in _LABEL_CORNERS + _LABEL_SIDES)


def _arrange_labels(places, left, anchors, sizes, pads, canvas, obstacles,
                    strict=False):
    """THE placement rule of observation labels, for one view: given
    where each label hangs now (`places`, a list of ``(corner, reach)``)
    and the place each one last left (`left`), move the labels that are
    covered and leave the others. Both lists are changed in place.
    Returns whether the labels came to rest.

    `anchors`, `sizes` and `pads` are ``(n, 2)``, ``(n, 2)`` and ``(n,)``
    arrays in points (`_label_geometry`, `_label_size_pt`,
    `_label_pads_pt`), `canvas` the canvas box and `obstacles` the
    ``(m, 4)`` boxes no label may be put on (`_label_obstacles_pt`).

    A label stays where it is unless an earlier label's box is on its
    own, it lies on an obstacle, or a place it was moved to has left the
    canvas. A label that must move takes, of the places around its point
    (`_LABEL_CANDIDATES`), the one that covers least; of equals the one
    at the shortest distance from its point, then not the place it last
    left, then the one its box travels least to reach. With no clear
    place it moves only to one that at least halves what it covers
    (`_LABEL_MOVE_GAIN`).

    `declutter_labels` calls it for a static figure, from the places
    the labels hang at. `solve_label_tracks` calls it to place a single
    frame of an animation as a still would be placed, with `strict`:
    a label at its usual place that reaches off the canvas moves too
    (in a static figure the layout has made room for the usual places,
    and such a label stays where it has always been)."""
    count = len(places)
    gap = _LABEL_CLEARANCE_PT
    candidates = _LABEL_CANDIDATES
    reaches = np.array([reach for _corner, reach in candidates])
    finite = np.all(np.isfinite(anchors), axis=1)
    boxes = np.full((count, 4), np.nan)
    for k in range(count):
        if finite[k]:
            corner, reach = places[k]
            boxes[k] = _label_box(anchors[k], sizes[k], corner, reach,
                                  pads[k])

    def _off_canvas(box):
        return (box[0] < canvas[0] or box[1] < canvas[1]
                or box[2] > canvas[2] or box[3] > canvas[3])

    def _area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def _may_stay(k, here, on_obstacle):
        return (_covered_area(here, boxes[:k]) <= 0.0
                and on_obstacle <= 0.0
                and ((places[k] == _LABEL_HOME and not strict)
                     or not _off_canvas(here)))

    def _all_at_rest():
        """Whether no label has a reason to move (every label looked at
        in one go: most views of an animation leave every label where
        it is)."""
        with np.errstate(invalid='ignore'):
            wide = (np.minimum(boxes[:, None, 2], boxes[None, :, 2])
                    - np.maximum(boxes[:, None, 0], boxes[None, :, 0]))
            tall = (np.minimum(boxes[:, None, 3], boxes[None, :, 3])
                    - np.maximum(boxes[:, None, 1], boxes[None, :, 1]))
            meet = (wide > 0) & (tall > 0)
            if np.tril(meet, -1).any():
                return False
            if len(obstacles) and np.any(
                    _covered_area(boxes, obstacles) > 0.0):
                return False
            off = ((boxes[:, 0] < canvas[0]) | (boxes[:, 1] < canvas[1])
                   | (boxes[:, 2] > canvas[2]) | (boxes[:, 3] > canvas[3]))
        if not strict:
            off &= np.array([place != _LABEL_HOME for place in places])
        return not off.any()

    if _all_at_rest():
        return True

    #: per label, the boxes of its candidate places with their clearance
    #: (worked out when the label first has to move)
    rooms = {}

    # (a label that moves frees the place it leaves and takes another, so
    # the labels are gone through until none of them moves. A label only
    # ever moves for a label given before it, so this ends.)
    settled = False
    #: how many moves have been made, and per label that count when it
    #: was last looked at and left where it is: nothing has changed for
    #: it since, so looking again would find the same
    moves, stayed_at = 0, {}
    for _ in range(_LABEL_PASSES):
        moved = False
        for k in range(count):
            if not finite[k] or stayed_at.get(k) == moves:
                continue
            anchor, size, pad = anchors[k], sizes[k], pads[k]
            here = boxes[k].copy()
            place = places[k]
            # it stays unless an earlier label's box is on its own, it is
            # on a legend, a colorbar or a title (`obstacles`), or a
            # place it was moved to has since left the canvas (in its
            # usual place a static figure's label at the canvas edge
            # stays where it has always been)
            on_obstacle = _covered_area(here, obstacles)
            if _may_stay(k, here, on_obstacle):
                stayed_at[k] = moves
                continue
            if k not in rooms:
                roomy = np.array([
                    _label_box(anchor, size, corner, reach, pad + gap)
                    for corner, reach in candidates])
                rooms[k] = (roomy, (
                    (roomy[:, 0] < canvas[0]) | (roomy[:, 1] < canvas[1])
                    | (roomy[:, 2] > canvas[2]) | (roomy[:, 3] > canvas[3]))
                    * (roomy[:, 2] - roomy[:, 0])
                    * (roomy[:, 3] - roomy[:, 1]))
            roomy, off_cost = rooms[k]
            # (every label but this one: its own row is blanked for the
            # sums, which skip NaN)
            boxes[k] = np.nan
            # a place off the canvas, or on a legend, a colorbar or a
            # title, only as a last resort: it costs the label's whole
            # box on top of what it covers there, so a label that lies on
            # another label does not get out of its way onto a title
            # (where every label hangs at first, the layout has made
            # room for it)
            cost_here = _covered_area(here, boxes) + (
                _area(here) if _off_canvas(here) else 0.0) + (
                _LABEL_OBSTACLE_WEIGHT * on_obstacle + _area(here)
                if on_obstacle > 0.0 else 0.0)
            on_obstacles = _covered_area(roomy, obstacles)
            costs = (_covered_area(roomy, boxes) + off_cost
                     + _LABEL_OBSTACLE_WEIGHT * on_obstacles
                     + (on_obstacles > 0.0) * (roomy[:, 2] - roomy[:, 0])
                     * (roomy[:, 3] - roomy[:, 1]))
            boxes[k] = here
            # a short arrow before a long one, and at that distance the
            # smallest change: the place the box travels least to reach
            change = np.hypot(
                0.5 * (roomy[:, 0] + roomy[:, 2]) - 0.5 * (here[0] + here[2]),
                0.5 * (roomy[:, 1] + roomy[:, 3]) - 0.5 * (here[1] + here[3]))
            # (the place it last left comes after the others at that
            # distance: a label pushed out of one place and then out of
            # the next does not hop straight back)
            back = np.array([one == left[k] for one in candidates])
            open_places = np.array([one != place for one in candidates])
            # the best place: least covered, then nearest reach, then
            # not the place just left, then the smallest change (the
            # first of equals, in the order of `candidates`)
            order = np.lexsort((change, back, reaches, costs))
            best = int(order[open_places[order]][0])
            cost = float(costs[best])
            if cost > 0.0 and cost >= _LABEL_MOVE_GAIN * cost_here:
                stayed_at[k] = moves
                continue
            corner, reach = candidates[best]
            left[k] = place
            places[k] = (corner, reach)
            boxes[k] = _label_box(anchor, size, corner, reach, pad)
            moved = True
            moves += 1
        if not moved:
            settled = True
            break
    return settled


def declutter_labels(ax, renderer=None):
    """Keep the observation labels of `ax` (`labels=`) off each other.

    Every label starts up and to the left of its point, so the labels of
    two points that are close on screen were drawn on top of each other
    (GH #297: "Subject 1" over "Subject 2" in `examples/plot_labels.py`).

    The labels are kept off the figure's furniture as well: its
    legends, colorbars, titles, suptitle and caption
    (`_label_obstacles_pt`; 1.2.1 review round 4). A label lying on one
    of them moves like a label lying on an earlier label, a place that
    covers one of them is taken only as a last resort (it costs the
    label's whole box and `_LABEL_OBSTACLE_WEIGHT` times what it covers
    there, as a place off the canvas costs the whole box), and this
    also holds for an axes with a single label.

    The placement (`_arrange_labels`) is STABLE: a label stays where it
    is for as long as its box is clear of the labels given before it, so
    a label that covers nothing is never touched and a figure whose
    labels were clear of each other is drawn exactly as before. Only a
    label whose box is covered by an earlier label's moves (or one that
    was moved and has since left the canvas), and it moves as little as
    it can: of the places around its point (four corners and four sides,
    at three distances) that are clear of every other label and on the
    canvas, it takes one at the shortest distance from its point, and of
    those the one nearest to where it is; with no clear place, a place
    that at least halves what it covers, or else it stays. A moved label
    is given `_LABEL_CLEARANCE_PT` of room and does not go back when its
    old place becomes free, so as a 3-D plot is turned by hand a label
    changes place only when the points themselves have moved onto each
    other.

    What a draw does therefore depends on where the labels were, which
    suits a figure the reader turns by hand and nothing else. So the
    state is pinned wherever a picture must be reproducible:
    `reset_label_places` puts the labels back at the start when `plot()`
    has finished a STATIC figure, so the same call always saves the same
    picture; and an ANIMATED plot's labels follow a table worked out for
    the whole animation, frame 0 to the last frame in order, by the same
    rule (`solve_label_tracks`): frame `i` shows row `i` of it, so a
    frame is the same picture whatever was drawn before it, and every
    frame's labels are placed for that frame's own view.

    Everything is worked out in POINTS, from the figure itself and not
    from the renderer of the draw (1.2.1 review round 2): the anchors
    and the canvas are the figure's own geometry divided by its dpi, and
    a label's size is `_label_size_pt`. The same figure is therefore
    arranged the same way on the screen, in a PNG at any dpi, in a PDF
    and in an SVG; only a change of the figure itself (its size, its
    layout, the camera, the labels) can change the arrangement. For the
    same reason a draw of a figure in which nothing has moved since the
    last one does not place anything again (the last problem and its
    answer are kept on the axes), and the labels are gone through until
    none of them moves, so a second draw has nothing left to do.

    It runs at draw time (`_LabelDeclutter`), which is the one path for a
    static figure, a figure the reader rotates, and a frame of an
    animation. The anchor of a 3-D label is projected here as well, with
    the camera and the box zoom of this very draw.
    """
    entries = _label_entries(ax)
    if not entries:
        return
    if renderer is None:
        _settle_axes_for_labels(ax)
    _project_label_anchors(ax, entries)
    if label_tracks(ax) is not None:
        # an animation's labels: the row of its table for the frame the
        # axes shows (`solve_label_tracks`)
        _apply_label_tracks(ax)
        return
    if renderer is not None and _PLOTTING[0] and getattr(
            ax, '_hyp_labels_held', False):
        # `plot()` is still laying the figure out
        # (`labels_held_while_plotting`)
        return
    # (a hidden label keeps its place and is no obstacle while hidden: a
    # 3-D plot's labels are hidden while the reader drags it round, and
    # they come back where they were unless their points have met)
    # (and a label an animation's plan placed stays where the plan had
    # it when the animation is gone: a figure read back from a pickle,
    # a copy of the figure)
    labels = [entry[0] for entry in entries
              if entry[0].get_visible() and not _label_user_placed(entry[0])
              and not _label_planned(entry[0])]
    if len(labels) > _LABEL_DECLUTTER_MAX:
        for label in labels:
            _send_label_home(label)
        return
    if not labels:
        return
    origin, to_pt, canvas, anchors = _label_geometry(ax, labels)
    sizes = np.array([_label_size_pt(label) for label in labels])
    pads = _label_pads_pt(labels)
    # what no label may be put on: legends, colorbars, titles, the
    # figure's suptitle and caption (1.2.1 review round 4)
    obstacles = _label_obstacles_pt(ax, renderer, origin, to_pt)
    if len(labels) < 2 and not len(obstacles):
        return
    # the same problem as on the last draw: nothing to work out again
    problem = (tuple(map(id, labels)), anchors.tobytes(), sizes.tobytes(),
               pads.tobytes(), canvas, obstacles.tobytes())
    start = tuple((_label_place(label),
                   getattr(label, '_hyp_place_left', None))
                  for label in labels)
    known = getattr(ax, '_hyp_label_answer', None)
    if known is not None and known['problem'] == problem:
        if start == known['end'] and known['settled']:
            return
        if start == known['start']:
            for label, (place, left_place) in zip(labels, known['end']):
                _set_label_place(label, *place)
                label._hyp_place_left = left_place
            return
    places = [place for place, _left in start]
    left = [left_place for _place, left_place in start]
    settled = _arrange_labels(places, left, anchors, sizes, pads, canvas,
                              obstacles)
    for label, was, place, left_place in zip(labels, start, places, left):
        if place != was[0]:
            label._hyp_place_left = left_place
            _set_label_place(label, *place)
    ax._hyp_label_answer = {
        'problem': problem, 'start': start, 'settled': settled,
        'end': tuple((_label_place(label),
                      getattr(label, '_hyp_place_left', None))
                     for label in labels)}


def _label_place(label):
    """Where `label` hangs: ``(corner, reach in points)``."""
    return getattr(label, '_hyp_place', _LABEL_HOME)


def _label_planned(label):
    """Whether `label` is where an animation's plan put it
    (`_set_label_offset`): anywhere between two named places, so it has
    no place of its own for a static figure's rule to start from."""
    return getattr(label, '_hyp_place', _LABEL_HOME) is None


def _send_label_home(label):
    """Put `label` back where every label starts, as if never moved."""
    label._hyp_place_left = None
    _set_label_place(label, *_LABEL_HOME)


def _set_label_place(label, corner, reach_pt):
    """Hang `label` on the `corner` side of its point, `reach_pt` away."""
    if _label_place(label) == (corner, reach_pt) and hasattr(
            label, '_hyp_place'):
        return
    sx, sy = corner
    label.set_position((sx * reach_pt, sy * reach_pt))
    label.set_ha('right' if sx < 0 else 'left' if sx > 0 else 'center')
    label.set_va('bottom' if sy > 0 else 'top' if sy < 0 else 'center')
    label._hyp_offset = (sx * reach_pt, sy * reach_pt)
    label._hyp_place = (corner, reach_pt)


def reset_label_places(target):
    """Put the observation labels of `target` (an axes, or a figure: all
    of its axes) back up and to the left of their points, where
    `declutter_labels` starts from on the next draw. A label the reader
    or the caller moved is left alone, and so are an animation's labels,
    which follow their table (`solve_label_tracks`)."""
    axes = [target] if hasattr(target, 'transData') else list(target.axes)
    for ax in axes:
        if getattr(ax, '_hyp_labels_held', False):
            # the figure is finished: its next draw arranges the labels
            ax._hyp_labels_held = False
            ax.stale = True
        if label_tracks(ax) is not None:
            continue
        for entry in _label_entries(ax):
            if not _label_user_placed(entry[0]) \
                    and not _label_planned(entry[0]):
                _send_label_home(entry[0])


#: the ``(label, x, y[, z])`` entries of the most recently annotated
#: plot (`_publish_labels`); each label is a weak proxy, usable while
#: its figure lives
labels_and_points = []


def _publish_labels(entries):
    """Show `entries`, one call's ``(label, x, y[, z])`` list, as the
    module's `labels_and_points`, without keeping the labels alive."""
    global labels_and_points
    labels_and_points = [(weakref.proxy(entry[0]),) + tuple(entry[1:])
                         for entry in entries]


class _LabelState:
    """What an animated axes' labels follow: `frames`, how to show any
    frame (`track_labels`; the frame function is held WEAKLY: it
    belongs to the animation, and refers to the axes); `tracks`, the
    table worked out from it (`solve_label_tracks`); `frame`, the frame
    the axes shows; `solving`, set while the table is being worked out;
    and, for `_apply_label_tracks`, how many frames have been shown
    (`shows`), at which of them the table was last worked out again
    because the figure had changed (`solved_at`), on how many frames
    in a row it has changed (`streak`), and whether a change is still
    waiting to be planned for (`pending`)."""

    def __init__(self):
        self.frames = self.tracks = None
        self.frame, self.solving = 0, False
        self.shows, self.solved_at, self.pending = 0, -10, False
        self.streak = 0


#: per animated axes, its `_LabelState`. Kept here and not on the axes:
#: a figure is pickled and copied without it (a figure read back from a
#: pickle has no animation to follow; its labels stay where the pickled
#: frame had them, `_label_planned`). Nothing in a state refers to its
#: axes strongly (the labels and the frame function are weak
#: references), so a closed figure and its table are freed.
_LABEL_STATES = weakref.WeakKeyDictionary()


def _label_state(ax):
    """The `_LabelState` of `ax` (a new, empty one if it has none)."""
    state = _LABEL_STATES.get(ax)
    if state is None:
        state = _LABEL_STATES[ax] = _LabelState()
    return state


def label_tracks(ax):
    """The table the labels of the animated axes `ax` follow
    (`solve_label_tracks`), or None for an axes whose labels are placed
    when it is drawn (every static figure)."""
    state = _LABEL_STATES.get(ax)
    return None if state is None else state.tracks


def tracks_labels(ax):
    """Whether `ax` holds an animation's labels (`track_labels`)."""
    state = _LABEL_STATES.get(ax)
    return state is not None and state.frames is not None


def forget_label_tracks(ax):
    """Drop the table of `ax` and what it was made from: its labels are
    placed when it is drawn from now on."""
    _LABEL_STATES.pop(ax, None)


#: how long a label of an animation takes to go from one place to the
#: next, in seconds (`_ease_label_tracks`), and the fewest and the most
#: frames that may take
_LABEL_EASE_SECONDS = 0.35
_LABEL_EASE_FRAMES = (5, 15)

#: the places a label of an ANIMATION can take: the eight directions of
#: a static figure's label and, further out than the usual distance, the
#: eight between them, at more distances (an animation's label is placed
#: for many views at once, and the blank canvas round the cube is where
#: a label can stay through all of them)
_LABEL_TRACK_REACHES_PT = (LABEL_OFFSET_PT, 2.25 * LABEL_OFFSET_PT,
                           3.5 * LABEL_OFFSET_PT, 5.0 * LABEL_OFFSET_PT,
                           6.5 * LABEL_OFFSET_PT)
_LABEL_TRACK_BETWEEN = tuple(
    (round(float(np.cos(turn)), 6), round(float(np.sin(turn)), 6))
    for turn in np.deg2rad(22.5 + 45.0 * np.arange(8)))
_LABEL_TRACK_CANDIDATES = tuple(
    (corner, reach) for reach in _LABEL_TRACK_REACHES_PT
    for corner in _LABEL_CORNERS + _LABEL_SIDES
    + (_LABEL_TRACK_BETWEEN if reach > LABEL_OFFSET_PT else ()))

#: what a frame at a place costs an animation's label
#: (`_plan_label_places`), in the units of `_LABEL_TRACK_SWITCH`, one
#: change of place: lying off the canvas or on static content, lying on
#: another label (each plus the covered share of the box), and, per
#: second, hanging one usual distance further out than usual
_LABEL_TRACK_OFF = 1e7
_LABEL_TRACK_COVERED = 1e3
_LABEL_TRACK_FAR_PER_SECOND = 0.4
_LABEL_TRACK_SWITCH = 3.0
#: and what each usual distance a change of place carries the label over
#: adds to it
_LABEL_TRACK_MOVE = 0.5
#: the longest way, in points, one change of place may carry a label
_LABEL_TRACK_MOVE_MAX_PT = 240.0
#: per second, what pointing straight into the cloud of labeled points
#: costs a label (its leader line then crosses the cloud and the other
#: labels' lines); pointing straight out of it costs nothing
_LABEL_TRACK_INWARD_PER_SECOND = 1.0

#: for how many glides' worth of frames a label of an animation stays at
#: a place it has moved to before it may move again: one glide to get
#: there, and then two of standing still, so no label goes out and
#: straight back
_LABEL_REST_EASES = 3

#: the clear space, in points, a label of an animation is given round its
#: box at a place it rests at (less than a static figure's
#: `_LABEL_CLEARANCE_PT`: a planned label cannot switch back and forth on
#: the border between two places, and room is scarcer over many views)
_LABEL_TRACK_CLEARANCE_PT = 1.0

#: how many times a label of an animation is planned again because a
#: glide of its plan crossed another label
_LABEL_TRACK_REPLANS = 4

#: how many times the labels that are not clear on every frame, and the
#: labels they lie on, are planned again among all the others
_LABEL_TRACK_ROUNDS = 3

#: how many labels x frames of an animation are planned frame by frame.
#: A longer animation is planned at evenly spaced frames, that many
#: over its labels and at least `_LABEL_TRACK_KEYS_MIN`, and its labels
#: move evenly from one planned frame to the next: the plan then takes
#: no longer, and its table is no larger (28 bytes a label and planned
#: frame: about 0.1 MB at the budget, 0.4 MB for 300 labels), however
#: long the animation is
_LABEL_TRACK_BUDGET = 3600
_LABEL_TRACK_KEYS_MIN = 40


def _label_center_offsets(place_index, sizes, candidates=None):
    """Where the MIDDLE of each label's text is, in points from its
    point, at the places `place_index` (indices into `candidates`, by
    default a static figure's `_LABEL_CANDIDATES`; any shape ending in
    the labels' axis) for labels of the text sizes `sizes` ``(n, 2)``."""
    candidates = _LABEL_CANDIDATES if candidates is None else candidates
    corners = np.array([corner for corner, _reach in candidates],
                       dtype=float)
    reaches = np.array([reach for _corner, reach in candidates])
    return corners[place_index] * (reaches[place_index][..., None]
                                   + 0.5 * sizes)


def _boxes_meet(boxes, others):
    """Per box of `boxes` ``(..., 4)``: how many boxes of `others`
    (``(m, 4)``, or ``(..., m, 4)`` with a set of boxes for each leading
    index of `boxes`) it overlaps, plus the share of its own area they cover
    (0.0 for a box that is clear; NaN boxes meet nothing)."""
    boxes = np.asarray(boxes, dtype=float)
    if not np.shape(others)[-2]:
        return np.zeros(boxes.shape[:-1])
    with np.errstate(invalid='ignore', divide='ignore'):
        wide = (np.minimum(boxes[..., None, 2], others[..., 2])
                - np.maximum(boxes[..., None, 0], others[..., 0]))
        tall = (np.minimum(boxes[..., None, 3], others[..., 3])
                - np.maximum(boxes[..., None, 1], others[..., 1]))
        hit = (wide > 0) & (tall > 0)
        area = np.where(hit, wide * tall, 0.0).sum(axis=-1)
        own = ((boxes[..., 2] - boxes[..., 0])
               * (boxes[..., 3] - boxes[..., 1]))
        return hit.sum(axis=-1) + np.where(own > 0, area / own, 0.0)


def _cheapest_places(cost, approach, move, rest, bars=None, start=None,
                     end=None):
    """The cheapest way through the frames for one label: per frame the
    index of the place it hangs at, minimizing the sum of `cost`
    ``(frames, places)`` over the frames plus, for every change of
    place, ``move[from, to]`` and ``approach[frame, to]`` (what gliding
    into the place so as to be there at `frame` costs), with at least
    `rest` frames spent at a place before the next change. `bars`, per
    frame None or a ``(places, places)`` array, is added to `move` for a
    change that arrives on that frame. `start` and `end` fix the first
    and the last frame's place. Exact (dynamic programming over the
    place and the frames spent there)."""
    n_frames, n_places = cost.shape
    rest = max(int(rest), 1)
    value = np.full((n_places, rest + 1), np.inf)
    if start is None:
        value[:, rest] = 0.0                # (free to move from the start)
    else:
        value[start, 0] = 0.0
    value += cost[0][:, None]
    came_from = np.zeros((n_frames, n_places), dtype=int)
    stayed_long = np.zeros((n_frames, n_places), dtype=bool)
    columns = np.arange(n_places)
    # (a place far dearer to leave than the cheapest cannot be the one
    # a change of place starts from: only the others are looked at)
    spread = float(np.ptp(move[np.isfinite(move)])) \
        if np.isfinite(move).any() else 0.0
    for frame in range(1, n_frames):
        ready = value[:, rest]
        barred = bars is not None and bars[frame] is not None
        within = spread + (float(bars[frame].max()) if barred else 0.0)
        lowest = ready.min()
        if np.isfinite(lowest):
            few = np.flatnonzero(ready <= lowest + within + 1e-9)
        else:
            few = columns[:1]
        through = ready[few, None] + move[few]
        if barred:
            through = through + bars[frame][few]
        best = np.argmin(through, axis=0)
        through = through[best, columns]
        source = few[best]
        new = np.full_like(value, np.inf)
        new[:, 1:rest] = value[:, 0:rest - 1]
        stayed_long[frame] = value[:, rest] <= value[:, rest - 1]
        new[:, rest] = np.minimum(value[:, rest], value[:, rest - 1])
        new[:, 0] = through + approach[frame]
        came_from[frame] = source
        value = new + cost[frame][:, None]
    if end is None:
        place, age = np.unravel_index(np.argmin(value), value.shape)
    else:
        place, age = end, int(np.argmin(value[end]))
    places = np.zeros(n_frames, dtype=int)
    for frame in range(n_frames - 1, -1, -1):
        places[frame] = place
        if frame == 0:
            break
        if age == 0:
            place, age = came_from[frame][place], rest
        elif age == rest:
            age = rest if stayed_long[frame][place] else rest - 1
        else:
            age -= 1
    return places


def _plan_label_places(hard, soft, move, taps, rest, loops, bars=None,
                       passing=None):
    """The places one label of an animation hangs at over one stretch of
    frames it is shown for: `hard` ``(frames, places)`` is what lying off
    the canvas, on static content or on a label placed before costs at
    each place on each frame, `soft` ``(places,)`` what hanging far out
    costs per frame, ``move[from, to]`` what a change of place costs.
    The cheapest plan (`_cheapest_places`) in which the label spends
    at least `rest` frames at a place it moved to, and in which a place
    is also clear while the label glides into it (a glide takes `taps`
    frames).

    For an animation that loops (`loops`) the plan is made round the
    loop: first through three rounds of it, to find the longest stay of
    the middle round; then once more starting and ending in the middle
    of that stay, so the last frame leads into the first with no change
    of place.

    `passing` ``(frames, places)`` is what a place costs on a frame the
    label is still on its way to it (by default `hard`;
    `solve_label_tracks` counts the other labels there and not static
    content, which the label is not on until it arrives).

    `bars`, per frame None or a ``(places, places)`` array, is added to
    what a change of place that arrives on that frame costs
    (`solve_label_tracks` bars a change whose glide was found to cross
    another label)."""
    n_frames = len(hard)
    cost = hard + soft
    rest = max(1, min(int(rest), n_frames // 3))
    if n_frames < 2:
        return np.argmin(cost, axis=1)
    bars = [None] * n_frames if bars is None else list(bars)
    passing = hard if passing is None else passing

    def _approach(table, wrap):
        """Per frame and place: the hard cost of the place over the
        `taps` frames before (the label is on its way there)."""
        total = np.zeros_like(table)
        for back in range(1, taps):
            moved = np.roll(table, back, axis=0)
            if not wrap:
                moved[:back] = 0.0
            total += moved
        return total

    if not loops:
        return _cheapest_places(cost, _approach(passing, False), move, rest,
                                bars)
    thrice = _cheapest_places(
        np.tile(cost, (3, 1)), _approach(np.tile(passing, (3, 1)), True),
        move, rest, bars * 3)
    middle = thrice[n_frames:2 * n_frames]
    changes = np.flatnonzero(middle != np.roll(middle, 1))
    if not len(changes):
        return middle
    # the longest stay of the round, and the frame in the middle of it
    lengths = np.diff(np.concatenate([changes, [changes[0] + n_frames]]))
    longest = int(np.argmax(lengths))
    turn = int(changes[longest] + lengths[longest] // 2) % n_frames
    place = int(middle[turn])
    plan = _cheapest_places(
        np.roll(cost, -turn, axis=0),
        np.roll(_approach(passing, True), -turn, axis=0), move, rest,
        bars[turn:] + bars[:turn], start=place, end=place)
    return np.roll(plan, turn)


def _ease_label_tracks(targets, shown, taps, periodic):
    """Turn the places an animation's labels take frame by frame
    (`targets`, ``(frames, labels, 2)`` offsets that change in steps)
    into the offsets they are drawn at: every step becomes a glide over
    the `taps` frames BEFORE the frame the label is at its new place
    (the frames to come are known, so the label is out of the way when
    the other arrives), always toward the new place (a weighted mean of
    the places around each frame, with weights that are never negative,
    so one step never overshoots or turns back).

    Each stretch of frames a label is shown for is eased on its own: a
    label that appears is at its place from its first frame. In an
    animation that loops without a seam (`periodic`) a label shown on
    every frame is eased round the loop, last frame into first."""
    from scipy.ndimage import convolve1d
    weights = np.hanning(taps + 2)[1:-1]
    weights = weights / weights.sum()
    # (`origin`: the glide is over at the frame the label had to move at)
    half = taps // 2
    eased = np.array(targets, dtype=float)
    always = shown.all(axis=0)
    if always.any():
        eased[:, always] = convolve1d(
            targets[:, always], weights, axis=0, origin=half,
            mode='wrap' if periodic else 'nearest')
    for k in np.flatnonzero(~always & shown.any(axis=0)):
        edges = np.flatnonzero(np.diff(np.concatenate(
            [[False], shown[:, k], [False]]).astype(int)))
        for first, end in zip(edges[::2], edges[1::2]):
            if end - first > half:
                eased[first:end, k] = convolve1d(
                    targets[first:end, k], weights, axis=0, origin=half,
                    mode='nearest')
    eased[~shown] = np.nan
    return eased


def _labels_loop(anchors, shown):
    """Whether an animation's last frame leads back into its first
    without a seam, as far as its labels can tell: the same labels are
    shown on both, and their points are no further apart between the two
    than between other neighboring frames (a spin of whole turns; not a
    reveal, whose last frame shows everything and whose first shows
    one point)."""
    if len(anchors) < 3 or not np.array_equal(shown[0], shown[-1]):
        return False
    if not shown[0].any():
        return True
    with np.errstate(invalid='ignore'):
        steps = np.hypot(*np.moveaxis(np.diff(anchors, axis=0), -1, 0))
        seam = np.hypot(*np.moveaxis(anchors[0] - anchors[-1], -1, 0))
    steps = np.where(shown[1:] & shown[:-1], steps, np.nan)
    if np.all(np.isnan(steps)):
        return False
    usual = np.nanmax(steps)
    return bool(np.nanmax(np.where(shown[0], seam, np.nan))
                <= 1.5 * usual + 1e-6)


def solve_label_tracks(ax, _independent=False):
    """Work out where every observation label of the ANIMATED axes `ax`
    is drawn on every frame, and keep the answer as a table (one row per
    frame) that the frames then show (`_apply_label_tracks`).

    The rule (owner decision 2026-11-23): on every frame a label is
    attached to its point where that frame shows it and has correct
    positioning (off the other labels and off legends, colorbars,
    titles, suptitle and caption, on the canvas); it may shift from
    frame to frame but must not wobble back and forth.

    Every frame is known before any is drawn, so the labels are not
    placed frame after frame: each label is given a PLAN for the whole
    animation. The labels are gone through in the order they were given
    (as on a static figure, an earlier label never gives way to a later
    one). For each, every place round its point (`_LABEL_TRACK_CANDIDATES`,
    measured with the geometry of a static figure's label: `_label_geometry`,
    `_label_size_pt`, `_label_obstacles_pt`) is priced on every frame:
    off the canvas or on static content (the cube's own turning axis
    labels included) costs most, lying on a label planned before, as
    that label is DRAWN on that frame, next, and hanging further from
    the point a little. The label takes the cheapest path through the
    frames (`_plan_label_places`) in which it rests
    `_LABEL_REST_EASES` glides at a place before it moves again, so it
    stays at one place for as long as there is one that is clear, moves
    BEFORE another label arrives, and never goes out and straight back.
    Changes of place are then turned into glides (`_ease_label_tracks`);
    a change whose glide would cross another label is barred on the
    frames it would, and the label planned again (up to
    `_LABEL_TRACK_REPLANS` times with 30 labels, more with fewer, fewer
    with more). Labels still not clear on some frame, and the labels
    they lie on, are then planned again among ALL the others
    (`_LABEL_TRACK_ROUNDS` times at most, up to 40 labels; once up to
    80; not above), and a plan is kept only if it leaves the label
    clear on more frames. The labels of the figure's other animated
    axes that were planned before count as labels to keep off (a grid
    of panels). Of places that are as clear a label prefers the nearer,
    and the one that points out of the cloud of labeled points, so its
    leader line does not cross the cloud.
    Where a glide would carry a label over static content the label
    jumps instead (``'forced'`` in the table marks those frames). A
    label that comes into view (a sliding window, a reveal) is planned
    for the frames it is shown on; an animation that loops without a
    seam (`_labels_loop`) is planned round the loop.

    The table belongs to the animation and not to a draw: it is worked
    out from the figure's geometry by running the frame function for
    every frame (`track_labels` supplies it; nothing is rendered), so
    frame `i` is the same picture whichever frames were drawn before
    it, in a saved file, and in a second identical call. It is worked
    out again when the figure it was made for has changed (its size,
    the axes' box, the labels the axes holds).

    With `_independent` every frame is instead placed on its own by the
    rule of a static figure (`_arrange_labels`, every label starting
    from its usual place) and nothing is eased: what each frame would
    get as a still, kept for comparison (tests, and the numbers in the
    changelog)."""
    state = _label_state(ax)
    spec = state.frames
    if spec is None or state.solving:
        return None
    n_frames, show_frame, interval_ms = spec
    show_frame = show_frame()
    if show_frame is None:
        # (the animation is gone: the table it left is kept as it is)
        return state.tracks
    total = n_frames
    entries = _label_entries(ax)
    labels = [entry[0] for entry in entries
              if not _label_user_placed(entry[0])]
    current = state.frame
    count = len(labels)
    # more labels than are ever arranged (`_LABEL_DECLUTTER_MAX`): they
    # stay at their usual places, and no frame is looked at
    crowded = count > _LABEL_DECLUTTER_MAX
    # the frames that are planned: all of them, or, in an animation with
    # more labels x frames than `_LABEL_TRACK_BUDGET`, that many evenly
    # spaced ones (the labels move evenly between two of them), so the
    # time the plan takes and the size of the table have a bound
    keys = np.arange(total)
    if crowded:
        keys = keys[:1]
    elif not _independent and count * total > _LABEL_TRACK_BUDGET:
        wanted = min(total, max(_LABEL_TRACK_KEYS_MIN,
                                _LABEL_TRACK_BUDGET // max(count, 1)))
        keys = np.unique(np.round(
            np.linspace(0, total - 1, wanted)).astype(int))
    n_frames = len(keys)
    state.solving = True
    try:
        anchors = np.full((n_frames, count, 2), np.nan)
        shown = np.zeros((n_frames, count), dtype=bool)
        turning = []
        for row, frame in enumerate(keys if not crowded else ()):
            show_frame(int(frame))
            _settle_axes_for_labels(ax)
            _project_label_anchors(ax, entries)
            origin, to_pt, canvas, anchors[row] = _label_geometry(
                ax, labels)
            shown[row] = [label.get_visible() for label in labels]
            # (the cube's own axis labels turn with it)
            turning.append(_whole_points(
                axis_text_boxes_px(ax), origin, to_pt))
        show_frame(min(current, total - 1))
        _settle_axes_for_labels(ax)
        _project_label_anchors(ax, entries)
        origin, to_pt, canvas, _now = _label_geometry(ax, labels)
        signature = _label_tracks_signature(ax)
        sizes = np.array([_label_size_pt(label) for label in labels],
                         dtype=float).reshape(count, 2)
        pads = _label_pads_pt(labels)
        obstacles = _label_obstacles_pt(ax, None, origin, to_pt,
                                        axis_text=False)
    finally:
        state.solving = False
    shown &= np.all(np.isfinite(anchors), axis=2)
    try:
        fps = 1000.0 / float(interval_ms)
    except (TypeError, ValueError, ZeroDivisionError):
        fps = 15.0
    # (planned frames a second)
    fps *= n_frames / float(total)
    if crowded:
        shown[:] = True
    taps = int(np.clip(round(_LABEL_EASE_SECONDS * fps),
                       *_LABEL_EASE_FRAMES))
    taps = max(1, min(taps, n_frames // 2)) | 1        # odd
    periodic = not _independent and _labels_loop(anchors, shown)
    candidates = _LABEL_CANDIDATES if _independent \
        else _LABEL_TRACK_CANDIDATES
    home = candidates.index(_LABEL_HOME)
    place_index = np.full((n_frames, count), home, dtype=int)
    forced = np.zeros((n_frames, count), dtype=bool)
    offsets = np.full((n_frames, count, 2), np.nan)
    half_boxes = 0.5 * sizes + pads[:, None]
    if _independent or crowded:
        for frame in range(n_frames if not crowded else 0):
            here = np.flatnonzero(shown[frame])
            still = np.vstack([obstacles, turning[frame]])
            if len(here) >= 2 or (len(here) and len(still)):
                now = [_LABEL_HOME] * len(here)
                _arrange_labels(now, [None] * len(here),
                                anchors[frame, here], sizes[here],
                                pads[here], canvas, still, strict=True)
                place_index[frame, here] = [
                    candidates.index(place) for place in now]
        offsets = np.where(
            shown[..., None],
            _label_center_offsets(place_index, sizes, candidates), np.nan)
        turning = turning or [np.zeros((0, 4))]
    else:
        reaches = np.array([reach for _corner, reach in candidates])
        aims = np.array([corner for corner, _reach in candidates],
                        dtype=float)
        aims /= np.hypot(aims[:, 0], aims[:, 1])[:, None]
        # the middle of the labeled points on each frame: a label that
        # points away from it has its leader line outside the cloud
        with np.errstate(invalid='ignore'):
            hub = np.nanmean(np.where(shown[..., None], anchors, np.nan),
                             axis=1) if count else np.zeros((n_frames, 2))
        far = ((reaches - LABEL_OFFSET_PT) / LABEL_OFFSET_PT
               * _LABEL_TRACK_FAR_PER_SECOND / fps
               + 1e-9 * np.arange(len(candidates)))
        #: per frame, the boxes of the cube's own axis labels (NaN rows
        #: fill the frames that show fewer)
        turns = np.full((n_frames, max(1, max(len(boxes)
                                              for boxes in turning)), 4),
                        np.nan)
        for frame, boxes in enumerate(turning):
            turns[frame, :len(boxes)] = boxes
        # the more labels, the fewer times each is planned again (the
        # time `plot()` takes to return grows with both)
        replans = int(np.clip(_LABEL_TRACK_REPLANS * 30 // max(count, 1),
                              1, 3 * _LABEL_TRACK_REPLANS))
        rounds = _LABEL_TRACK_ROUNDS if count <= 40 else (
            1 if count <= 80 else 0)
        #: per frame, each label's box as it is drawn (NaN: not planned
        #: yet, or not shown)
        drawn = np.full((n_frames, count, 4), np.nan)
        # (and, after them, the labels of the figure's other animated
        # axes that are planned already: a panel's labels reach into
        # the gaps of a grid, where the next panel's do too)
        for other in ax.figure.axes:
            theirs = None if other is ax else label_tracks(other)
            if theirs is not None and theirs.get('boxes') is not None \
                    and np.array_equal(theirs['frames'], keys):
                drawn = np.concatenate([drawn, theirs['boxes']], axis=1)

        def _crossed(k, where, frames):
            """Per frame of `frames`: whether label `k`, its middle
            `where` (one row per frame of `frames`) from its point,
            lies on another label, on static content or over the canvas
            edge."""
            middle = anchors[frames, k] + where
            boxes = np.column_stack([middle - half_boxes[k],
                                     middle + half_boxes[k]])
            others = np.delete(drawn[frames], k, axis=1)
            bad = (boxes[:, 0] < canvas[0]) | (boxes[:, 1] < canvas[1]) \
                | (boxes[:, 2] > canvas[2]) | (boxes[:, 3] > canvas[3])
            bad |= _boxes_meet(boxes, obstacles) > 0
            bad |= _boxes_meet(boxes, others) > 0
            bad |= _boxes_meet(boxes, turns[frames]) > 0
            return bad

        # how far along a glide is on each frame before the one it ends on
        step = np.zeros((4 * taps, 1, 1))
        step[2 * taps:] = 1.0
        along = step if taps < 3 else _ease_label_tracks(
            step, np.ones((4 * taps, 1), dtype=bool), taps, False)
        along = [(int(at) - 2 * taps, float(along[at, 0, 0]))
                 for at in range(4 * taps)
                 if 1e-3 < along[at, 0, 0] < 1.0 - 1e-3]

        def _bar(k, middles, source, target, bars, move):
            """Bar label `k` from the change of place `source` to
            `target` on every frame a glide between the two that ends
            there would cross something (`bars`, for
            `_plan_label_places`)."""
            crossed = np.zeros(n_frames, dtype=bool)
            ends = np.arange(n_frames)
            for back, share in along:
                frames = ends + back
                if periodic:
                    frames, at = frames % n_frames, ends
                else:
                    inside = (frames >= 0) & (frames < n_frames)
                    frames, at = frames[inside], ends[inside]
                seen = shown[frames, k]
                frames, at = frames[seen], at[seen]
                if len(frames):
                    where = ((1.0 - share) * middles[source]
                             + share * middles[target])
                    crossed[at] |= _crossed(
                        k, np.broadcast_to(where, (len(frames), 2)), frames)
            for at in np.flatnonzero(crossed):
                if bars[at] is None:
                    bars[at] = np.zeros_like(move)
                bars[at][source, target] = _LABEL_TRACK_COVERED

        def _plan(k):
            """Plan label `k` among the labels in `drawn`: its place on
            every frame, its middle on every frame (eased), the frames
            it jumps on, and on how many frames it is not clear."""
            middles = _label_center_offsets(
                np.arange(len(candidates)), sizes[k][None], candidates)
            # (a place is clear with `_LABEL_TRACK_CLEARANCE_PT` to spare)
            reach = half_boxes[k] + _LABEL_TRACK_CLEARANCE_PT
            centers = anchors[:, k][:, None, :] + middles[None]
            rooms = np.concatenate([centers - reach, centers + reach],
                                   axis=2)
            # (of static content and the canvas edge, with a static
            # figure's `_LABEL_CLEARANCE_PT`)
            reach = half_boxes[k] + _LABEL_CLEARANCE_PT
            wide = np.concatenate([centers - reach, centers + reach],
                                  axis=2)
            with np.errstate(invalid='ignore'):
                off = ((wide[..., 0] < canvas[0])
                       | (wide[..., 1] < canvas[1])
                       | (wide[..., 2] > canvas[2])
                       | (wide[..., 3] > canvas[3]))
            hard = _LABEL_TRACK_OFF * (
                off + np.ceil(_boxes_meet(wide, obstacles)))
            hard += _LABEL_TRACK_OFF * np.ceil(
                _boxes_meet(wide, turns[:, None]))
            near = np.flatnonzero(np.isfinite(drawn[..., 0]).any(axis=0)
                                  & (np.arange(drawn.shape[1]) != k))
            covered = np.zeros_like(hard)
            if len(near):
                covered = _LABEL_TRACK_COVERED * _boxes_meet(
                    rooms, drawn[:, None, near, :])
            covered = np.where(shown[:, k][:, None], covered, 0.0)
            hard = np.where(shown[:, k][:, None], hard, 0.0) + covered
            # (pointing into the cloud: the leader line crosses it)
            away = anchors[:, k] - hub
            length = np.hypot(away[:, 0], away[:, 1])
            with np.errstate(invalid='ignore', divide='ignore'):
                away = np.where(length[:, None] > 1.0,
                                away / length[:, None], 0.0)
            soft = far + (0.5 * _LABEL_TRACK_INWARD_PER_SECOND / fps
                          * (1.0 - np.nan_to_num(away) @ aims.T))
            # (a long way to a new place costs more than a short one)
            # (and a label does not sweep across the whole plot in one
            # glide: no change of place is longer than
            # `_LABEL_TRACK_MOVE_MAX_PT`)
            way = np.hypot(*np.moveaxis(
                middles[:, None, :] - middles[None, :, :], -1, 0))
            move = (_LABEL_TRACK_SWITCH
                    + _LABEL_TRACK_MOVE * way / LABEL_OFFSET_PT)
            move[way > _LABEL_TRACK_MOVE_MAX_PT] = np.inf
            np.fill_diagonal(move, np.inf)
            if shown[:, k].all():
                stretches = [(0, n_frames, periodic)]
            else:
                edges = np.flatnonzero(np.diff(np.concatenate(
                    [[False], shown[:, k], [False]]).astype(int)))
                stretches = [(first, end, False)
                             for first, end in zip(edges[::2], edges[1::2])]
            def _route(rest):
                """The label's plan with at least `rest` frames at each
                place: places, middles, jumps, the number of frames it
                is not clear on."""
                places = np.full(n_frames, home, dtype=int)
                bars = [None] * n_frames
                for _attempt in range(replans + 1):
                    for first, end, loops in stretches:
                        places[first:end] = _plan_label_places(
                            hard[first:end], soft[first:end], move, taps,
                            rest, loops, bars[first:end],
                            covered[first:end])
                    targets = middles[places][:, None, :]
                    eased = targets if taps < 3 else _ease_label_tracks(
                        targets, shown[:, [k]], taps, periodic)
                    # a glide that crosses another label, static content
                    # or the canvas edge: that change of place is barred
                    # on the frames it would, and the label is planned
                    # again
                    gliding = np.flatnonzero(shown[:, k] & (
                        np.abs(eased - targets).max(axis=(1, 2)) > 0.5))
                    crossing = gliding[_crossed(k, eased[gliding, 0],
                                                gliding)] \
                        if len(gliding) else gliding
                    changes = set()
                    for frame in crossing:
                        # (the glide ends at the next change of place)
                        for later in range(frame, frame + taps + 1):
                            at = later % n_frames if periodic else later
                            if at >= n_frames:
                                break
                            if places[at] != places[at - 1]:
                                changes.add((int(places[at - 1]),
                                             int(places[at])))
                                break
                    if not changes or _attempt == replans:
                        break
                    for source, target in sorted(changes):
                        _bar(k, middles, source, target, bars, move)
                # a glide that would still carry the label over a
                # legend, a title, an axis label or the canvas edge,
                # all of which both its ends are clear of: the label
                # jumps instead
                def _out(where):
                    middle = anchors[:, k] + where[:, 0]
                    boxes = np.column_stack([middle - half_boxes[k],
                                             middle + half_boxes[k]])
                    with np.errstate(invalid='ignore'):
                        return ((boxes[:, 0] < canvas[0])
                                | (boxes[:, 1] < canvas[1])
                                | (boxes[:, 2] > canvas[2])
                                | (boxes[:, 3] > canvas[3])
                                | (_boxes_meet(boxes, obstacles) > 0)
                                | (_boxes_meet(boxes, turns) > 0))

                jumps = np.zeros(n_frames, dtype=bool)
                bad = _out(eased) & ~_out(targets) & shown[:, k]
                for frame in np.flatnonzero(bad):
                    by = np.arange(frame - taps, frame + taps + 1)
                    by = (by % n_frames if periodic
                          else by[(by >= 0) & (by < n_frames)])
                    jumps[by] = True
                jumps &= shown[:, k]
                if jumps.any():
                    # (the jump is made as early as the place the label
                    # leaves is no longer clear of them and the place
                    # it goes to is)
                    for at in np.flatnonzero(
                            jumps & (places != np.roll(places, 1))):
                        if at == 0 and not periodic:
                            continue
                        source, target = places[at - 1], places[at]
                        leaves = _out(np.broadcast_to(
                            middles[source], (n_frames, 1, 2)))
                        lands = _out(np.broadcast_to(
                            middles[target], (n_frames, 1, 2)))
                        for back in range(1, taps + 1):
                            frame = at - back
                            if frame < 0 and not periodic:
                                break
                            frame %= n_frames
                            if (places[frame] != source or not shown[frame, k]
                                    or not leaves[frame] or lands[frame]):
                                break
                            places[frame] = target
                    targets = middles[places][:, None, :]
                eased[jumps] = targets[jumps]
                where = np.where(shown[:, k][:, None], eased[:, 0], np.nan)
                visible = np.flatnonzero(shown[:, k])
                unclear = int(_crossed(k, where[visible], visible).sum()) \
                    if len(visible) else 0
                return places, where, jumps, unclear

            return _route(_LABEL_REST_EASES * taps + 1)

        def _keep(k, plan):
            place_index[:, k], offsets[:, k], forced[:, k] = plan[:3]
            middle = anchors[:, k] + offsets[:, k]
            drawn[:, k] = np.column_stack([middle - half_boxes[k],
                                           middle + half_boxes[k]])

        unclear = np.zeros(count, dtype=int)
        for k in range(count):
            if shown[:, k].any():
                plan = _plan(k)
                _keep(k, plan)
                unclear[k] = plan[3]
        # a label that is still not clear on some frame, and the labels
        # it lies on, are planned again among ALL the others (the first
        # time round a label only knew of the labels before it), for as
        # long as that leaves fewer frames with a label on another
        for _round in range(rounds):
            again = np.zeros(count, dtype=bool)
            for k in np.flatnonzero(unclear):
                frames = np.flatnonzero(shown[:, k])
                bad = frames[_crossed(k, offsets[frames, k], frames)]
                with np.errstate(invalid='ignore'):
                    met = (
                        (np.minimum(drawn[bad, k, None, 2], drawn[bad, :, 2])
                         > np.maximum(drawn[bad, k, None, 0],
                                      drawn[bad, :, 0]))
                        & (np.minimum(drawn[bad, k, None, 3],
                                      drawn[bad, :, 3])
                           > np.maximum(drawn[bad, k, None, 1],
                                        drawn[bad, :, 1])))
                again |= met.any(axis=0)[:count]
                again[k] = True
            better = False
            for k in np.flatnonzero(again):
                before = (place_index[:, k].copy(), offsets[:, k].copy(),
                          forced[:, k].copy())
                frames = np.flatnonzero(shown[:, k])
                was = int(_crossed(k, offsets[frames, k], frames).sum())
                plan = _plan(k)
                if plan[3] < was:
                    _keep(k, plan)
                    better = True
                else:
                    _keep(k, before)
            for k in np.flatnonzero(shown.any(axis=0)):
                frames = np.flatnonzero(shown[:, k])
                unclear[k] = _crossed(k, offsets[frames, k], frames).sum()
            if not better or not unclear.any():
                break
    # (each label's box on each frame, for the figure's other axes)
    middle = anchors + offsets
    label_boxes = None if _independent or crowded else np.concatenate(
        [middle - half_boxes, middle + half_boxes],
        axis=2).astype(np.float32)
    # (kept small: 28 bytes a label and planned frame)
    state.tracks = {
        'labels': [weakref.ref(label) for label in labels],
        'offsets': offsets.astype(np.float32),
        'places': place_index.astype(np.int16), 'shown': shown,
        'forced': forced, 'signature': signature, 'periodic': periodic,
        'ease_frames': taps, 'candidates': candidates,
        'boxes': label_boxes, 'frames': keys, 'n_frames': total}
    _apply_label_tracks(ax)
    return state.tracks


def _label_tracks_signature(ax):
    """What an animation's label table was made for: the canvas in
    points, the box of the axes in it, the labels the axes holds, the
    size of each label's text and of its frame, and everything the
    labels are kept off (`_label_obstacles_pt`: legends, titles, figure
    texts, the other axes). A figure given another size, an axes given
    more labels, a label given a longer text or a larger font, a text
    added to the figure: each needs the table worked out again."""
    fig = ax.get_figure(root=True)
    labels = [entry[0] for entry in _label_entries(ax)
              if not _label_user_placed(entry[0])]
    origin, to_pt, _canvas, _anchors = _label_geometry(ax, [])
    return (tuple(np.round(fig.get_size_inches() * 72.0, 3)),
            tuple(np.round(ax.get_position().bounds, 6)),
            tuple(id(label) for label in labels),
            tuple(_label_size_pt(label) for label in labels),
            _label_pads_pt(labels).tobytes(),
            _label_obstacles_pt(ax, None, origin, to_pt,
                                axis_text=False).tobytes())


def _apply_label_tracks(ax, frame=None):
    """Put the labels of the animated axes `ax` where its table has them
    on `frame` (the frame the axes shows, when not given). Called by the
    frame function of every animation style (`_sync_anim_labels`), where
    a table made for a figure that has since changed
    (`_label_tracks_signature`) is worked out again first, and at draw
    time (`declutter_labels`).

    A figure that changes on EVERY frame (an `on_frame=` callback that
    rewrites a figure text) is not planned again on every frame: the
    table is worked out again at the first two changes in a row, and
    once more when the figure has stopped changing.

    Between two planned frames (an animation over
    `_LABEL_TRACK_BUDGET`) a label is the matching way between its
    offsets on the two."""
    state = _label_state(ax)
    tracks = state.tracks
    if frame is not None:
        state.frame = int(frame)
        if state.solving:
            return
        state.shows += 1
        if state.frames is not None and tracks is None:
            solve_label_tracks(ax)
            return
        if state.frames is not None:
            signature = _label_tracks_signature(ax)
            changed = tracks['signature'] != signature
            recent = state.shows - state.solved_at <= 1
            if changed:
                state.streak = state.streak + 1 if recent else 1
                state.solved_at = state.shows
            if changed and state.streak > 2:
                # (changing frame after frame: wait until it stops)
                tracks['signature'] = signature
                state.pending = True
            elif changed or (state.pending and not recent):
                state.pending = False
                solve_label_tracks(ax)
                return
    if tracks is None:
        return
    offsets, keys = tracks['offsets'], tracks['frames']
    at = min(max(state.frame, 0), tracks['n_frames'] - 1)
    if len(keys) == tracks['n_frames']:
        row = offsets[at]
    else:
        low = int(np.searchsorted(keys, at, side='right')) - 1
        high = min(low + 1, len(keys) - 1)
        share = ((at - keys[low]) / float(keys[high] - keys[low])
                 if high > low else 0.0)
        before, after = offsets[low], offsets[high]
        row = np.where(np.isnan(before), after,
                       np.where(np.isnan(after), before,
                                (1.0 - share) * before + share * after))
    for ref, (dx, dy) in zip(tracks['labels'], row):
        label = ref()
        if label is None or label.axes is not ax \
                or not np.isfinite(dx) or _label_user_placed(label):
            continue
        _set_label_offset(label, float(dx), float(dy))


def _set_label_offset(label, dx, dy):
    """Draw `label` with the middle of its text (`dx`, `dy`) points from
    its point: how an animation's table puts a label anywhere on the
    way from one place to the next."""
    if getattr(label, '_hyp_offset', None) == (dx, dy) and getattr(
            label, '_hyp_place', None) is None:
        return
    label.set_position((dx, dy))
    label.set_ha('center')
    label.set_va('center')
    label._hyp_offset = (dx, dy)
    # (no named place: `_set_label_place` sets one afresh if this label
    # is ever arranged as a static figure's)
    label._hyp_place = None
    label._hyp_place_left = None


def track_labels(ax, n_frames, show_frame, interval_ms=None):
    """Have the labels of the ANIMATED axes `ax` follow a table worked
    out for the whole animation (`solve_label_tracks`), now and whenever
    the figure changes: `show_frame(i)` runs the animation's frame
    function for frame `i` of `n_frames` (the updater alone: nothing is
    rendered and no `on_frame=` callback is called), and `interval_ms`
    is the time between two frames.

    `show_frame` is kept by a WEAK reference (it refers to the
    animation, which refers to the figure: a strong one would keep
    every closed figure alive): the caller keeps it alive for as long
    as the animation lives, a function as an attribute of the
    animation, a bound method through its object."""
    if not _label_entries(ax):
        return None
    state = _label_state(ax)
    weak = (weakref.WeakMethod(show_frame)
            if hasattr(show_frame, '__self__') else weakref.ref(show_frame))
    state.frames = (int(n_frames), weak, interval_ms)
    state.tracks = None
    return solve_label_tracks(ax)


class _LabelDeclutter(matplotlib.artist.Artist):
    """Invisible child of an axes that carries observation labels: it is
    drawn before them (lowest zorder) and puts them clear of each other
    for the draw that follows (`declutter_labels`). Being part of the
    draw, it sees the figure as it is then: after any layout, at any
    size, at the camera angle of the frame being drawn."""

    def __init__(self):
        super().__init__()
        self.set_zorder(-1e6)
        self.set_in_layout(False)

    def draw(self, renderer):
        """Place the axes' labels for this draw; draws nothing itself."""
        if self.axes is not None and self.get_visible():
            declutter_labels(self.axes, renderer)
        self.stale = False


def _watch_labels(ax, labels_and_points):
    """Record the labels one call has drawn into `ax` and give the axes
    its `_LabelDeclutter` (one per axes).

    The axes keeps the labels of EVERY call that drew into it, so two
    `hyp.plot(..., labels=, ax=ax)` calls are kept off each other like
    the labels of one call (1.2.1 review round 2: the second call's list
    replaced the first's, whose labels were then never moved). What an
    `ax.clear()` removed is forgotten: the labels, the declutter artist
    itself (the old one is no longer a child of the axes, so a new one
    is added) and, when none of an animation's labels are left, the
    table the animation's labels followed (`solve_label_tracks`)."""
    alive = set(ax.texts)
    kept = [entry for entry in
            (getattr(ax, '_hyp_labels_and_points', None) or [])
            if entry[0] in alive]
    if not kept:
        forget_label_tracks(ax)
    ax._hyp_labels_and_points = kept + [
        entry for entry in labels_and_points if entry[0] in alive]
    ax._hyp_label_answer = None
    ax._hyp_labels_held = True
    watcher = getattr(ax, '_hyp_label_declutter', None)
    if watcher is None or watcher not in ax.get_children():
        ax._hyp_label_declutter = ax.add_artist(_LabelDeclutter())


def _connect_label_events(fig, ax):
    """Hide the labels of `ax` while a mouse button is down and put them
    back, at their points, when it is let go (a 3-D axes is rotated in
    between, and its labels are placed in the projected plane).

    ONE pair of callbacks per axes, however many `plot()` calls label
    it, and they read the labels the axes holds NOW (`_label_entries`).
    Each call used to connect a pair of its own that kept that call's
    labels: a second labeled plot into the same axes repositioned every
    earlier call's labels again, and after an ``ax.clear()`` the first
    call's callbacks, still connected, reached for the removed labels
    (whose ``axes`` is None) and raised ``AttributeError`` out of every
    button release, leaving the new labels hidden (1.2.1 review round
    3)."""
    # (which axes this canvas already serves is kept on the canvas: a
    # figure given another canvas has none of the callbacks, and nothing
    # is left on the axes or the figure that would have to be pickled)
    canvas = fig.canvas
    served = getattr(canvas, '_hyp_label_axes', None)
    if served is None:
        served = canvas._hyp_label_axes = weakref.WeakSet()
    if ax in served:
        return
    served.add(ax)

    def hide_labels(event):
        """Button press: hide every label the axes holds now (a 3-D
        axes is about to be turned, and its labels are placed in the
        projected plane)."""
        for entry in _label_entries(ax):
            entry[0]._visible = False

    def update_position(event):
        """Button release: put every label the axes holds now back at
        its point (projected with the camera the drag left, on a 3-D
        axes), show it again and redraw the canvas."""
        entries = _label_entries(ax)
        if not entries:
            return
        # `ax.get_proj()` / `proj3d.proj_transform` only exist on 3D
        # (Axes3D) axes -- a 2D `Axes` lacks `get_proj`. `annotate_plot`
        # stores 4-tuples (label, x, y, z) for 3D vs. 3-tuples
        # (label, x, y) for 2D, so branch the same way here.
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
        for entry in entries:
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

    canvas.mpl_connect("button_press_event", hide_labels)
    canvas.mpl_connect("button_release_event", update_position)


class _TraceFollowers(matplotlib.artist.Artist):
    """Invisible child of an animated axes that keeps the SECOND artists
    of a trace with the first (`follow_artist`), at draw time, before
    anything else of the axes is drawn."""

    def __init__(self):
        super().__init__()
        self.set_zorder(-1e6)
        self.set_in_layout(False)
        self.followers = []

    def draw(self, renderer):
        """Bring every follower in line with its leader; draws nothing."""
        for follower, with_alpha in self.followers:
            leader = getattr(follower, '_hyp_leader', None)
            if leader is None:
                continue
            if not leader.get_visible():
                if follower.get_visible():
                    follower.set_visible(False)
                    follower._hyp_hidden_with_leader = True
            elif getattr(follower, '_hyp_hidden_with_leader', False):
                # (only what was hidden HERE is shown again: a follower
                # hidden by anything else stays hidden)
                follower.set_visible(True)
                follower._hyp_hidden_with_leader = False
            if with_alpha and follower.get_alpha() != leader.get_alpha():
                follower.set_alpha(leader.get_alpha())
        self.stale = False


def follow_artist(ax, follower, leader, alpha=False):
    """Have `follower` share the fate of `leader`, the artist of the same
    trace that `FrameContext.artists` hands to callbacks: it is drawn
    only when `leader` is visible, and with ``alpha=True`` at `leader`'s
    opacity.

    One trace can be two artists: a `head_marker=` beside its line, and,
    with a continuous `hue=`, the markers of a line-and-marker format
    beside the colored line. A callback (``on_frame=``) holds only the
    first, so hiding or fading it left the second on screen (1.2.1
    review round 2). A follower that is hidden for a reason of its own
    (no points yet, or hidden by the caller) stays hidden: only a
    follower that was hidden WITH its leader is shown again with it.
    Calling this again gives `follower` a new leader."""
    follower._hyp_leader = leader
    watcher = getattr(ax, '_hyp_trace_followers', None)
    if watcher is None or watcher not in ax.get_children():
        watcher = ax._hyp_trace_followers = ax.add_artist(_TraceFollowers())
    if not any(one is follower for one, _alpha in watcher.followers):
        watcher.followers.append((follower, alpha))


def _apply_title(ax, text, font=None, title_kwargs=None, family=None):
    """Set `ax`'s title, honoring the resolved `font=` and `title_kwargs=`
    (GH #285).

    With `title_kwargs=None` and `family=None` this makes EXACTLY the call
    hypertools has always made (`ax.set_title(text)`, or `ax.set_title(text,
    fontproperties=font)` when a `font=` was resolved), so an un-styled
    title's Text artist is byte-identical to before.

    `family` (a font-family list: hypertools' fallback stack) is given
    EXPLICITLY when no `font=` is: an axes created outside hypertools'
    ``rc_context`` -- a caller's ``ax=``, every `panels=` cell -- holds a
    title Text whose family is the bare ``'sans-serif'`` alias, resolved at
    DRAW time against the caller's rcParams, so those titles rendered in
    DejaVu Sans while a figure of hypertools' own used Noto Sans (1.1
    release review, figure QA). A family named in `title_kwargs` wins.

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
    _family_keys = ('fontproperties', 'font_properties', 'family',
                    'fontfamily', 'fontname', 'name', 'font')
    if title_kwargs:
        kwargs = {}
        if font is not None and 'fontproperties' not in title_kwargs:
            kwargs['fontproperties'] = font
        elif family is not None and not any(k in title_kwargs
                                            for k in _family_keys):
            kwargs['fontfamily'] = family
        kwargs.update(title_kwargs)
        return ax.set_title(text, **kwargs)
    if font is not None:
        return ax.set_title(text, fontproperties=font)
    if family is not None:
        return ax.set_title(text, fontfamily=family)
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

#: The full, unrevealed series is drawn once underneath in this gray, so the
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
    axes is deliberately maximized to the full canvas so a rotating zoomed
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


def add_companion_panel(fig, spec, cmap=None, norm=None, font=None,
                        default_color=None):
    """Draw one `companion=` panel and return the state its per-frame
    updater needs (GH #285).

    The panel shows the whole series once, faintly, then reveals it: a line
    (or, with ``hue=``, a `LineCollection` colored through the plot's own
    color scale) up to the reveal head, an optional trailing rolling mean,
    and an optional marker on the head itself.

    The line and head marker are drawn in ``spec['color']``, else in
    `default_color` -- `plot()` passes the color of the trajectory the
    panel accompanies -- else in matplotlib's ``'C0'`` (only a direct
    caller that passes neither reaches that last fallback; 1.1 visual
    review L12).
    """
    from matplotlib.collections import LineCollection

    rect = _grow_for_companion(fig, spec)
    pax = fig.add_axes(rect)
    x, y = _companion_xy(spec['data'])
    color = spec['color'] or default_color or 'C0'

    pax.plot(x, y, color=COMPANION_GHOST_COLOR, linewidth=0.6)
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    if spec['hue'] is not None:
        # the PLOT's own resolved color scale when there is one (a
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
    # Notebook review 2026-09: full-curve visibility must not freeze the
    # current-time marker. Keep the curve extent separate from the clock.
    end = i if panel['reveal'] else panel['n_rows'] - 1
    panel['revealed'].set_segments(panel['segments'][:end])
    if panel['hue'] is not None:
        panel['revealed'].set_array(panel['hue'][1:end + 1])
    if panel['trend'] is not None:
        panel['trend'].set_data(panel['x'][:end + 1], panel['rolling'][:end + 1])
    if panel['head'] is not None:
        panel['head'].set_data([panel['x'][i]], [panel['y'][i]])
        if panel['head_colors'] is not None:
            panel['head'].set_markerfacecolor(panel['head_colors'][i])
    return panel


class _AxisLabelExtent(matplotlib.artist.Artist):
    """A draw-nothing figure artist whose window extent is the union of an
    `Axes3D`'s axis-label extents, so `Figure.get_tightbbox` (and with it
    ``savefig(bbox_inches='tight')``) includes the labels that
    `Axes3D.get_tightbbox` leaves out. See the 3-D label branch of `_draw`.
    """

    def __init__(self, ax):
        super().__init__()
        self.axes_ref = ax
        self.set_in_layout(True)
        self.set_clip_on(False)

    def draw(self, renderer):
        """Draw nothing: the artist exists only for its extent."""
        return None

    def get_window_extent(self, renderer=None):
        """The union of the axes' visible, non-empty axis-label extents
        (display pixels), or a null box when there is none."""
        from matplotlib.transforms import Bbox
        ax = self.axes_ref
        if renderer is None:
            try:
                renderer = ax.figure.canvas.get_renderer()
            except AttributeError:
                return Bbox.null()
        boxes = []
        for axis in getattr(ax, '_axis_map', {}).values():
            label = axis.label
            if not (label.get_visible() and label.get_text()):
                continue
            box = label.get_window_extent(renderer)
            if box.width > 0 and box.height > 0:
                boxes.append(box)
        return Bbox.union(boxes) if boxes else Bbox.null()


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
                       legend_kwargs=None, ax=None):
    """The `ax.legend(...)` keyword arguments hypertools draws a legend with.

    Factored out of `_draw` (GH #285) so `plot()` can REBUILD the legend
    with identical placement/styling after it adds a labeled overlay
    (a multi-model `predict=` fan, or `truth=`) -- the legend is built
    inside `_draw`, before those artists exist, and a rebuild that guessed
    at the placement would move the legend on exactly the figures that
    gained an entry.

    `ax` is the axes the legend is for. A 2-D hypertools axes fills the
    canvas and the frame the reader sees is a square drawn INSIDE it
    (`frame_2d`), so a caller's ``loc='upper left'`` put the legend in
    the corner of the invisible axes: across the drawn frame's border,
    its keys left of the frame (1.2.1 review round 4; 1.2.0 too). With
    `ax` given, a named ``loc`` is read against the drawn frame, as it
    is read against the axes' frame in matplotlib.
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
        if 'loc' in legend_kwargs and 'bbox_to_anchor' not in legend_kwargs:
            # a caller's `loc=` names a place ON the axes ('upper left');
            # keeping hypertools' outside-right anchor would hang the
            # legend off the right edge with its upper-left corner at the
            # anchor (1.1 release review, feature-tour 9.6)
            call.pop('bbox_to_anchor', None)
            call.pop('borderaxespad', None)
            frame = getattr(ax, '_hyp_frame_box', None)
            if (frame is not None and not is_3d
                    and isinstance(legend_kwargs['loc'], (str, int))
                    and 'bbox_transform' not in legend_kwargs):
                # the frame the reader sees, in data coordinates
                call['bbox_to_anchor'] = frame
                call['bbox_transform'] = ax.transData
        call.update(legend_kwargs)
        # matplotlib ignores `fontsize=` whenever `prop=` is given, so with
        # a `font=` the user's size silently lost; fold it into the
        # FontProperties instead so legend_kwargs still wins.
        if font is not None and 'fontsize' in call and 'prop' in call \
                and call['prop'] is font:
            prop = font.copy()
            prop.set_size(call.pop('fontsize'))
            call['prop'] = prop
    return call


def show_legend_keys(legend):
    """Make every key of `legend` visible, and return the legend.

    matplotlib builds a legend key as a copy of the artist it stands for,
    visibility included. An ``animate='morph'`` animation HIDES each
    morphing dataset's own artist for good -- the one traveling cloud is
    all that is drawn -- so every key of its legend was an invisible copy:
    labels with nothing beside them (1.2.0 review, round 3; also in
    1.1.0). A listed entry always shows its key: the dataset's marker in
    its color."""
    if legend is not None:
        for handle in legend.legend_handles:
            if handle is not None and not handle.get_visible():
                handle.set_visible(True)
    return legend


def _recolor_legend_handles(legend, colors):
    """Apply `legend_colors=`'s plain color list to an already-built
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
    im.set_clip_path(patches.Rectangle(
        (-UNIT_FRAME_SCALE, -UNIT_FRAME_SCALE), 2 * UNIT_FRAME_SCALE,
        2 * UNIT_FRAME_SCALE, transform=ax.transData))


def _draw_density_2d(ax, points_list, density, density_colors,
                     clip_unit=True):
    """Draw each dataset's (or, with ``per_group=False``, one pooled) 2-D
    KDE density layer (GH #108/#191); each grid reaches `KDE_GRID_BANDWIDTHS`
    kernel widths past its own cloud, so the glow fades out inside it
    (see `kde_grid_2d`)."""
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


def held_frame_sequence(n_frames, hold_frames=0):
    """What `FuncAnimation` is given as ``frames=`` for an animation of
    `n_frames` content frames whose last frame is held for `hold_frames`
    more (``hold=``, GH #299): the int `n_frames` itself without a hold
    (exactly what was always passed), else the list ``[0, ..., n-1, n-1,
    ..., n-1]``."""
    n_frames, hold_frames = int(n_frames), int(hold_frames or 0)
    if hold_frames <= 0:
        return n_frames
    return list(range(n_frames)) + [n_frames - 1] * hold_frames


def content_frame_count(line_ani):
    """How many DISTINCT frames an animation draws: its frame count less
    the frames a ``hold=`` repeats at the end (GH #299). Frame indices
    handed to a frame function must stay below this."""
    count = getattr(line_ani, '_hyp_content_frames', None)
    return int(line_ani._save_count if count is None else count)


class PanelAnimationStub:
    """One panel's share of an animated `panels=` grid (GH #295).

    An ordinary animated plot hands its frame function to a
    `FuncAnimation`. A panel of an animated grid must not: the grid has ONE
    animation, whose frame function calls every panel's. So a panel's
    `_draw` returns this instead -- the frame function, its arguments, the
    frame count and the frame interval, under the names `FuncAnimation`
    itself uses (`_func`, `_args`, `_save_count`, `_interval`), so the code
    in `plot()` that wraps `line_ani._func` (per-point colors, frame hooks)
    works on either unchanged.

    `zoom_scale` is the panel's handle on its own 3-D camera zoom: the grid
    lowers ``zoom_scale[0]`` until the panel's rotating cube fits its cell
    (`plot._fit_panel_cubes`), starting from `start_scale`.
    """

    #: the ``zoom_scale`` the grid's cube fit starts from: the one that
    #: undoes the extra box zoom a single animated plot is drawn with
    #: (`_anim_box_zoom`, 1.125 at the default ``zoom=1``; it fills a whole
    #: canvas), so a panel at the default `zoom=` starts at matplotlib's
    #: default box zoom -- the size a STATIC grid's cube is drawn at. A
    #: static grid and the same grid animated with ``rotations=0`` then
    #: draw the same cubes, and `zoom=` still scales a panel's cube
    #: relative to that.
    start_scale = 1.0 / _anim_box_zoom(1)

    def __init__(self, func, args, n_frames, interval, zoom_scale=None):
        self._func = func
        self._args = tuple(args)
        self._save_count = int(n_frames)
        self._interval = interval
        #: one-element list holding the factor the panel's frame function
        #: multiplies its 3-D box zoom by (read on every frame)
        self.zoom_scale = [1.0] if zoom_scale is None else zoom_scale

    def draw(self, frame):
        """Draw frame `frame` of this panel."""
        return self._func(frame, *self._args)


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
    legend_order=None,
    panel_cell=False,
    head_marker=None,
    tail_linewidth=None,
    trail_alpha=0.3,
    hold_frames=0,
):
    """
    Draws the plot

    `head_marker` / `tail_linewidth` / `trail_alpha` / `hold_frames`
    (GH #299), animations only. `head_marker`: None, or one resolved spec
    per drawn dataset (a dict of `Line2D` marker properties with at least
    ``'marker'``, or None for a dataset without one) -- a marker drawn at
    the last vertex of that dataset's head line on every frame
    (`_place_head_marker`). `tail_linewidth`: the line width of the head
    line (the recent window), while the trail keeps the dataset's
    ``linewidth``. `trail_alpha`: the factor a trail's alpha is the
    dataset's alpha times (0.3, the historical constant). `hold_frames`:
    how many times the LAST frame is repeated at the end of the animation
    (`_new_animation`).

    `legend_order` (1.1 release review): the legend labels in the order
    their entries should be listed, or None for drawn-artist order. The
    categorical LINE path draws one artist per contiguous run, so its
    legend followed the order the categories first APPEAR along the data
    (clusters ``0, 2, 1``) while the marker path lists them sorted; the
    entries it names are sorted into this order among their own
    positions, every other entry (an earlier call's, a forecast's) staying
    where it was.

    `raw_data` (GH #141): the PRE-interpolation per-dataset points, same
    length as `x`/`fmt`. Used only by the STATIC (non-animated) plot1D/2D/
    3D functions below: a dataset whose `fmt` combines a marker AND a line
    (e.g. 'o-') is drawn as two artists -- a smoothed line from `x` (the
    already-interpolated data) plus markers at the raw sample points from
    `raw_data` -- so markers land on the true data regardless of how dense
    the smoothed line is. A LINE fmt given its marker by ``marker=``/
    ``markers=`` stays one artist, whose ``markevery`` picks out the raw
    samples among the smoothed vertices. Ignored (may be None) for pure
    line/marker-only styles. An
    ANIMATED marker+line style keeps ONE artist per dataset (so its legend
    handle shows marker and line), drawn against the smoothed frame-grid
    `x` data; `raw_data` then locates each observation's nearest drawn
    vertex, and each frame's ``markevery`` marks only those
    (`_mark_observations`) -- before the 1.1 release review every
    interpolated vertex carried a marker.

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

    `axis_scale` (GH #285): ``'unit'`` (the historical behavior) draws the
    hypertools frame square (half-width `UNIT_FRAME_SCALE`) and pins the 2-D axes to ``+-UNIT_FRAME_LIMIT`` --
    `plot()` has already mean-centered and rescaled the data into ``[-1, 1]``
    for it. ``'data'`` draws NO frame square, leaves matplotlib's own ticks
    and spines visible, and takes its limits from `xlim`/`ylim` (which
    `plot()` computes from the full data, forecasts included, so an
    animation's viewport never jumps) or from matplotlib's autoscale when
    both are None. 3-D is always ``'unit'`` (`plot()` refuses the other).

    `xlim`/`ylim` (GH #285): explicit ``(low, high)`` axis limits, honored
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

    # whether the caller handed us their own axes (their styling -- e.g. a
    # grid they turned on -- is theirs to keep; see the axis_scale='data'
    # branch below)
    _caller_ax = ax is not None
    # `panel_cell=True` (GH #295): this call draws ONE PANEL OF AN ANIMATED
    # `panels=` GRID into the caller's axes. The animated builders below
    # then draw on that axes instead of creating a figure, and hand back a
    # `PanelAnimationStub` (the frame function and its arguments) instead of
    # a `FuncAnimation`: the grid owns the one animation that drives every
    # panel on a shared clock.
    _cell_ax = ax if (panel_cell and ax is not None) else None
    # ...and a factor on the animated 3-D box zoom, which the grid lowers
    # (through the stub) until the rotating cube stays inside its cell. A
    # single animated plot starts at 1.0 and is lowered only if its cube
    # would reach the canvas edge or a legend (`plot._keep_cube_clear`).
    _cell_zoom_scale = [1.0]

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
            lo, out = a, dense[a:b]
        elif b <= a:
            lo, out = a * step, dense[0:0]
        else:
            lo, out = a * step, dense[a * step:(b - 1) * step + 1]
        if artist is not None:
            _mark_observations(i, artist, lo, len(out))
        return out

    # markers at the TRUE observations in an animation (the shared marker
    # contract; 1.1 release review): an animated line is resampled onto
    # the frame grid (`plot._interp_anim_line`) and densified again above,
    # so its vertices are not the observations, and a marker+line style
    # ('o-', or a line fmt with marker=/markers=) marked every one of them
    # -- a tube of ~500 markers for 48 samples. Mark, on the SAME artist
    # (its legend handle keeps the marker), only the drawn vertex nearest
    # each observation: found by position, in a window around the
    # observation's proportional place along the curve, so it holds for
    # whatever observation -> grid mapping the resampling produces.
    _obs_vertices_cache = {}

    def _observation_vertices(i):
        if i in _obs_vertices_cache:
            return _obs_vertices_cache[i]
        found = None
        if (animate and raw_data is not None and i < len(raw_data)
                and raw_data[i] is not None):
            raw = np.asarray(raw_data[i], dtype=float)
            dense = np.asarray(_aa_curves[i][0], dtype=float)
            n, d = raw.shape[0], dense.shape[0]
            if (raw.ndim == 2 and dense.ndim == 2 and n and d
                    and raw.shape[1] == dense.shape[1]):
                if n == 1 or d == 1:
                    found = np.array([0])
                else:
                    est = np.rint(np.arange(n) * (d - 1) / (n - 1)).astype(int)
                    w = int(np.ceil((d - 1) / (n - 1)))
                    idx = []
                    for k in range(n):
                        lo, hi = max(0, est[k] - w), min(d, est[k] + w + 1)
                        gap = np.linalg.norm(dense[lo:hi] - raw[k], axis=1)
                        idx.append(lo + int(np.argmin(gap)))
                    found = np.unique(idx)
        _obs_vertices_cache[i] = found
        return found

    def _mark_observations(i, artist, lo, count):
        marker = artist.get_marker() if hasattr(artist, 'get_marker') else None
        if marker in (None, 'None', 'none', '', ' '):
            return
        if not has_line_component(_fmt_at(i)) and artist.get_linestyle() in (
                'None', 'none', '', ' '):
            return                  # markers only: its vertices ARE the rows
        verts = _observation_vertices(i)
        if verts is None:
            return
        local = verts[(verts >= lo) & (verts < lo + count)] - lo
        artist.set_markevery([int(v) for v in local])

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
        # an explicit marker= (or its markers= alias) wins over the fmt's
        # marker, as the comment above promises and as plotly draws it:
        # the split below used to discard it, so fmt='-o' with
        # marker=['o', 's'] drew two circle datasets (1.1 release review).
        _fmt_has_marker = marker_char is not None
        _explicit_marker = ikwargs.get('marker')
        if _explicit_marker is not None:
            marker_char = (None if isinstance(_explicit_marker, str)
                           and _explicit_marker.strip().lower() in ('', 'none')
                           else _explicit_marker)
        if (line_token is not None and marker_char is not None
                and not _fmt_has_marker):
            # a LINE fmt given its marker by marker=/markers= ('-' with
            # markers='o'): one artist, as before, but marked only at the
            # TRUE samples -- every one an exact vertex of the smoothed line
            # (`antialias_line`: ``dense[::step]`` IS the data) -- rather
            # than at all ~20x-denser interpolated vertices, which
            # contradicted `antialias=`'s promise (1.1 release review)
            n_dense, n_raw = len(coords[0]), len(raw_coords[0])
            one_kwargs = dict(ikwargs)
            one_kwargs.setdefault('linestyle', line_token)
            if fmt_color is not None:
                one_kwargs.setdefault('color', fmt_color)
            if (raw_data is not None and 1 < n_raw < n_dense
                    and (n_dense - 1) % (n_raw - 1) == 0):
                one_kwargs.setdefault('markevery', list(range(
                    0, n_dense, (n_dense - 1) // (n_raw - 1))))
            ax.plot(*coords, **one_kwargs)
            return
        if line_token is not None and marker_char is not None:
            line_kwargs = {k: v for k, v in ikwargs.items() if k != 'marker'}
            line_kwargs.setdefault('linestyle', line_token)
            if fmt_color is not None:
                line_kwargs.setdefault('color', fmt_color)
            # the line artist carries the legend label, so it also carries
            # the MARKER -- with `markevery=[]` it draws none along the
            # interpolated vertices (the markers-only artist below draws
            # them at the raw sample points), but its legend handle shows
            # marker + line, as 's--' promises. Before the 1.1 release
            # review the handle showed only the dashes.
            line_kwargs['marker'] = marker_char
            line_kwargs['markevery'] = []
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
            # tagged so run-indexed consumers (the forecast/truth overlays
            # in plot.py) skip it: it is the SAME run's markers, not a run
            ax.plot(*marker_coords, **marker_kwargs)[0] \
                ._hyp_marker_companion = True
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
            # x is the ROW index: static antialiasing densified `data[i]`
            # upstream (uniformly, every original row kept), so its vertices
            # span the same 0..n_rows-1 as the raw rows. Plotting it with no
            # x put it on the VERTEX index (0..936 for 40 rows) while the
            # forecast/truth overlays continue in rows -- squashing a
            # forecast 24x at the far end (1.1 release review)
            _xs = row_index_x(raw.shape[0], data[i].shape[0])
            _plot_possibly_split(
                ax, (_xs, data[i][:, 0]),
                (np.arange(raw.shape[0], dtype=float), raw[:, 0]), i)
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

    _call_labels = []

    def _own_labels():
        """The ``(annotation, x, y[, z])`` entries `annotate_plot` recorded
        for THIS call (the axes itself holds the labels of every call that
        drew into it, `_watch_labels`)."""
        return _call_labels[0] if _call_labels else []

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

        # this call's labels. `_own_labels` reads this; the axes gets
        # them added to its own list below (`_watch_labels`). The module
        # global `labels_and_points` still shows the LAST annotated
        # plot's ``(label, x, y[, z])`` entries, as it always has, but
        # through weak references (`_publish_labels`): holding the
        # labels themselves kept the last labeled figure alive after it
        # was closed (1.2.1 review round 5).
        labels_and_points = []
        _publish_labels(labels_and_points)
        _call_labels[:] = [labels_and_points]

        if lengths is not None:
            within = [j for L in lengths for j in range(int(L))]
            # ...and which drawn trace each label belongs to, so an
            # animation can test it against THAT trace's drawn window
            run_of = [r for r, L in enumerate(lengths) for _ in range(int(L))]
        else:
            within = list(range(len(data)))
            run_of = None

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
                        xytext=(-LABEL_OFFSET_PT, LABEL_OFFSET_PT),
                        textcoords="offset points",
                        ha="right",
                        va="bottom",
                        **_label_callout_kwargs(label_alpha),
                        **_label_font_kwargs,
                    )
                    label._hyp_offset = (-LABEL_OFFSET_PT, LABEL_OFFSET_PT)
                    label._hyp_point_idx = within[idx]
                    label._hyp_global_idx = idx
                    label._hyp_run_idx = (run_of[idx] if run_of is not None
                                          else None)
                    labels_and_points.append((label, x[0], x[1], x[2]))
                elif data[0].shape[-1] == 2:
                    x2, y2 = x[0], x[1]
                    label = ax.annotate(
                        labels[idx],
                        xy=(x2, y2),
                        xytext=(-LABEL_OFFSET_PT, LABEL_OFFSET_PT),
                        textcoords="offset points",
                        ha="right",
                        va="bottom",
                        **_label_callout_kwargs(label_alpha),
                        **_label_font_kwargs,
                    )
                    label._hyp_offset = (-LABEL_OFFSET_PT, LABEL_OFFSET_PT)
                    label.draggable()
                    label._hyp_point_idx = within[idx]
                    label._hyp_global_idx = idx
                    label._hyp_run_idx = (run_of[idx] if run_of is not None
                                          else None)
                    labels_and_points.append((label, x[0], x[1]))
        # labels that would cover each other are moved apart on every
        # draw (GH #297), static, rotated by hand or animated alike
        _watch_labels(ax, labels_and_points)
        _publish_labels(labels_and_points)
        fig.canvas.draw()

    def _sync_anim_labels(num, window_frames, all_visible=False, revealed=None,
                          hide_all=False, windows=None):
        """Per-animation-frame label bookkeeping (QC 2026-07): show each
        per-point label ONLY while its datapoint is currently drawn (previously
        every label was drawn on every frame), and reproject the visible ones
        for the (possibly rotated) camera. The visibility rule depends on the
        animation style:

        * window / parallel: the datapoint is inside the head window its
          trace was JUST drawn over -- ``windows[run]``, the ``(start,
          end)`` row bounds the updater sliced the artist with (matched on
          ``_hyp_point_idx``, the within-trace row, and ``_hyp_run_idx``).
          A trace's rows are not frames (1.1 visual review L8: every line
          keeps its observations on a grid of at least one row per frame),
          so the historical ``[num - window_frames, num]`` rule, still the
          fallback without `windows`, showed labels at the wrong time;
        * serial: the datapoint has been REVEALED, i.e. its global index
          (``_hyp_global_idx``) ``<= revealed`` (serial accumulates points, so
          there is no trailing edge);
        * spin (``all_visible=True``): every point is always drawn;
        * morph (``hide_all=True``): the single traveling cloud does not
          correspond to the original labeled points, so labels are hidden for
          the duration of the morph.
        """
        # this call's labels (none if `annotate_plot` did not run), and
        # of them only THIS axes' labels
        laps = [e for e in _own_labels()
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
                r = getattr(label, "_hyp_run_idx", None)
                if j is None:
                    visible = True
                elif windows is not None and r is not None \
                        and r < len(windows):
                    visible = windows[r][0] <= j < windows[r][1]
                else:
                    visible = lo <= j <= num
            label.set_visible(visible)
            if visible and is_3d:
                x2, y2, _ = proj3d.proj_transform(entry[1], entry[2], entry[3],
                                                  proj)
                label.xy = (x2, y2)
        # (the draw of the frame works out where the label's text and
        # arrow go from that; doing it here as well cost a millisecond
        # a label on every frame)
        # ...and each label where the animation's table has it on this
        # frame (`solve_label_tracks`)
        _apply_label_tracks(ax, num)

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
                if any(isinstance(el, (list, tuple)) for el in labels):
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
            # a nested per-dataset labels= may be a tuple of tuples as well
            # as a list of lists (the validator accepts both); flatten
            # either, or the tuple is drawn as its literal repr.
            if any(isinstance(el, (list, tuple)) for el in labels):
                labels = list(itertools.chain(*labels))
            annotate_plot(X, labels, lengths=lengths)
            _connect_label_events(fig, ax)

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
        # put the label on the side of the point facing the axes center, so a
        # point near an edge never pushes its label out of the window
        fx, fy = ax.transAxes.inverted().transform(
            ax.transData.transform((x2, y2)))
        annotate_plot_explore.label = ax.annotate(
            label,
            xy=(x2, y2),
            xytext=(-20 if fx > 0.5 else 20, 20 if fy < 0.5 else -20),
            textcoords="offset points",
            ha="right" if fx > 0.5 else "left",
            va="bottom" if fy < 0.5 else "top",
            **_label_callout_kwargs(0.5, arrowstyle="->", facecolor="yellow"),
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
                (-scale, -scale),
                2 * scale,
                2 * scale,
                **square_kwargs
            )
        )

    def frame_2d(ax):
        """Draw the 2-D frame and set the 2-D axis limits for `axis_scale`.

        ``'unit'`` (the default, and everything drawn before GH #285) draws
        hypertools' frame square (half-width `UNIT_FRAME_SCALE`) and pins both axes to ``+-UNIT_FRAME_LIMIT``,
        because `plot()` has already rescaled the data into ``[-1, 1]``.
        ``'data'`` draws no square and applies `xlim`/`ylim` when `plot()`
        computed (or the caller passed) them, leaving matplotlib's autoscale
        in charge otherwise. Called by BOTH the static 2-D path and
        `animate_plot2D`, so the two cannot drift apart.
        """
        if axis_scale != 'data':
            plot_square(ax, scale=UNIT_FRAME_SCALE, **frame_kwargs)
            # where a legend placed by name goes (`legend_call_kwargs`)
            ax._hyp_frame_box = (-UNIT_FRAME_SCALE, -UNIT_FRAME_SCALE,
                                 2 * UNIT_FRAME_SCALE, 2 * UNIT_FRAME_SCALE)
            ax.set_xlim(-UNIT_FRAME_LIMIT, UNIT_FRAME_LIMIT)
            ax.set_ylim(-UNIT_FRAME_LIMIT, UNIT_FRAME_LIMIT)
            return
        ax._hyp_frame_box = None
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
        ax.set_box_aspect(
            None, zoom=_anim_box_zoom(zoom) * _cell_zoom_scale[0])

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
        _holders = (None if _windows is None else _head_holders(
            [win.reached for win in _windows]))
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
            _place_head_marker(line, draw_window,
                               _holders is None or _holders[i])
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

        # per-point labels track their datapoint's visibility: the SAME row
        # window each head line was just sliced with (L8)
        _sync_anim_labels(num, tail_duration, windows=head_bounds)
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
        ax.set_box_aspect(
            None, zoom=_anim_box_zoom(zoom) * _cell_zoom_scale[0])

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
        ax.set_box_aspect(
            None, zoom=_anim_box_zoom(zoom) * _cell_zoom_scale[0])

        lengths = [d.shape[0] for d in data_lines]
        total_points = sum(lengths)
        revealed = total_points * num / max(1, total_frames - 1)
        _counts = serial_reveal_counts(lengths, num, total_frames)
        _holders = _serial_head_holders(_counts, lengths)

        windows = []
        window_spcs = []
        _head_windows = []                                  # GH #285
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
            if trail is not None and trail_bounds is None:
                # no trail on this frame: say so on the artist as well, or
                # a renderer that redraws it from its recorded rows
                # (`plot._apply_multicolor_animation`) keeps showing the
                # trail of the frame drawn before
                trail._hyp_row_window = (0, 0)
            _head_windows.append(tuple(int(b) for b in head_bounds))
            line.set_data(head[:, 0:2].T)
            line.set_3d_properties(head[:, 2])
            # (frame 0 of a serial reveal has drawn nothing yet: the
            # head marker waits at the first dataset's first point, as a
            # parallel reveal's frame 0 shows it there)
            _place_head_marker(
                line, head if len(head) or not _holders[i] else data[:1],
                _holders[i])
            _set_serial_linewidth(line, trail_bounds is not None)
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
                # the head window each artist was JUST drawn over: start
                # is 0 for a plain cumulative reveal and moves past the
                # dataset's beginning once a trail flag gives the reveal
                # a comet-head (GH #285; `FrameContext.window_bounds`).
                window_bounds=tuple(_head_windows))
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
        ax.set_box_aspect(
            None, zoom=_anim_box_zoom(zoom) * _cell_zoom_scale[0])

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

    def _own_color(kw, idx, head):
        """`kw` (a trail's plot kwargs) with the color of the dataset's
        head line, unless `kw` or the dataset's format string names one.
        Left to the axes' color cycle a trail took the next color after
        the head lines', which is its dataset's only when EVERY dataset
        has a trail: ``chemtrails=[False, True]`` drew dataset 1's trail
        in dataset 0's color (found with GH #299)."""
        if "color" in kw or "c" in kw:
            return kw
        fmt_color = None
        if isinstance(_fmt_at(idx), str) and _process_plot_format is not None:
            try:
                fmt_color = _process_plot_format(_fmt_at(idx))[2]
            except Exception:  # noqa: BLE001 - matplotlib reports it itself
                fmt_color = None
        if fmt_color is None:
            kw["color"] = head.get_color()
        return kw

    def _head_linewidths(linewidths, style=None):
        """The line width of each dataset's HEAD line (the recent window):
        `tail_linewidth` when given (GH #299), else the dataset's own. A
        `style` with no reveal ('spin' draws every dataset in full, 'morph'
        its own cloud) has no tail, so the lines keep their own width
        there, as the warning `plot()` gives for it says."""
        if tail_linewidth is None or style in ("spin", "morph"):
            return list(linewidths)
        return [tail_linewidth] * len(linewidths)

    def _remember_linewidths(lines, linewidths, head_linewidths):
        for line, base, head in zip(lines, linewidths, head_linewidths):
            line._hyp_linewidths = (base, head)

    def _set_serial_linewidth(line, revealing):
        """A serial reveal draws a finished dataset with its head line, in
        full: `tail_linewidth` applies to that line only while it is the
        comet head of the dataset being revealed (GH #299)."""
        widths = getattr(line, '_hyp_linewidths', None)
        if widths is not None and widths[0] != widths[1]:
            line.set_linewidth(widths[1] if revealing else widths[0])

    def _add_head_markers(ax, lines, head_linewidths, has_head):
        """Create each dataset's `head_marker=` artist (GH #299): ONE
        marker, placed at the last vertex of the dataset's head line on
        every frame (`_place_head_marker`). Its color, alpha and size
        follow the dataset unless the spec styles them: the head line's
        color, the dataset's ``alpha=``, and the dataset's ``markersize=``
        (matplotlib's default marker size without one) -- but never less
        than 2.5 times the head line's width, so the marker shows on a
        thick tail. Never a legend entry."""
        if head_marker is None or not has_head:
            return
        for idx, (line, spec) in enumerate(zip(lines, head_marker)):
            if spec is None:
                continue
            dataset_kw = (kwargs_list[idx]
                          if isinstance(kwargs_list[idx], dict) else {})
            props = dict(spec)
            pinned = 'color' in props
            props.setdefault('color', line.get_color())
            if dataset_kw.get('alpha') is not None:
                props.setdefault('alpha', dataset_kw['alpha'])
            props.setdefault('markersize', max(
                float(dataset_kw.get('markersize')
                      or plt.rcParams['lines.markersize']),
                2.5 * float(head_linewidths[idx])))
            props.setdefault('zorder', line.get_zorder() + 1)
            empty = ([], [], []) if hasattr(ax, 'get_proj') else ([], [])
            (artist,) = ax.plot(*empty, linestyle='None', **props)
            artist.set_label('_nolegend_')
            artist.set_clip_on(False)
            artist.set_in_layout(False)
            artist._hyp_trace_index = idx
            artist._hyp_trace_role = 'head_marker'
            artist._hyp_color_pinned = pinned
            line._hyp_head_marker = artist
            # hidden with its line (a callback holds the line alone)
            follow_artist(ax, artist, line)

    def _head_holders(reached):
        """Which drawn runs carry their dataset's `head_marker=` on this
        frame, given which of them the reveal has `reached`: ONE per
        input dataset, the run its current point is in.

        `hue=` categories and `cluster=` cut a dataset into runs, each
        its own line with its own marker artist; every run the reveal had
        passed kept its marker at its last point, so a dataset in two
        categories ended with two head markers, one stuck where the first
        category stops (1.2.1 review round 4). The marker of the run that
        holds the head is drawn, in that run's (the category's) color."""
        reached = [bool(flag) for flag in reached]
        if ownership is None or ownership.n_runs != len(reached):
            return reached
        holds = [False] * len(reached)
        for dataset in range(ownership.n_datasets):
            runs = [r for r in ownership.runs_of(dataset) if reached[r]]
            if runs:
                holds[max(runs, key=lambda r: ownership.run_span(r)[0])] = True
        return holds

    def _serial_head_holders(counts, lengths):
        """`_head_holders` for a serial reveal, where a run is reached
        once one of its rows is revealed. Before anything is (frame 0),
        the first dataset that has rows holds its marker, at its first
        point: frame 0 was an empty frame while a parallel reveal's
        frame 0 already showed the head (1.2.1 review round 4)."""
        reached = [count > 0 for count in counts]
        if not any(reached):
            first = next((i for i, n in enumerate(lengths) if n > 0), None)
            return [i == first for i in range(len(reached))]
        return _head_holders(reached)

    def _place_head_marker(line, points, holds=True):
        """Move `line`'s `head_marker=` artist (if it has one) to the last
        of the `points` the line was just given; no points, no marker.
        Nor when the line is not the run that `holds` its dataset's head
        (`_head_holders`)."""
        artist = getattr(line, '_hyp_head_marker', None)
        if artist is None:
            return
        last = np.asarray(points)[-1:] if holds else np.asarray(points)[:0]
        artist.set_data(last[:, 0], last[:, 1])
        if last.shape[1] >= 3 and hasattr(artist, 'set_3d_properties'):
            artist.set_3d_properties(last[:, 2])

    def _new_animation(fig, func, n_frames, fargs):
        """The animation object for frame function `func`: the
        `FuncAnimation` an ordinary animated plot returns, or -- for one
        cell of an animated panel grid (`panel_cell=True`) -- a
        `PanelAnimationStub` carrying the same function, arguments, frame
        count and interval for the grid's single animation to drive.

        `hold_frames` (``hold=``, GH #299) repeats the last frame at the
        end: the frame SEQUENCE becomes ``0..n-1`` followed by ``n-1``
        that many times (`held_frame_sequence`), so the frame function --
        and every wrapper `plot()` puts around it later -- is simply
        called with the last index again and nothing downstream has to
        know about the hold. A panel cell never holds: its grid does."""
        if _cell_ax is not None:
            return PanelAnimationStub(func, fargs, n_frames,
                                      1000 / frame_rate,
                                      zoom_scale=_cell_zoom_scale)
        line_ani = HyperFuncAnimation(
            fig, func, held_frame_sequence(n_frames, hold_frames),
            fargs=fargs, interval=1000 / frame_rate, blit=False,
            repeat=False)
        line_ani._hyp_content_frames = int(n_frames)
        # the factor every 3-D frame function multiplies its box zoom by:
        # `plot._keep_cube_clear` lowers it when the turning cube would
        # reach the canvas edge or a legend (GH #297)
        line_ani._hyp_zoom_scale = _cell_zoom_scale
        return line_ani

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
        if _cell_ax is not None:
            # one cell of an animated panel grid: the grid made the figure
            # and lays its cells out itself
            ax = _cell_ax
            fig = ax.figure
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            # give the (axis-off) 3D axes the full canvas: with the default
            # subplot margins, the zoomed cube overflows the axes viewport
            # at some rotation angles, clipping cube corners and data
            ax.set_position([0.0, 0.0, 1.0, 1.0])

        # create lines.
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
        def _trail_kwargs(kw, idx):
            kw = dict(kw) if isinstance(kw, dict) else {}
            kw["alpha"] = trail_alpha * kw.pop("alpha", 1.0)
            return _own_color(kw, idx, lines[idx])

        head_linewidths = _head_linewidths(linewidths, style)

        # GH #299: a trail artist is created for EVERY dataset that asks for
        # one, whatever its format. It used to be gated on `is_line` -- "is
        # this a PURE line format" -- so a line+marker format ('-o', 'o-',
        # '.-', '--s') silently got no chemtrails/precog/bullettime trail at
        # all, and a markers-only dataset got its trail of faded markers
        # only when some OTHER dataset of the call happened to be a pure
        # line. The trail is drawn in the dataset's own format, exactly as
        # its head is: a line+marker trail carries faded markers at the
        # observations (`_mark_observations`), a markers-only trail is its
        # faded markers -- what the plotly backend has always drawn.
        def _fmt_args(idx):
            return (fmt[idx],) if fmt is not None else ()

        lines = [
            ax.plot(dat[0:1, 0], dat[0:1, 1], dat[0:1, 2], *_fmt_args(idx),
                    linewidth=head_linewidths[idx], **kwargs_list[idx])[0]
            for idx, dat in enumerate(x)
        ]
        trail = [
            ax.plot(dat[0:1, 0], dat[0:1, 1], dat[0:1, 2], *_fmt_args(idx),
                    linewidth=linewidths[idx],
                    **_trail_kwargs(kwargs_list[idx], idx))[0]
            if _wants_trail(idx) else None
            for idx, dat in enumerate(x)
        ]
        _remember_linewidths(lines, linewidths, head_linewidths)
        # head_marker= (GH #299): created AFTER the trails, so the head
        # lines and then the trails stay the axes' first line artists
        # (`plot._apply_multicolor_animation` finds them by position)
        _add_head_markers(ax, lines, head_linewidths,
                          style not in ("spin", "morph"))
        for _idx, _trail_line in enumerate(trail):
            if _trail_line is not None:
                _trail_line.set_label('_nolegend_')
                # which dataset the trail belongs to: with a per-dataset
                # flag list not every dataset has one, so its place among
                # the trails does not say
                _trail_line._hyp_trace_index = _idx
                _trail_line._hyp_trace_role = 'trail'

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
            # the shared default both backends read (L9: 1.5 pt dots were
            # ~3 px here and sub-pixel on plotly)
            morph_markersize = (_mkw.get("markersize")
                                or _morph.MORPH_DEFAULT_MARKERSIZE_PT)
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
            line_ani = _new_animation(
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
            )
        elif style == "serial":
            line_ani = _new_animation(
                fig,
                update_lines_serial,
                max(1, int(round(frame_rate * duration))),
                fargs=(x, lines, trail, cube_scale_anim, window_frames,
                       rotations, zoom, chemtrails, precog, bullettime, elev),
            )
        elif style == "spin":
            line_ani = _new_animation(
                fig,
                update_lines_spin,
                max(1, int(round(frame_rate * duration))),
                fargs=(x, lines, cube_scale_anim, rotations, zoom, elev),
            )
        elif style == "morph":
            n_morph_datasets = len(morph_state["indices"])
            total_frames = max(1, int(round(frame_rate * duration)))
            frame_counts, _, azimuths = _morph.morph_schedule(
                n_morph_datasets, total_frames, rotations, azim)
            morph_state["frame_counts"] = frame_counts
            line_ani = _new_animation(
                fig,
                update_morph,
                sum(frame_counts),
                fargs=(morph_state, cube_scale_anim, azimuths, zoom, elev),
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
        _holders = (None if _windows is None else _head_holders(
            [win.reached for win in _windows]))
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
            _place_head_marker(line, window,
                               _holders is None or _holders[i])

        _sync_anim_labels(num, tail_duration, windows=head_bounds)
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
        _holders = _serial_head_holders(_counts, lengths)

        _head_windows = []                                  # GH #285
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
            if trail is not None and trail_bounds is None:
                # no trail on this frame: say so on the artist as well, or
                # a renderer that redraws it from its recorded rows
                # (`plot._apply_multicolor_animation`) keeps showing the
                # trail of the frame drawn before
                trail._hyp_row_window = (0, 0)
            _head_windows.append(tuple(int(b) for b in head_bounds))
            line.set_data(head[:, 0], head[:, 1])
            # (frame 0 of a serial reveal has drawn nothing yet: the
            # head marker waits at the first dataset's first point, as a
            # parallel reveal's frame 0 shows it there)
            _place_head_marker(
                line, head if len(head) or not _holders[i] else data[:1],
                _holders[i])
            _set_serial_linewidth(line, trail_bounds is not None)
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
                window_bounds=tuple(_head_windows))
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
        if _cell_ax is not None:
            # one cell of an animated panel grid (see animate_plot3D)
            ax = _cell_ax
            fig = ax.figure
        else:
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
        def _trail_kwargs(kw, idx):
            kw = dict(kw) if isinstance(kw, dict) else {}
            kw["alpha"] = trail_alpha * kw.pop("alpha", 1.0)
            return _own_color(kw, idx, lines[idx])

        head_linewidths = _head_linewidths(linewidths, style)

        # every flagged dataset gets its trail, whatever its format (GH #299:
        # see animate_plot3D)
        def _fmt_args(idx):
            return (fmt[idx],) if fmt is not None else ()

        lines = [
            ax.plot(dat[0:1, 0], dat[0:1, 1], *_fmt_args(idx),
                    linewidth=head_linewidths[idx], **kwargs_list[idx])[0]
            for idx, dat in enumerate(x)
        ]
        trail = [
            ax.plot(dat[0:1, 0], dat[0:1, 1], *_fmt_args(idx),
                    linewidth=linewidths[idx],
                    **_trail_kwargs(kwargs_list[idx], idx))[0]
            if _wants_trail(idx) else None
            for idx, dat in enumerate(x)
        ]
        _remember_linewidths(lines, linewidths, head_linewidths)
        # head_marker= (GH #299): created AFTER the trails, so the head
        # lines and then the trails stay the axes' first line artists
        # (`plot._apply_multicolor_animation` finds them by position)
        _add_head_markers(ax, lines, head_linewidths,
                          style not in ("spin", "morph"))
        for _idx, _trail_line in enumerate(trail):
            if _trail_line is not None:
                _trail_line.set_label('_nolegend_')
                # which dataset the trail belongs to: with a per-dataset
                # flag list not every dataset has one, so its place among
                # the trails does not say
                _trail_line._hyp_trace_index = _idx
                _trail_line._hyp_trace_role = 'trail'

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
            morph_markersize = (_mkw.get("markersize")
                                or _morph.MORPH_DEFAULT_MARKERSIZE_PT)
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
            line_ani = _new_animation(
                fig,
                update_lines_parallel_2d,
                max(1, int(round(frame_rate * duration))),
                fargs=(x, lines, trail, window_frames, chemtrails, precog,
                      bullettime),
            )
        elif style == "serial":
            line_ani = _new_animation(
                fig,
                update_lines_serial_2d,
                max(1, int(round(frame_rate * duration))),
                fargs=(x, lines, trail, window_frames, chemtrails, precog,
                      bullettime),
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
            line_ani = _new_animation(
                fig,
                update_morph_2d,
                sum(frame_counts),
                fargs=(morph_state,),
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
    # spines are dropped, the way a plain time-series panel is drawn. The
    # gridlines seaborn's 'whitegrid' style (applied by `plot()`) puts on
    # hypertools' own axes are removed, matching the plotly backend's
    # `showgrid=False` (1.1 release review: the matplotlib data-scale plot
    # drew a gray grid the plotly one never had). A caller's `ax=` keeps
    # whatever grid its owner gave it.
    # the font stack in force NOW (`plot()` draws inside the rc_context
    # that sets it), given explicitly to the axis labels and title below:
    # a caller's `ax=` (every `panels=` cell) created its label and title
    # Text artists outside that context, with the bare 'sans-serif' alias,
    # which resolves at DRAW time to the caller's rcParams -- DejaVu Sans
    # beside hypertools' own Noto Sans figures (1.1 release review). On
    # hypertools' own axes this is the family they were created with.
    _text_family = list(plt.rcParams['font.family'])
    if axis_scale == 'data':
        for _side in ('top', 'right'):
            if _side in ax.spines:
                ax.spines[_side].set_visible(False)
        if not _caller_ax:
            ax.grid(False)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontfamily=_text_family)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontfamily=_text_family)
        if x_date:
            # the x column holds `date2num` day numbers (see `plot()`'s
            # ndims=1 series mode); without a date converter they would tick
            # as five-digit floats
            ax.xaxis_date()
            # ...and matplotlib's default date formatter writes every tick
            # as a full 'YYYY-MM-DD', which collide at the default figure
            # size (7 of 7 adjacent pairs overlapped for a 30-day index;
            # 1.1 release review, F13). The concise formatter writes only
            # what changes between ticks, with the rest once as an offset,
            # which is also what plotly's date axis draws.
            import matplotlib.dates as mdates
            _locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(_locator)
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(_locator))
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
            ax.set_xlabel(xlabel, fontfamily=_text_family)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontfamily=_text_family)
        if zlabel is not None:
            ax.set_zlabel(zlabel, fontfamily=_text_family)
        # `Axes3D.get_tightbbox` measures its axes "for layout only", which
        # drops the axis LABELS (matplotlib's `_get_tightbbox_for_layout_
        # only`), so a `bbox_inches='tight'` save -- every notebook's inline
        # render -- cut the z-label off at the right edge (1.1 release
        # review, feature-tour 9.15). A draw-nothing figure artist whose
        # extent is the labels' puts them back into the figure's tight bbox.
        if not any(isinstance(a, _AxisLabelExtent) and a.axes_ref is ax
                   for a in ax.figure.artists):
            ax.figure.add_artist(_AxisLabelExtent(ax))
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
            ax.set_xlabel(xlabel, fontfamily=_text_family)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontfamily=_text_family)
        # zlabel on a 2-D/1-D plot is rejected upstream in plot.py
        # (ValueError, before the pipeline even runs) -- zlabel is
        # guaranteed None here.

    # add labels
    add_labels(x, labels, explore=explore)

    # add title
    if title is not None:
        _apply_title(ax, title, font=font, title_kwargs=title_kwargs,
                     family=_text_family)

    # add legend: to the RIGHT of the plot, vertically centered on the
    # box (never overlapping the data). `prop=font` (GH #205) applies the
    # SAME resolved font to every legend text entry -- `_fit_right_legend`
    # (plot.py) measures the legend's true extent from these Text artists'
    # own fontproperties, so this also fixes multibyte legend clipping
    # without any change needed there.
    if legend is not None or legend_entries is not None:
        _legend_call = legend_call_kwargs(
            is_3d=hasattr(ax, "get_proj"), zlabel=zlabel, font=font,
            legend_kwargs=legend_kwargs, ax=ax)
        if legend_entries is not None:
            # explicit entries (matrix/mixture hue palette swatches, or
            # `legend_colors=[(label, color), ...]`): proxy handles, since
            # no drawn artist carries these labels.
            show_legend_keys(ax.legend(
                handles=_legend_proxy_handles(legend_entries, fmt),
                **_legend_call))
        else:
            if legend_order:
                _handles, _labels = ax.get_legend_handles_labels()
                _rank = {str(lbl): k for k, lbl in enumerate(legend_order)}
                _slots = [j for j, lbl in enumerate(_labels) if lbl in _rank]
                _sorted = sorted(_slots, key=lambda j: _rank[_labels[j]])
                if _sorted != _slots:
                    _perm = list(range(len(_labels)))
                    for _slot, _src in zip(_slots, _sorted):
                        _perm[_slot] = _src
                    _legend_call = dict(
                        _legend_call, handles=[_handles[j] for j in _perm],
                        labels=[_labels[j] for j in _perm])
            _legend_artist = show_legend_keys(ax.legend(**_legend_call))
            if legend_colors is not None:
                _recolor_legend_handles(_legend_artist, legend_colors)

    if size is not None:
        fig.set_size_inches(size)

    if line_ani is not None and not isinstance(line_ani, PanelAnimationStub):
        _make_save_dpi_safe(line_ani)

    return fig, ax, data, line_ani
