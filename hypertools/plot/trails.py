#!/usr/bin/env python
"""Shared per-dataset trail-flag broadcasting for the ``chemtrails``/
``precog``/``bullettime`` ``plot()`` kwargs (GH #127).

Each of ``chemtrails``/``precog``/``bullettime`` accepts a bool (broadcast to
every dataset) or a list/tuple of bool (one entry per FINAL -- post
cluster/hue-reshape -- drawn dataset), mirroring ``surface=``'s per-dataset
list form (see :func:`hypertools.plot.surface.broadcast_surface`). Unlike
``surface=``/``density=``, there is no dict form here -- these kwargs have
always been plain booleans, just now optionally per-dataset.

This module also owns :func:`anim_window_bounds`, the frame -> row mapping
that paces the parallel/'window' head window AND its trail for one dataset.
It lives here rather than in either backend because BOTH backends call it:
head and trail geometry are the same question, and a shared callee is the
only arrangement in which the two renderers cannot drift apart.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

__all__ = ["broadcast_trail_flag", "anim_window_bounds", "RunWindow",
           "dataset_window_bounds", "run_head_param"]


def anim_window_bounds(num, total_frames, n_points, window_frames):
    """Map animation frame `num` (of `total_frames`) onto one dataset's
    row indices for the parallel/'window' styles.

    Animations are paced by the FRAME grid (``total_frames ==
    round(frame_rate * duration)``), not by any single dataset's row count:
    line datasets are pre-interpolated onto that exact grid by ``plot.py``
    (identity mapping), while marker-only and 1-point datasets keep their
    raw rows and are paced here instead (release-1.0 audit: F04-003
    multi-dataset truncation, F04-005/F05-010 marker-only pacing, F05-012
    single-point datasets). The identity holds for every frame count but
    ONE: a request so short that ``round(frame_rate * duration)`` falls
    below 2 still resamples lines to 2 rows, because PCHIP needs two
    samples to interpolate between. Such a dataset takes the rescale branch
    below like any other off-grid one, and both backends still agree,
    because both consume the same resampled array.

    BOTH backends call this one function, per dataset, per frame -- that is
    the point of it living here rather than inside either backend. The
    plotly renderer used to carry its own transcription of this arithmetic
    that computed ONE window per frame from the LONGEST dataset and merely
    clamped shorter datasets into it; a 5-row marker dataset plotted beside
    a 15-row line went blank for 9 of its 15 frames because the shared
    window slid past its end, while matplotlib kept a correctly-paced
    2-point window alive to the last frame. Two smaller transcription drifts
    rode along: a missing ``- 1`` in ``start`` (every steady-state frame one
    point short) and a ``max(2, ...)`` floor on ``end`` where matplotlib
    floors at 1 (frame 0 of a ``precog`` trail one point short). A shared
    callee cannot drift from itself.

    Parameters
    ----------
    num : int
        Current frame index, ``0 <= num < total_frames``.
    total_frames : int
        Total number of animation frames.
    n_points : int
        This dataset's row count.
    window_frames : int
        The opaque head window's length in frames.

    Returns
    -------
    tuple of (int, int, int)
        ``(start, end, trail_stop)``: the head window is ``data[start:end]``,
        frozen at the trajectory's end once the dataset is fully revealed --
        a shorter dataset never vanishes mid-animation. Its length tops out
        at ``w + 1`` rows, where ``w`` is ``window_frames`` for a dataset
        already on the frame grid and the RESCALED
        ``round(window_frames * n_points / total_frames)`` for one that is
        not: a 5-row dataset in a 15-frame animation with
        ``window_frames=2`` gets ``w = 1``, so its head maxes out at 2 rows,
        not 3. A chemtrails trail is
        ``data[0:trail_stop]`` -- 0 rows until the head window actually
        starts sliding (F05-001: the historical ``num - window + 1`` stop
        went NEGATIVE for early frames, so Python's negative indexing drew
        nearly the whole FUTURE trajectory as a "past" trail, then blinked
        empty). A precog trail is ``data[end - 1:]`` (sharing the head's
        last vertex, so there is no one-segment gap -- F05-008).
    """
    total = max(1, int(total_frames))
    end = int(np.ceil((num + 1) * n_points / total))
    end = max(1, min(n_points, end))
    if n_points == total:
        w = int(window_frames)
    else:
        # rescale the window (given in frames) onto this dataset's rows
        w = int(round(window_frames * n_points / total))
    start = max(0, end - 1 - w)
    trail_stop = max(0, end - w)
    return start, end, trail_stop


@dataclass(frozen=True)
class RunWindow:
    """What one drawn run shows at one frame.

    Four named bounds rather than the ``(start, end, trail_stop)`` triple
    `anim_window_bounds` returns, because a run that the dataset's clock has
    not reached yet needs a FOURTH state the triple cannot express. The
    historical precog slice is ``data[end - 1:]``; with ``end == 0`` that is
    ``data[-1:]`` -- one point of a not-yet-revealed category sitting on
    screen from frame 0. Naming the future bound separately makes that slice
    unwritable.

    That zero comes from the PROJECTION, not from `anim_window_bounds`, which
    clamps ``end = max(1, min(n_points, end))`` (`trails.py:86`) and so cannot
    return 0 for any real run -- swept over every frame of 7488 ``(total,
    grid, window)`` combinations, zero hits. It is `count_from` in
    `dataset_window_bounds` that returns 0, for a run the dataset's clock has
    not yet reached, and that is exactly the state the triple could not name.

    Attributes
    ----------
    head_start, head_end : int
        The opaque head is ``data[head_start:head_end]``. Both 0 for a run the
        clock has not reached; both `g_run` for a run the sliding window has
        moved past.
    past_stop : int
        A chemtrails trail is ``data[0:past_stop]``. 0 until the head window
        actually starts sliding (F05-001).
    future_start : int
        A precog trail is ``data[future_start:]``, sharing the head's last
        vertex so there is no one-segment gap (F05-008). 0 -- the WHOLE run --
        for a run the clock has not reached: all of it is still ahead.
    reached : bool
        Whether the dataset's clock has entered this run at all -- defined as
        ``head_end > 0``, i.e. the run has at least one drawn vertex on
        screen. Deliberately NOT a second comparison of the head parameter
        against the run's first row: those two agree today only because
        `_param` returns 0 for a dataset with no extent (a ONE-ROW dataset)
        and the first run's `first_row` is also 0, so a degenerate value and
        a real boundary coincide. Two derivations of one fact can drift;
        `head_end` already carries it. (The two were measured equal over 1116
        windows before the substitution, so this changes no behaviour.)
    grid : int
        The run's drawn row count. Carried so `run_head_param` can invert the
        projection from the window ALONE -- the reveal schedule must read the
        head position back off the objects the backends actually sliced with,
        not recompute it from the frame index, or the two can drift.
    """

    head_start: int
    head_end: int
    past_stop: int
    future_start: int
    reached: bool
    grid: int


def _param(idx, g, span):
    """Grid-row index -> source-parameter offset within a run or dataset.

    `plot._interp_anim_line` resamples `n` source rows onto
    ``linspace(0, n - 1, g)`` with exact endpoints, so grid row `idx` sits at
    source parameter ``idx * span / (g - 1)``. Exact rational arithmetic, not
    float: the unregrouped case must project to the IDENTITY, and it does so
    only if the round trip cancels exactly.
    """
    if g < 2 or span <= 0:
        return Fraction(0)
    return Fraction(int(idx) * int(span), int(g) - 1)


def dataset_window_bounds(num, total_frames, ownership, grid_lengths,
                          window_frames):
    """One `RunWindow` per RUN, from ONE clock per source dataset.

    `hue=`/`cluster=` cut each input dataset into contiguous same-category
    runs, each drawn as its own trace and each resampled onto the same frame
    grid. Pacing every trace with `anim_window_bounds` then advances all of
    one dataset's runs at once, so a single trajectory animates in several
    disjoint time windows simultaneously rather than sweeping once (measured
    on `dev-1.0`: three runs of a 30-row dataset all at 247 points on frame 3
    of 12). Driving them from the dataset's own clock and projecting onto each
    run restores the sweep: earlier runs are complete, one run holds the head,
    later runs are empty.

    `anim_window_bounds` is CALLED here rather than reimplemented, so the
    window rescaling, the F05-001 negative-chemtrails clamp and the F05-008
    precog overlap keep exactly one implementation -- the same reason both
    backends share that function.

    The projection goes through the SOURCE PARAMETER, not through row counts,
    and uses each run's DRAWN span (`TraceOwnership.draw_span`, which counts
    the bridge vertex `patch_lines` appended). Both choices are load-bearing:
    quantizing to source rows first double-rounds (76 grid rows became 101 in
    a 9-row, 12-frame check), and using the OWNED span instead of the drawn
    one desynchronizes every category boundary by one vertex, leaving an
    observation on screen that the reveal schedule reports as invisible.

    Parameters
    ----------
    num, total_frames : int
        Frame index and count, as for `anim_window_bounds`.
    ownership : hypertools.plot.ownership.TraceOwnership
        Which run came from which dataset, from which of its rows, and
        whether it carries a bridge vertex.
    grid_lengths : sequence of int
        Each RUN's drawn row count (post-interpolation).
    window_frames : int
        The opaque head window's length in frames.

    Returns
    -------
    list of RunWindow
        Indexed by run.
    """
    windows = [None] * ownership.n_runs
    for dataset in range(ownership.n_datasets):
        runs = ownership.runs_of(dataset)
        n_rows = ownership.row_count(dataset)
        g_ref = max(int(grid_lengths[r]) for r in runs)
        start, end, trail_stop = anim_window_bounds(
            num, total_frames, g_ref, window_frames)
        span_ref = n_rows - 1
        p_head = _param(end - 1, g_ref, span_ref)
        p_start = _param(start, g_ref, span_ref)
        p_trail = (None if trail_stop == 0
                   else _param(trail_stop - 1, g_ref, span_ref))
        for r in runs:
            first_row, _ = ownership.run_span(r)
            span = ownership.draw_span(r)
            g_run = int(grid_lengths[r])

            def count_from(p, _a=first_row, _s=span, _g=g_run):
                # a COUNT of drawn grid rows: `data[0:count]`
                if p is None or p < _a:
                    return 0
                if _g < 2 or _s <= 0:
                    # nothing to slide along (a 1-row unbridged run): the
                    # clock either has reached it or has not
                    return _g
                j = min((p - _a) * (_g - 1) / _s, Fraction(_g - 1))
                return min(_g, int(j) + 1)     # int() floors; j >= 0 here

            def index_from(p, _a=first_row, _s=span, _g=g_run):
                # an INDEX into the drawn grid: `data[index:...]`
                if p <= _a:
                    return 0
                if _g < 2 or _s <= 0:
                    return 0
                return min(_g, int((p - _a) * (_g - 1) / _s))

            head_end = count_from(p_head)
            windows[r] = RunWindow(
                head_start=index_from(p_start),
                head_end=head_end,
                past_stop=count_from(p_trail),
                # `head_end == 0` is exactly "this run has nothing drawn", so
                # both of these follow from it and neither re-derives it: an
                # unreached run's precog is its WHOLE future (`data[0:]`, R5)
                # because `max(0, -1)` is 0, and `reached` is the same test.
                future_start=max(0, head_end - 1),
                reached=head_end > 0,
                grid=g_run)
    return windows


def run_head_param(window, ownership, run):
    """Source parameter of a run's DRAWN head, or None if it has none.

    The inverse of the projection in `dataset_window_bounds`, and the reason
    the reveal schedule and the renderer cannot describe different states: a
    dataset's visible rows are read back off the windows that were actually
    produced, never computed a second time from the frame index. Everything
    it needs is on the `RunWindow` and the ownership, so it cannot be handed
    a stale frame number.
    """
    if not window.reached or window.head_end <= 0:
        return None
    first_row, _ = ownership.run_span(run)
    span = ownership.draw_span(run)
    if span <= 0 or window.grid < 2:
        # an all-or-nothing run: its single row IS its head
        return Fraction(first_row)
    return first_row + _param(window.head_end - 1, window.grid, span)


def broadcast_trail_flag(value, n, name):
    """Normalize one of chemtrails/precog/bullettime to exactly `n`
    per-dataset booleans.

    Parameters
    ----------
    value : bool or list/tuple/numpy.ndarray of bool
        The raw kwarg value: a single bool (broadcast to every dataset) or
        a list/tuple/1-D boolean array (one entry per dataset --
        release-1.0 audit F05-005: an ndarray, the natural result of any
        array computation, used to crash with numpy's cryptic "truth value
        of an array is ambiguous" error).
    n : int
        The number of (final, post cluster/hue-reshape) datasets to draw.
    name : str
        The kwarg's name (``'chemtrails'``, ``'precog'``, or
        ``'bullettime'``), used in the error messages.

    Returns
    -------
    list of bool
        Length exactly `n`.

    Raises
    ------
    ValueError
        `value` is a list/tuple/array whose length does not match `n`.
    TypeError
        `value` is a non-bool scalar (e.g. a str or dict -- F05-005: those
        were previously coerced to True silently).
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(
                f"{name} list has {len(value)} entries but there are {n} "
                f"datasets to plot; pass a single bool to apply it to "
                "every dataset, or a list matching the dataset count."
            )
        return [bool(v) for v in value]
    if value is None:
        return [False] * n
    if not isinstance(value, (bool, np.bool_)) and value not in (0, 1):
        raise TypeError(
            f"{name} must be a bool (or a list/tuple/array of bool, one "
            f"entry per dataset); got {type(value).__name__}: {value!r}."
        )
    return [bool(value)] * n
