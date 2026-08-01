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

import numpy as np

__all__ = ["broadcast_trail_flag", "anim_window_bounds"]


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
