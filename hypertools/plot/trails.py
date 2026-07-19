#!/usr/bin/env python
"""Shared per-dataset trail-flag broadcasting for the ``chemtrails``/
``precog``/``bullettime`` ``plot()`` kwargs (GH #127).

Each of ``chemtrails``/``precog``/``bullettime`` accepts a bool (broadcast to
every dataset) or a list/tuple of bool (one entry per FINAL -- post
cluster/hue-reshape -- drawn dataset), mirroring ``surface=``'s per-dataset
list form (see :func:`hypertools.plot.surface.broadcast_surface`). Unlike
``surface=``/``density=``, there is no dict form here -- these kwargs have
always been plain booleans, just now optionally per-dataset.
"""

import numpy as np

__all__ = ["broadcast_trail_flag"]


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
