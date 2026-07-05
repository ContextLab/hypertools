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

__all__ = ["broadcast_trail_flag"]


def broadcast_trail_flag(value, n, name):
    """Normalize one of chemtrails/precog/bullettime to exactly `n`
    per-dataset booleans.

    Parameters
    ----------
    value : bool or list/tuple of bool
        The raw kwarg value: a single bool (broadcast to every dataset) or
        a list/tuple of bool (one entry per dataset).
    n : int
        The number of (final, post cluster/hue-reshape) datasets to draw.
    name : str
        The kwarg's name (``'chemtrails'``, ``'precog'``, or
        ``'bullettime'``), used in the ``ValueError`` message.

    Returns
    -------
    list of bool
        Length exactly `n`.

    Raises
    ------
    ValueError
        `value` is a list/tuple whose length does not match `n`.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(
                f"{name} list has {len(value)} entries but there are {n} "
                f"datasets to plot; pass a single bool to apply it to "
                "every dataset, or a list matching the dataset count."
            )
        return [bool(v) for v in value]
    return [bool(value)] * n
