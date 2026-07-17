#!/usr/bin/env python
"""`hypertools.plot` subpackage: the `plot()` dispatcher plus the
matplotlib/plotly drawing backends and their helpers.

Importing this package initializes the matplotlib backend immediately
(`backend._init_backend()`): the interactive GUI backend must be chosen
before pyplot state exists, honoring the ``$HYPERTOOLS_BACKEND``
environment variable (see `backend.set_interactive_backend`).
"""
from .backend import _init_backend


_init_backend()
