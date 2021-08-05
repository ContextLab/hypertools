#!/usr/bin/env python
import warnings
import matplotlib as mpl

from .backend import _init_backend, set_interactive_backend, reset_backend, manage_backend, contextmanager
from .plot import plot

_init_backend()
