#!/usr/bin/env python
"""HyperTools: visualize and manipulate high-dimensional data.

The classic API lives at the top level: `plot`, `analyze`, `reduce`,
`align`, `normalize`, `describe`, `cluster`, `manip`, `predict`, `impute`,
`load`, `save`, `apply_model`, `supported_models`, `Pipeline`, and
`set_interactive_backend`, plus the `io` submodule and `HyperAnimation`
(the return type of animated plots). Exceptions raised by hypertools
(`HypertoolsError`, `HypertoolsBackendError`, `HypertoolsIOError`) are
also importable from here.

Import-form note: several top-level functions share a name with the
subpackage they live in, so attribute access like
``hypertools.plot.backend`` resolves against the ``plot`` *function* and
raises AttributeError. ``import hypertools.plot.backend as backend``
fails the same way (the ``as``-binding is resolved via attribute access
on the ``plot`` function) and raises ImportError. Use
``from hypertools.plot import backend`` for submodule access.
"""

from .config import __version__
from .plot.plot import plot
from .plot.backend import set_interactive_backend
from .plot.hyper_animation import HyperAnimation
from .io.load import load
from .tools.analyze import analyze
from .reduce.reduce import reduce
from .align.align import align
from .tools.normalize import normalize
from .reduce.describe import describe
from .cluster.cluster import cluster
from .core.model import apply_model, supported_models
from .core.pipeline import Pipeline
from .core.exceptions import (HypertoolsError, HypertoolsBackendError,
                              HypertoolsIOError)
from .manip.manip import manip
from .predict.predict import predict
from .impute.impute import impute
from .io.save import save
from . import io

#: the public API (2026-07 release audit, X1-014): `from hypertools import *`
#: yields exactly these documented names. Without __all__, star-imports also
#: leaked internal submodules bound as attributes by the imports above
#: (`config`, `core`, `datageometry`, `external`, `tools`, ...).
__all__ = [
    'plot', 'analyze', 'reduce', 'align', 'normalize', 'describe',
    'cluster', 'manip', 'predict', 'impute', 'load', 'save',
    'apply_model', 'supported_models', 'Pipeline',
    'set_interactive_backend', 'HyperAnimation', 'io',
    'HypertoolsError', 'HypertoolsBackendError', 'HypertoolsIOError',
]
