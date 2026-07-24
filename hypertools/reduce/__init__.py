"""hypertools.reduce subpackage: the `reduce()` dispatcher, `describe()`,
and the `Reducer` base class it dispatches to.

NOTE: the top-level package re-exports the FUNCTION as `hypertools.reduce`
(shadowing this subpackage as an attribute), but the class remains
importable from here, e.g. ``from hypertools.reduce import Reducer``.
"""
from .common import Reducer
from .reduce import reduce
from .describe import describe

__all__ = ['reduce', 'describe', 'Reducer']
