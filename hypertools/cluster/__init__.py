"""hypertools.cluster subpackage: the `cluster()` dispatcher and the
`Clusterer` base class it dispatches to (plus the `models`/`mixture_models`
registries kept for backward compatibility).

NOTE: the top-level package re-exports the FUNCTION as `hypertools.cluster`
(shadowing this subpackage as an attribute), but the class remains
importable from here, e.g. ``from hypertools.cluster import Clusterer``.
"""
from .common import Clusterer
from .cluster import cluster, models, mixture_models

__all__ = ['cluster', 'Clusterer', 'models', 'mixture_models']
