"""hypertools.align subpackage: the `align()` dispatcher plus the Aligner
classes it dispatches to.

NOTE: the top-level package re-exports the FUNCTION as `hypertools.align`
(shadowing this subpackage as an attribute), but the classes remain
importable from here, e.g. ``from hypertools.align import HyperAlign``.
"""
from .common import Aligner, pad, trim_and_pad
from .procrustes import procrustes, Procrustes
from .hyperalign import HyperAlign
from .srm import SharedResponseModel, DeterministicSharedResponseModel, RobustSharedResponseModel
from .null import NullAlign
from .align import align

__all__ = ['align', 'Aligner', 'HyperAlign', 'Procrustes', 'NullAlign',
           'SharedResponseModel', 'DeterministicSharedResponseModel',
           'RobustSharedResponseModel', 'procrustes', 'pad', 'trim_and_pad']
