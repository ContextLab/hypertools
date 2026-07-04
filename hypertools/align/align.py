"""hyp.align dispatcher: resolve an aligner spec and fit_transform it.

Wrapped by datawrangler's funnel so any input (array / DataFrame / list / text /
polars) arrives as DataFrame(s); the resolved Aligner (list-based, sklearn-
compatible) is applied directly. NOT routed through core.apply_model, whose
stack-and-fit-once recipe is wrong for aligning a *list* to a shared template.
"""
import datawrangler as dw

from .common import Aligner
from .hyperalign import HyperAlign
from .procrustes import Procrustes
from .srm import SharedResponseModel, DeterministicSharedResponseModel, RobustSharedResponseModel
from .null import NullAlign
from ..core.shared import unpack_model


ALIGNERS = [HyperAlign, SharedResponseModel, DeterministicSharedResponseModel,
            RobustSharedResponseModel, Procrustes, NullAlign]


@dw.decorate.funnel
def align(data, model='HyperAlign', **kwargs):
    resolved = unpack_model(model, valid=ALIGNERS, parent_class=Aligner)
    if isinstance(resolved, type):
        resolved = resolved(**kwargs)
    elif isinstance(resolved, dict):
        cls = resolved['model']
        resolved = cls(*resolved.get('args', []), **resolved.get('kwargs', {}))
    return resolved.fit_transform(data)
