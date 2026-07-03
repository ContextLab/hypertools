"""hyp.manip dispatcher: resolve a manipulator spec and fit_transform it.

Wrapped by datawrangler's funnel so any input (array / DataFrame / list / text
/ polars) arrives as DataFrame(s); the resolved Manipulator (sklearn-compatible,
DataFrame-based) is applied directly rather than via the array-based
core.apply_model.
"""
import datawrangler as dw

from .common import Manipulator
from .normalize import Normalize
from .zscore import ZScore
from .smooth import Smooth
from .resample import Resample
from ..core.shared import unpack_model
from ..core.configurator import apply_defaults


MANIPULATORS = [Normalize, ZScore, Smooth, Resample]


@dw.decorate.funnel
def manip(data, model="ZScore", **kwargs):
    resolved = unpack_model(model, valid=MANIPULATORS, parent_class=Manipulator)
    if isinstance(resolved, type):
        resolved = resolved(**kwargs)
    elif isinstance(resolved, dict):
        cls = resolved["model"]
        resolved = cls(*resolved.get("args", []), **resolved.get("kwargs", {}))
    return resolved.fit_transform(data)
