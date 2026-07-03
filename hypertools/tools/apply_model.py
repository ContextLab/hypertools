# Relocated to hypertools.core.model (HyperTools 2.0). Shim keeps the old import
# path (hypertools.tools.apply_model) working for existing callers and tests.
from ..core.model import (  # noqa: F401
    apply_model,
    supported_models,
    _build_registry,
)
