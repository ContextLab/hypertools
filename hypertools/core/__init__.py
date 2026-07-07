"""HyperTools 1.0 core: model dispatch, configuration, and shared utilities."""

from .exceptions import (
    HypertoolsError,
    HypertoolsBackendError,
    HypertoolsIOError,
)
from .shared import RobustDict, unpack_model, get
from .configurator import get_default_options, apply_defaults
from .model import apply_model, supported_models
from .pipeline import Pipeline, build_pipeline, CANONICAL_ORDER
