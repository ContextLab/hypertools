"""HyperTools 2.0 core: model dispatch, configuration, and shared utilities."""

from .exceptions import (
    HypertoolsError,
    HypertoolsBackendError,
    HypertoolsIOError,
)
from .shared import RobustDict, unpack_model
from .configurator import get_default_options, apply_defaults
