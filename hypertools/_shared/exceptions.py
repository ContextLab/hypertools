# Moved to hypertools.core.exceptions (HyperTools 2.0 refactor). Kept as a shim
# so existing imports (hypertools._shared.exceptions) keep working.
from ..core.exceptions import (  # noqa: F401
    HypertoolsError,
    HypertoolsBackendError,
    HypertoolsIOError,
)
