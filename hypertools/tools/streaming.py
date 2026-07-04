# Moved to hypertools.io.streaming (HyperTools 2.0). Shim preserves the old path.
# NOTE: plot_stream (the animated renderer) rides along here for now; its move to
# hypertools.plot is deferred to Plan 6.
from ..io.streaming import *  # noqa: F401,F403
from ..io.streaming import is_stream, row_to_vector, _fit_stream_models, plot_stream  # noqa: F401
