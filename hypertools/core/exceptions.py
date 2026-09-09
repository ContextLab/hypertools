class HypertoolsError(Exception):
    """Base class for all hypertools-specific exceptions."""
    pass


class _InsufficientHistoryError(ValueError):
    """A forecast needs more observations (including after time resampling).

    Internal signal for animations to wait for more revealed history; other
    fitting errors must still propagate. Public callers can catch ValueError.
    """


class HypertoolsBackendError(HypertoolsError):
    """Raised when a plotting backend (matplotlib/plotly) cannot satisfy a request."""
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class HypertoolsIOError(HypertoolsError, OSError):
    """Raised for hypertools-specific I/O failures (e.g. loading/streaming data)."""
    def __init__(self, message):
        super().__init__(message)
        self.message = message
