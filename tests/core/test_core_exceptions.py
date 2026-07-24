def test_exceptions_importable_from_core():
    from hypertools.core.exceptions import (
        HypertoolsError, HypertoolsBackendError, HypertoolsIOError,
    )
    assert issubclass(HypertoolsBackendError, HypertoolsError)
    assert issubclass(HypertoolsIOError, HypertoolsError)
    assert issubclass(HypertoolsIOError, OSError)


def test_shared_exceptions_are_the_same_objects():
    # the _shared shim must re-export the SAME class objects, not copies
    from hypertools.core import exceptions as core_exc
    from hypertools._shared import exceptions as shared_exc
    assert core_exc.HypertoolsError is shared_exc.HypertoolsError
    assert core_exc.HypertoolsIOError is shared_exc.HypertoolsIOError
