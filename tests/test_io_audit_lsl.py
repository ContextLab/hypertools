# -*- coding: utf-8 -*-
"""Regression tests for the 2026-07 release audit findings on
hyp.io.lsl_stream (unit F22-io-streaming-lsl, hypertools/io/lsl.py side).
Every pylsl-dependent test spins up a REAL pylsl.StreamOutlet -- no mocks
(mirrors tests/test_lsl_streaming.py)."""

import threading
import time

import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools._shared.exceptions import HypertoolsIOError

try:
    import pylsl
    PYLSL_AVAILABLE = True
except ImportError:
    PYLSL_AVAILABLE = False

requires_pylsl = pytest.mark.skipif(
    not PYLSL_AVAILABLE,
    reason='pylsl is not installed -- install it with '
           '`pip install "hypertools[lsl]"`')


# --------------------------------------------------------------- F22-009
# string-typed LSL streams (e.g. Markers) must be rejected with a clear
# error at resolve time instead of yielding strings that crash hyp.plot
# with a raw float-conversion ValueError.

@requires_pylsl
def test_string_channel_stream_rejected_at_resolve():
    name = f'HypAuditMarkers-{threading.get_ident()}-{time.time_ns()}'
    info = pylsl.StreamInfo(name, 'Markers', 1, 0, 'string',
                            f'hypertools-test-{name}')
    outlet = pylsl.StreamOutlet(info)   # real outlet, alive during resolve
    try:
        with pytest.raises(HypertoolsIOError, match='string'):
            hyp.io.lsl_stream(name=name, timeout=5.0)
    finally:
        del outlet


# --------------------------------------------------------------- F22-012
# multiple matching streams: the silently-picked first match must at least
# be announced with a warning naming the choice.

@requires_pylsl
def test_multiple_matching_streams_warn():
    stamp = time.time_ns()
    stream_type = f'HypAuditMulti{stamp}'
    infos = [pylsl.StreamInfo(f'HypAuditMultiA-{stamp}', stream_type, 4,
                              100.0, 'float32', f'hyp-test-a-{stamp}'),
             pylsl.StreamInfo(f'HypAuditMultiB-{stamp}', stream_type, 4,
                              100.0, 'float32', f'hyp-test-b-{stamp}')]
    outlets = [pylsl.StreamOutlet(i) for i in infos]
    try:
        with pytest.warns(RuntimeWarning, match='using the first'):
            stream = hyp.io.lsl_stream(type=stream_type, timeout=5.0,
                                       minimum=2)
        stream.close()
    finally:
        del outlets


@requires_pylsl
def test_single_matching_stream_does_not_warn():
    import warnings as _warnings
    name = f'HypAuditSingle-{threading.get_ident()}-{time.time_ns()}'
    info = pylsl.StreamInfo(name, 'EEG', 4, 100.0, 'float32',
                            f'hyp-test-{name}')
    outlet = pylsl.StreamOutlet(info)
    try:
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            stream = hyp.io.lsl_stream(name=name, timeout=5.0)
        stream.close()
        assert not [w for w in caught if 'using the first' in str(w.message)]
    finally:
        del outlet


# --------------------------------------------------------------- F22-011
# the docstring must document the mid-stream silence abort and must not
# suggest string-typed 'Markers' streams.

def test_lsl_docstring_documents_midstream_abort_and_numeric_only():
    doc = hyp.io.lsl_stream.__doc__
    assert 'silence' in doc or 'stops delivering' in doc.lower() or \
        'mid-stream' in doc
    assert "'Markers'" not in doc
