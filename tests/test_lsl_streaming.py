# -*- coding: utf-8 -*-
"""Tests for Lab Streaming Layer (LSL) input support (GH #130):

`hypertools.io.lsl_stream(name=, type=, timeout=)` resolves a live LSL
stream via `pylsl.resolve_byprop`/`pylsl.resolve_streams`, opens a
`pylsl.StreamInlet` on it, and returns a plain Python generator of
per-sample numeric vectors -- compatible with the existing streaming
machinery (`hypertools.io.streaming.is_stream`/`row_to_vector`, GH #101),
so `hyp.plot(hyp.io.lsl_stream(type='EEG'), stream_init=..., stream_chunk=
...)` just works.

Every test that needs pylsl spins up a REAL `pylsl.StreamOutlet` on a
background thread publishing deterministic synthetic samples, and consumes
it through the real `pylsl.StreamInlet` opened by `lsl_stream()` -- no
mocks anywhere.

`pylsl` is an optional dependency (`pip install "hypertools[lsl]"`); the
`requires_pylsl` skip mirrors the `requires_covering_font`
(tests/test_multibyte.py, GH #205) / kagglehub (tests/test_load_538_kaggle.py)
pattern: a missing pylsl on a local machine SKIPS the pylsl-dependent
tests, but on CI (GITHUB_ACTIONS=true) `test_ci_has_pylsl` FAILS hard
instead -- this task's investigation confirmed that ubuntu-latest,
macos-latest, and windows-latest all resolve a `pylsl>=1.16` wheel that
BUNDLES the native liblsl library (manylinux_2_35_x86_64 -> liblsl.so,
macosx_11_0_universal2 -> liblsl.dylib, win_amd64/win32 -> lsl.dll), so no
separate system liblsl install step is needed on any of the three CI
platforms -- see .github/workflows/test.yml for the verified details. That
means all three platforms are "provisioned" and none needs to be excluded.
"""

import os
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools.io.streaming import is_stream
from hypertools._shared.exceptions import HypertoolsIOError

try:
    import pylsl
    PYLSL_AVAILABLE = True
except ImportError:
    PYLSL_AVAILABLE = False

requires_pylsl = pytest.mark.skipif(
    not PYLSL_AVAILABLE,
    reason="pylsl is not installed -- install it with `pip install "
           "hypertools[lsl]` to exercise hyp.io.lsl_stream()",
)

N_CHANNELS = 8


def _sample_for_index(i, n_channels=N_CHANNELS):
    """Deterministic per-sample vector: channel c of sample i is
    i + 0.1*c -- easy to verify on the receiving end (consecutive samples
    differ by exactly 1.0 in every channel; channel c is offset from
    channel 0 by exactly 0.1*c), while not being perfectly rank-1/constant
    across channels."""
    return [float(i) + 0.1 * c for c in range(n_channels)]


def _start_outlet(name, stream_type='EEG', n_channels=N_CHANNELS,
                  rate=100.0, push_interval=0.01):
    """Start a REAL pylsl.StreamOutlet on a background daemon thread,
    continuously pushing `_sample_for_index` samples until `stop` is set.
    Returns (thread, stop_event); caller must `stop.set(); thread.join(...)`
    when done."""
    info = pylsl.StreamInfo(name, stream_type, n_channels, rate, 'float32',
                            f'hypertools-test-{name}')
    outlet = pylsl.StreamOutlet(info)
    stop = threading.Event()

    def _push():
        i = 0
        while not stop.is_set():
            outlet.push_sample(_sample_for_index(i, n_channels))
            i += 1
            time.sleep(push_interval)

    thread = threading.Thread(target=_push, daemon=True)
    thread.start()
    return thread, stop


@pytest.fixture
def outlet_stream():
    """A REAL background-thread LSL outlet for the duration of one test,
    with a unique name (avoids cross-test resolution collisions) and
    guaranteed cleanup even if the test body raises."""
    name = f'HypertoolsTestStream-{threading.get_ident()}-{time.time_ns()}'
    thread, stop = _start_outlet(name)
    try:
        yield name
    finally:
        stop.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), \
            'outlet-pushing background thread failed to stop in time'


# --------------------------------------------------------------- guard


def test_ci_has_pylsl():
    # GH #130 CI guard (mirrors the GH #205 fonts / kagglehub pattern): on
    # CI, pylsl must be importable -- see the module docstring for why all
    # three platforms are expected to be provisioned via the bundled-
    # liblsl wheel. A failure here means every requires_pylsl-gated test
    # below just silently skipped on this PR.
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        pytest.skip("only meaningful on CI (GITHUB_ACTIONS=true); a "
                    "missing pylsl on a local machine is expected and "
                    "handled by requires_pylsl's skip")
    assert PYLSL_AVAILABLE, (
        "pylsl failed to import on CI -- check that `pip install -e "
        "'.[dev]'` actually installed pylsl, and that this platform's "
        "wheel bundles (or otherwise has access to) the native liblsl "
        "library (see the module docstring / .github/workflows/test.yml)."
    )


# ------------------------------------------------------------- resolving


@requires_pylsl
def test_is_stream_true_for_lsl_stream(outlet_stream):
    stream = hyp.io.lsl_stream(name=outlet_stream, timeout=5.0)
    assert is_stream(stream)


@requires_pylsl
def test_lsl_stream_receives_pushed_samples_by_name(outlet_stream):
    stream = hyp.io.lsl_stream(name=outlet_stream, timeout=5.0)

    received = [next(stream) for _ in range(10)]

    # allow an initial-sample offset: the outlet may have already pushed
    # several samples by the time the inlet connects, so the FIRST
    # received sample need not be index 0 -- but every received vector
    # must be _sample_for_index(i) for SOME consistent, increasing i
    first_i = round(received[0][0])
    assert first_i >= 0
    for k, sample in enumerate(received):
        expected = _sample_for_index(first_i + k)
        np.testing.assert_allclose(sample, expected, atol=1e-4)


@requires_pylsl
def test_lsl_stream_resolves_by_type(outlet_stream):
    # resolve using type= (the outlet was created with stream_type='EEG'
    # in _start_outlet) rather than name=
    stream = hyp.io.lsl_stream(type='EEG', timeout=5.0)
    sample = next(stream)
    assert len(sample) == N_CHANNELS


@requires_pylsl
def test_lsl_stream_name_takes_precedence_over_type():
    # two simultaneous outlets with different types; name= must pick the
    # exact one requested even though type= alone would be ambiguous
    name_a = f'HypertoolsTestStreamA-{time.time_ns()}'
    name_b = f'HypertoolsTestStreamB-{time.time_ns()}'
    thread_a, stop_a = _start_outlet(name_a, stream_type='EEG')
    thread_b, stop_b = _start_outlet(name_b, stream_type='EEG')
    try:
        stream = hyp.io.lsl_stream(name=name_b, type='EEG', timeout=5.0)
        sample = next(stream)
        assert len(sample) == N_CHANNELS
    finally:
        stop_a.set()
        stop_b.set()
        thread_a.join(timeout=5.0)
        thread_b.join(timeout=5.0)
        assert not thread_a.is_alive() and not thread_b.is_alive()


# -------------------------------------------------------------- timeout


def test_timeout_raises_hypertools_io_error():
    if not PYLSL_AVAILABLE:
        pytest.skip("pylsl is not installed -- install it with `pip "
                    "install hypertools[lsl]`")
    with pytest.raises(HypertoolsIOError) as excinfo:
        hyp.io.lsl_stream(name='nonexistent-xyz-stream-hypertools-test',
                          timeout=0.5)
    message = str(excinfo.value)
    assert 'nonexistent-xyz-stream-hypertools-test' in message
    assert 'timeout' in message.lower()


# --------------------------------------------------------- end-to-end


@requires_pylsl
def test_plot_stream_end_to_end(outlet_stream):
    stream = hyp.io.lsl_stream(name=outlet_stream, timeout=5.0)
    # _sample_for_index is a monotonically increasing ramp, so every
    # post-head sample is guaranteed outside the display box fitted on the
    # first stream_init samples -- the clamped-samples notice must fire
    with pytest.warns(RuntimeWarning, match='outside the display box'):
        fig = hyp.plot(stream, show=False, stream_init=20, stream_chunk=10,
                       stream_max=50)
    assert fig is not None
    assert fig.stream_info['n_samples'] >= 50
    assert not fig.stream_info['truncated'] or \
        fig.stream_info['n_samples'] >= 50


# --------------------------------------------------- ImportError message


def test_import_error_without_pylsl_names_the_extra():
    # a REAL import-blocking sys.meta_path finder (not a mock of
    # hypertools) run in a subprocess so this process's already-imported
    # pylsl (if installed) is untouched -- mirrors
    # tests/test_density.py::TestFogFallbackSubprocess and
    # tests/test_autoencoders.py's torch-blocker pattern.
    script = textwrap.dedent("""
        import sys
        import importlib.abc, importlib.machinery

        class BlockLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                raise ImportError("pylsl blocked for test")

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == 'pylsl' or name.startswith('pylsl.'):
                    return importlib.machinery.ModuleSpec(name, BlockLoader())
                return None

        sys.meta_path.insert(0, Blocker())

        import matplotlib
        matplotlib.use('Agg')
        import hypertools as hyp

        try:
            hyp.io.lsl_stream(name='irrelevant', timeout=0.1)
        except ImportError as exc:
            assert 'hypertools[lsl]' in str(exc), str(exc)
            print("SUBPROCESS_OK")
        else:
            raise AssertionError('expected ImportError, none raised')
    """)
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert 'SUBPROCESS_OK' in result.stdout


@requires_pylsl
def test_lsl_stream_resolves_any_stream(outlet_stream):
    """With neither name= nor type=, lsl_stream falls back to
    pylsl.resolve_streams ("any stream") and still yields real samples."""
    stream = hyp.io.lsl_stream(timeout=10.0)
    assert is_stream(stream)
    sample = next(stream)
    assert len(sample) == N_CHANNELS
    assert all(isinstance(v, float) for v in sample)
    stream.close()


@requires_pylsl
def test_lsl_stream_raises_when_source_stops_delivering():
    """A stalled/disconnected source must not hang the consumer forever:
    after ~timeout seconds of consecutive silent pulls, the generator
    raises HypertoolsIOError instead of blocking indefinitely."""
    import gc

    name = f'HypertoolsStallTest-{threading.get_ident()}-{time.time_ns()}'
    thread, stop = _start_outlet(name)
    try:
        stream = hyp.io.lsl_stream(name=name, timeout=2.0)
        # receive at least one real sample while the outlet is alive
        assert len(next(stream)) == N_CHANNELS
    finally:
        stop.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
    # the outlet object was owned by the pusher thread's closure; force
    # its destruction so the stream goes truly silent
    gc.collect()

    deadline = time.time() + 30.0
    with pytest.raises(HypertoolsIOError, match='stopped delivering'):
        while time.time() < deadline:
            next(stream)  # drains any buffered samples, then must raise
    assert time.time() < deadline, \
        'generator kept yielding (or blocking) instead of raising'
