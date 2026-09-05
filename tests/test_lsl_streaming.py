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

**Running these on a machine with real LSL hardware.** LSL resolution is
machine- and subnet-wide, so an attached amplifier's outlets match
`lsl_stream()`'s "any stream" fallback exactly as this module's own outlets
do. The tests below handle that themselves (see `_foreign_lsl_streams`), but
to reproduce a CLEAN machine deliberately -- e.g. to run the sole-outlet
branch on a laptop with an EEG cap plugged in -- give liblsl a private
session, which scopes resolution to processes sharing the same SessionID::

    printf '[lab]\\nSessionID = hypertools-tests\\n' > /tmp/lsl.cfg
    LSLAPICFG=/tmp/lsl.cfg .venv/bin/python -m pytest tests/test_lsl_streaming.py

Verified 2026-08-04: with that set, `pylsl.resolve_streams()` returns `[]` on
a machine where four STARSTIM-8 outlets were otherwise visible, and both
branches of `test_lsl_stream_resolves_any_stream` pass in their respective
environments.

**Machines that cannot create an outlet at all.** Binding liblsl's service
ports can fail outright (``All local ports were found occupied`` ->
``RuntimeError: could not create stream outlet``), which would otherwise
turn every outlet-backed test here into a setup error that reads like a
library regression. `_outlet_capability_reason` probes that once and
`_require_outlets` classifies it: a SKIP quoting the real liblsl error
locally, a hard FAILURE under `GITHUB_ACTIONS` or `HYPERTOOLS_REQUIRE_LSL=1`
(a provisioned runner is expected to pass the probe, so there its failing is
the news). Everything that needs no outlet -- validation, the ImportError
message, resolution failures, the ambiguity advice -- keeps running either
way, and `test_the_outlet_capability_probe_AGREES_with_reality` pins the
probe against a real outlet so it cannot silently skip a working machine.
The *classification* is pinned separately, by
`test_the_capability_POLICY_skips_here_and_FAILS_where_it_must`: only one of
its branches is reachable on any given machine, so that one drives both from
a real liblsl failure string with the environment as the only variable.
"""

import os
import subprocess
import sys
import textwrap
import threading
import time
import warnings

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


#: Set to any non-empty value to turn "this machine cannot provision an LSL
#: outlet" from a skip into a failure. Always on under GITHUB_ACTIONS: a CI
#: runner is provisioned, so there the probe failing IS the news.
REQUIRE_LSL_ENV = 'HYPERTOOLS_REQUIRE_LSL'

#: '' once outlets are known to work here, the real failure text if they do
#: not, None before the (single) probe has run.
_outlet_capability = None


def _outlet_capability_reason():
    """Probe ONCE, for real, whether this machine can create an LSL outlet.

    Creating a `pylsl.StreamOutlet` binds liblsl's service ports, and that
    can fail for reasons that have nothing to do with hypertools -- observed
    as ``All local ports were found occupied`` followed by ``RuntimeError:
    could not create stream outlet``, which turns every outlet-backed test
    in this module into a setup error at once. That is an unprovisionable
    environment, and it must be named as one: reported as a library
    regression it is noise, and swallowed silently it would hide a real
    liblsl breakage. The probe is what tells the two apart -- and it never
    stands in for the outlets themselves, which every test still creates for
    real.
    """
    global _outlet_capability
    if _outlet_capability is None:
        info = pylsl.StreamInfo(f'HypertoolsOutletProbe-{time.time_ns()}',
                                'EEG', 1, 100.0, 'float32', 'hypertools-probe')
        try:
            probe = pylsl.StreamOutlet(info)
        except Exception as err:
            _outlet_capability = f'{type(err).__name__}: {err}'
        else:
            _outlet_capability = ''
            del probe           # release the port again immediately
    return _outlet_capability


def _require_outlets():
    """Skip (locally) or fail (on CI) when `_outlet_capability_reason` says
    no outlet can be created here. Validation, import-error and
    resolution-failure tests do not call this -- they need no outlet, so
    they keep running and keep covering the library either way."""
    reason = _outlet_capability_reason()
    if not reason:
        return
    message = (f'this machine cannot create a pylsl.StreamOutlet, so the '
               f'real-outlet integration tests cannot be provisioned here '
               f'(the tests themselves are untouched): {reason}')
    if os.environ.get('GITHUB_ACTIONS') == 'true' \
            or os.environ.get(REQUIRE_LSL_ENV):
        pytest.fail(message)
    pytest.skip(message)


def _start_outlet(name, stream_type='EEG', n_channels=N_CHANNELS,
                  rate=100.0, push_interval=0.01):
    """Start a REAL pylsl.StreamOutlet on a background daemon thread,
    continuously pushing `_sample_for_index` samples until `stop` is set.
    Returns (thread, stop_event); caller must `stop.set(); thread.join(...)`
    when done."""
    _require_outlets()
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


#: Every outlet name this module creates. Anything else resolvable is a real
#: outlet belonging to the machine, not to the test run.
_TEST_STREAM_PREFIXES = ('HypertoolsTestStream-', 'HypertoolsStallTest-')


#: One wait time shared by the "any stream" probe and the `lsl_stream()`
#: call it decides between. They MUST match: `pylsl.resolve_streams` waits
#: the full time and returns everything it heard, but a shorter wait hears
#: less -- measured on a machine with five outlets, `wait_time=0.5` found
#: two of them and `wait_time=1.0` found all five. Probing for less time
#: than the call gets would therefore pick the sole-outlet branch on a
#: machine that has foreign hardware, and fail there for the wrong reason.
_ANY_STREAM_WAIT = 10.0


def _foreign_lsl_streams(wait_time=_ANY_STREAM_WAIT):
    """Names of resolvable LSL outlets this module did NOT create.

    `lsl_stream()` with neither `name=` nor `type=` means "any stream", and
    LSL resolution is MACHINE-WIDE (and subnet-wide): a real EEG amplifier,
    an accelerometer bridge, or a colleague's recorder all match. On such a
    machine "any" can legitimately resolve to an outlet that is idle for the
    whole timeout, or has a different channel count -- so an assertion about
    THIS module's outlet is not merely flaky there, it is wrong.

    Observed 2026-08-04 on a machine with a STARSTIM-8 attached: four
    outlets (Accelerometer/Markers/Quality/EEG, 3 and 8 channels), the
    accelerometer resolving first and delivering nothing, so
    `test_lsl_stream_resolves_any_stream` failed with `HypertoolsIOError`
    after 10s of silence.
    """
    return sorted(
        info.name() for info in pylsl.resolve_streams(wait_time=wait_time)
        if not info.name().startswith(_TEST_STREAM_PREFIXES))


@requires_pylsl
def test_lsl_stream_resolves_any_stream(outlet_stream):
    """With neither name= nor type=, lsl_stream falls back to
    pylsl.resolve_streams ("any stream").

    What that yields depends on the machine, so this checks the two
    outcomes separately rather than assuming a quiet one. NEITHER branch
    skips: the ambiguous branch is the only place the ambiguity warning can
    be exercised at all, because it needs an outlet this test cannot create.
    """
    foreign = _foreign_lsl_streams()
    if not foreign:
        # sole outlet: "any" is unambiguously ours, so it must deliver
        stream = hyp.io.lsl_stream(timeout=_ANY_STREAM_WAIT)
        assert is_stream(stream)
        sample = next(stream)
        assert len(sample) == N_CHANNELS
        assert all(isinstance(v, float) for v in sample)
        stream.close()
        return

    # Foreign outlets present. `lsl_stream()` must SAY so rather than
    # silently binding to one of them, and must name enough of them for the
    # user to pick with `name=`. Whether the chosen stream then delivers is
    # a fact about someone else's hardware, so it is not asserted here --
    # `test_lsl_stream_raises_when_source_stops_delivering` covers a silent
    # source against an outlet this module owns.
    #
    # `pytest.warns` is deliberately NOT used: it turns "no warning" into a
    # failure before this test can ask WHY there was none, and one of the
    # two reasons is somebody else's amplifier being unplugged mid-test.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            stream = hyp.io.lsl_stream(timeout=_ANY_STREAM_WAIT)
        except HypertoolsIOError:
            stream = None       # resolved a foreign outlet that is idle
    if stream is not None:
        stream.close()
    ambiguity = [w for w in caught if 'LSL streams match' in str(w.message)]

    if not ambiguity:
        still = _foreign_lsl_streams()
        assert not still, (
            f'foreign LSL outlets {still} are resolvable, and this test owns '
            f'one more, but lsl_stream() reported no ambiguity -- it bound '
            f'to a stream silently')
        pytest.skip(f'the foreign LSL outlets {foreign} stopped announcing '
                    f'themselves partway through this test, so there was no '
                    f'ambiguity left to detect')

    assert issubclass(ambiguity[0].category, RuntimeWarning)
    message = str(ambiguity[0].message)
    assert 'using the first one' in message
    assert 'Pass name= to select a specific stream' in message
    # The warning lists the first FIVE streams in resolver order; `foreign`
    # is every foreign stream in alphabetical order. So the right assertion
    # is that the two overlap -- not that any particular element of one
    # appears in the other, which is a coin flip once six outlets exist.
    assert [name for name in foreign if repr(name) in message], (
        f'the ambiguity warning named none of the foreign outlets '
        f'{foreign}: {message}')


@requires_pylsl
def test_lsl_stream_by_NAME_is_unaffected_by_foreign_outlets(outlet_stream):
    """The documented escape from the ambiguity above. This is what the
    warning tells the user to do, so it must work on exactly the machines
    that emit the warning -- and it must not warn, because `name=` resolves
    one outlet."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        stream = hyp.io.lsl_stream(name=outlet_stream, timeout=10.0)
        sample = next(stream)
        stream.close()
    assert len(sample) == N_CHANNELS
    assert all(isinstance(v, float) for v in sample)
    assert not [w for w in caught
                if 'LSL streams match' in str(w.message)], (
        'name= resolved exactly one outlet, so the ambiguity warning must '
        'not fire')


# ------------------------------------ the ambiguity warning's own advice


@requires_pylsl
def test_the_outlet_capability_probe_AGREES_with_reality():
    """A probe that wrongly answered "no outlets here" would silently skip
    every outlet-backed test in this module at once -- the one failure mode
    a skip must never have. So this creates a real outlet WITHOUT the guard
    and requires the probe's verdict to match what actually happened, in
    both directions. It therefore holds on a machine that cannot create
    outlets too, where it pins the probe as the reason the others skipped.
    """
    info = pylsl.StreamInfo(f'HypertoolsTestStream-agree-{time.time_ns()}',
                            'EEG', 1, 100.0, 'float32', 'hypertools-agree')
    try:
        outlet = pylsl.StreamOutlet(info)
    except Exception as err:
        assert _outlet_capability_reason(), (
            f'creating an outlet really failed ({type(err).__name__}: '
            f'{err}), but the probe reported this machine as capable')
    else:
        del outlet
        assert _outlet_capability_reason() == '', (
            f'an outlet was created successfully, but the probe reported '
            f'this machine as incapable: {_outlet_capability_reason()}')


@requires_pylsl
def test_the_capability_POLICY_skips_here_and_FAILS_where_it_must(monkeypatch):
    """`test_..._AGREES_with_reality` pins the probe's *verdict* against a
    real outlet -- but it cannot pin what is DONE with that verdict, because
    on any one machine only one branch is reachable (here, the capable one).
    The consequence of getting the other branch wrong is severe and silent:
    a skip where CI should have failed hides a real liblsl breakage behind
    "unsupported environment" forever.

    So this drives the policy directly, on a REAL liblsl failure string
    harvested from a REAL failed call, with the environment as the only
    thing varied. Nothing is faked: `_outlet_capability` is this module's
    own memo of an already-answered question, and it is restored afterwards
    so the next test re-probes for real."""
    try:
        pylsl.StreamInfo('HypertoolsPolicyProbe', 'EEG', -1, 100.0,
                         'float32', 'hypertools-policy')
    except Exception as err:
        real_reason = f'{type(err).__name__}: {err}'
    else:
        pytest.skip('this pylsl accepts a negative channel count, so no real '
                    'liblsl failure string is available to drive the policy '
                    'with -- refusing to invent one')

    this_module = sys.modules[__name__]
    monkeypatch.setattr(this_module, '_outlet_capability', real_reason)

    def classify(**env):
        """Run the policy under exactly `env` and NAME what it did.

        `pytest.raises(pytest.fail.Exception)` would be the obvious spelling
        and it is the wrong one: an inverted policy raises `Skipped` there,
        `raises` does not catch it, and this test would report itself as
        skipped -- silently, which is the precise failure it exists to
        prevent. Catching both outcomes and comparing names makes every
        wrong answer an assertion failure. (Verified by mutation: forcing
        the CI branch off turns this test red, not green and not skipped.)"""
        monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
        monkeypatch.delenv(REQUIRE_LSL_ENV, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        try:
            _require_outlets()
        except pytest.skip.Exception as skipped:
            return 'skip', str(skipped)
        except pytest.fail.Exception as failed:
            return 'fail', str(failed)
        return 'proceed', ''

    # (1) an unprovisionable laptop: skip, quoting what liblsl actually said
    verdict, message = classify()
    assert verdict == 'skip', (
        f'an unprovisionable local machine must skip, not {verdict}')
    assert real_reason in message, (
        'the skip must carry the real reason, or an unprovisionable machine '
        'is indistinguishable from a broken one')

    # (2) a provisioned CI runner: the same condition is the news itself
    for env in ({REQUIRE_LSL_ENV: '1'}, {'GITHUB_ACTIONS': 'true'}):
        verdict, message = classify(**env)
        assert verdict == 'fail', (
            f'{env} must FAIL rather than {verdict} -- a provisioned runner '
            f'that cannot make an outlet is the news, not an excuse')
        assert real_reason in message, f'{env} must say why it failed'

    # (3) a capable machine proceeds -- under every one of those settings
    monkeypatch.setattr(this_module, '_outlet_capability', '')
    for env in ({}, {REQUIRE_LSL_ENV: '1'}, {'GITHUB_ACTIONS': 'true'}):
        verdict, _ = classify(**env)
        assert verdict == 'proceed', (
            f'a capable machine must proceed to the real outlet tests, but '
            f'with {env} the policy said {verdict}')


@requires_pylsl
def test_the_ambiguity_ADVICE_is_executable_on_the_path_that_gives_it():
    """The multi-match warning ends by telling the user what to do next, so
    that advice has to work on the path that emitted it. The two paths call
    different pylsl functions: `resolve_byprop` accepts `minimum=`,
    `resolve_streams` does not -- so a single shared caveat advising
    `minimum=2` was advising a `TypeError` to every "any stream" user.

    Pinned against the REAL signatures, so a future pylsl that adds or drops
    the argument fails here rather than in somebody's warning text."""
    import inspect

    from hypertools.io.lsl import _ambiguity_caveat

    assert 'minimum' in inspect.signature(pylsl.resolve_byprop).parameters
    assert 'minimum' not in inspect.signature(
        pylsl.resolve_streams).parameters

    assert 'minimum=2' in _ambiguity_caveat(any_stream=False)
    assert 'minimum=2' not in _ambiguity_caveat(any_stream=True)
    # the any-stream path's own remedy is a longer wait, and it says so
    assert 'timeout=' in _ambiguity_caveat(any_stream=True)

    # reaching a private helper from its own module's tests is fine; letting
    # it become part of the hypertools.io surface is not
    assert not hasattr(hyp.io, '_ambiguity_caveat'), (
        'the caveat builder is internal to hypertools.io.lsl and must not be '
        're-exported from hypertools.io')


@requires_pylsl
def test_minimum_really_is_forwarded_to_the_byprop_resolver(outlet_stream):
    """The other half of the advice: `minimum=2` must actually reach
    `pylsl.resolve_byprop` and actually force the ambiguity detection the
    warning promises. Two REAL outlets of the same type are live here, so a
    resolver honouring `minimum=2` finds both and `lsl_stream()` warns --
    where the default `minimum=1` may return the first and warn about
    nothing."""
    second = f'HypertoolsTestStream-second-{time.time_ns()}'
    thread, stop = _start_outlet(second, stream_type='EEG')
    try:
        with pytest.warns(RuntimeWarning, match='LSL streams match'):
            stream = hyp.io.lsl_stream(type='EEG', timeout=10.0, minimum=2)
        stream.close()
    finally:
        stop.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive()


@requires_pylsl
def test_minimum_on_the_ANY_STREAM_path_is_a_pylsl_TypeError():
    """And the documented consequence of the advice the caveat no longer
    gives: `resolve_kwargs` are forwarded verbatim, so `minimum=` on the
    "any stream" path is `pylsl.resolve_streams`' own `TypeError`. Needs no
    outlet -- the call fails before any stream is resolved."""
    with pytest.raises(TypeError, match='minimum'):
        hyp.io.lsl_stream(timeout=0.5, minimum=2)


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
    # the silence abort is terminal: the inlet is released with it
    assert stream.closed


# ----------------------------------------------- deterministic teardown


def _inlet_ref(stream):
    """A weakref to the pylsl.StreamInlet behind `stream`. It dies when the
    stream has dropped the inlet -- which it does only after destroying
    it (`lsl_destroy_inlet`, the ONLY liblsl call that fully shuts an
    inlet down; see the `hypertools.io.lsl` module docstring)."""
    import weakref
    return weakref.ref(stream._inlet)


@requires_pylsl
def test_close_destroys_the_inlet_and_is_idempotent(outlet_stream):
    stream = hyp.io.lsl_stream(name=outlet_stream, timeout=5.0)
    assert len(next(stream)) == N_CHANNELS
    assert not stream.closed
    ref = _inlet_ref(stream)
    assert ref() is not None

    stream.close()

    assert stream.closed
    assert ref() is None, 'pylsl.StreamInlet survived close(): the inlet ' \
                          'was only close_stream()ed, not destroyed'
    # a closed stream is exhausted, exactly like a closed generator ...
    with pytest.raises(StopIteration):
        next(stream)
    # ... and closing it again is a no-op
    stream.close()
    assert stream.closed


@requires_pylsl
def test_context_manager_closes_the_inlet(outlet_stream):
    with hyp.io.lsl_stream(name=outlet_stream, timeout=5.0) as stream:
        assert is_stream(stream)
        assert len(next(stream)) == N_CHANNELS
        ref = _inlet_ref(stream)
    assert stream.closed
    assert ref() is None


@requires_pylsl
def test_dropping_the_last_reference_destroys_the_inlet(outlet_stream):
    import gc
    stream = hyp.io.lsl_stream(name=outlet_stream, timeout=5.0)
    assert len(next(stream)) == N_CHANNELS
    ref = _inlet_ref(stream)
    del stream
    gc.collect()
    assert ref() is None, 'a garbage-collected stream left its inlet alive'


@requires_pylsl
def test_plot_stream_leaves_the_stream_open_for_reuse(outlet_stream):
    # hyp.plot stops PULLING at stream_max; it does not own the stream, so
    # the caller decides when the inlet goes (the tutorial closes it
    # explicitly, before tearing down its outlet)
    stream = hyp.io.lsl_stream(name=outlet_stream, timeout=5.0)
    with pytest.warns(RuntimeWarning, match='outside the display box'):
        hyp.plot(stream, show=False, stream_init=20, stream_chunk=10,
                 stream_max=40)
    assert not stream.closed
    assert len(next(stream)) == N_CHANNELS
    stream.close()
    assert stream.closed


_TEARDOWN_SCRIPT = """
    import sys, threading, time
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    import pylsl
    import hypertools as hyp

    MODE = sys.argv[1]
    NAME = 'HypertoolsTeardownTest-%d' % time.time_ns()
    info = pylsl.StreamInfo(NAME, 'EEG', 6, 100.0, 'float32',
                            'hypertools-test-' + NAME)
    outlet = pylsl.StreamOutlet(info)
    stop = threading.Event()

    def _push():
        i = 0
        while not stop.is_set():
            outlet.push_sample([float(np.sin(0.02 * i * (c + 1)))
                                for c in range(6)])
            i += 1
            time.sleep(0.01)

    thread = threading.Thread(target=_push, daemon=True)
    thread.start()

    stream = hyp.io.lsl_stream(name=NAME, timeout=5.0)
    fig = hyp.plot(stream, stream_init=40, stream_chunk=20, stream_max=120,
                   show=False)
    assert fig.stream_info['n_samples'] >= 120, fig.stream_info

    if MODE == 'closed_first':
        stream.close()          # the tutorial's clean-up cell
    elif MODE == 'never_closed':
        pass                    # a script/notebook that just exits
    else:
        raise AssertionError(MODE)
    stop.set()
    thread.join(timeout=5.0)
    print('SUBPROCESS_OK')
"""


@requires_pylsl
@pytest.mark.parametrize('mode', ['closed_first', 'never_closed'])
def test_teardown_leaves_no_liblsl_error(mode):
    """Streaming into hyp.plot and exiting -- with or without an explicit
    `close()` -- must not leave a pylsl.StreamInlet for liblsl to complain
    about. Before the fix, exiting with the stream still open logged
    ``data_receiver.cpp:344 ERR| Stream transmission broke off (Input
    stream error.); re-connecting...`` to stderr (measured 3/3 runs on
    2026-09-04): the inlet's data thread outlived the outlet, and
    `close_stream()` alone does not silence it (liblsl only suppresses the
    message once the inlet is DESTROYED). Real outlet, real inlet, real
    interpreter exit, in a subprocess so stderr is the process's own."""
    _require_outlets()
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(_TEARDOWN_SCRIPT), mode],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout}\nstderr={result.stderr}")
    assert 'SUBPROCESS_OK' in result.stdout
    assert 'broke off' not in result.stderr, result.stderr
    assert 'ERR|' not in result.stderr, result.stderr


_TUTORIAL_UNDER_LOAD_SCRIPT = """
    import sys, threading, time
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    import pylsl
    import hypertools as hyp

    SAVE_PATH, CYCLES = sys.argv[1], int(sys.argv[2])
    NAME = 'HypertoolsTutorialLoadTest-%d' % time.time_ns()
    info = pylsl.StreamInfo(NAME, 'EEG', 6, 100.0, 'float32',
                            'hypertools-test-' + NAME)
    outlet = pylsl.StreamOutlet(info)
    stop = threading.Event()

    def _push():                       # the tutorial's synthetic outlet
        i = 0
        while not stop.is_set():
            outlet.push_sample([float(np.sin(2 * np.pi * (0.5 + 0.1 * c)
                                             * i * 0.02)) for c in range(6)])
            i += 1
            time.sleep(0.01)

    threading.Thread(target=_push, daemon=True).start()

    # a busy interpreter: pure-Python threads that hold the GIL while the
    # stream is being closed (what a live kernel's own threads do)
    hogging = threading.Event()

    def _hog():
        x = 0
        while not stop.is_set():
            if hogging.is_set():
                x += 1
            else:
                time.sleep(0.01)

    for _ in range(8):
        threading.Thread(target=_hog, daemon=True).start()

    for k in range(CYCLES):
        stream = hyp.io.lsl_stream(name=NAME, timeout=5.0)
        fig = hyp.plot(stream, stream_init=200, stream_chunk=20,
                       stream_max=600, save_path=SAVE_PATH, frame_rate=5,
                       show=False)
        assert fig.stream_info['n_samples'] >= 600, fig.stream_info
        hogging.set()
        sys.setswitchinterval(0.1)
        time.sleep(0.3)
        stream.close()                 # the tutorial's clean-up cell ...
        time.sleep(0.3)
        hogging.clear()
        sys.setswitchinterval(0.005)
        assert stream.closed
    stop.set()                         # ... then the outlet goes away
    time.sleep(0.3)
    print('SUBPROCESS_OK')
"""


@requires_pylsl
def test_tutorial_close_under_load_leaves_no_liblsl_error(tmp_path):
    """The LSL tutorial's own sequence -- outlet thread, `hyp.plot(stream,
    stream_init=200, stream_chunk=20, stream_max=600, save_path=.mp4,
    frame_rate=5)`, `stream.close()`, stop the outlet -- must not log
    liblsl's ``ERR| Stream transmission broke off`` at `close()`.

    `close()` used to `close_stream()` the inlet and only then destroy
    it. liblsl 1.17.7 (data_receiver.cpp) logs that line from the
    receiver thread whenever a cancelled read fails while the
    connection's shutdown flag is still clear -- and `close_stream()`
    never sets that flag, only a destroy does. So the two calls raced
    the receiver thread, and a busy interpreter (a kernel's threads
    holding the GIL between the two ctypes calls) made it lose: 4 of 6
    executions of docs/tutorials/lsl_streaming.ipynb logged the line
    into the clean-up cell. The GIL hogs below reproduce that on demand:
    3 cycles under load hit 9 of 12 cycles (4 of 4 processes) with the
    old release path, 0 of 12 with a direct `lsl_destroy_inlet` (which
    sets the flag BEFORE cancelling, and joins the receiver thread).
    Real outlet, real inlet, real subprocess stderr; no mocks."""
    _require_outlets()
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(_TUTORIAL_UNDER_LOAD_SCRIPT),
         str(tmp_path / 'lsl_streaming.mp4'), '3'],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout}\nstderr={result.stderr}")
    assert 'SUBPROCESS_OK' in result.stdout
    assert 'broke off' not in result.stderr, result.stderr
    assert 'ERR|' not in result.stderr, result.stderr
    assert (tmp_path / 'lsl_streaming.mp4').stat().st_size > 0
