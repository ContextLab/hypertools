# -*- coding: utf-8 -*-
"""Regression tests for backend-management state safety (QC audit 2026-07,
unit F06-plot-backends).

Covers:
- F06-001: $HYPERTOOLS_BACKEND set to any built-in candidate backend name
  (e.g. TkAgg/tkagg/MacOSX/QtAgg) crashed `import hypertools` with an
  UnboundLocalError masking a ValueError. Tested with REAL subprocesses.
- F06-002: a failed backend switch inside the set_interactive_backend
  context (including the one manage_backend wraps around hyp.plot) leaked
  IN_SET_CONTEXT=True, kept the bad HYPERTOOLS_BACKEND value, and cleared
  BACKEND_WARNING for the rest of the session.
- F06-003: set_interactive_backend accepted any garbage value ('bogus',
  None, 42, lists, bytes) silently; failures surfaced only at plot time.
- F06-004: nested set_interactive_backend contexts: the inner __exit__
  reset IN_SET_CONTEXT to False while the outer context was still active.
- F06-007: no public way to reset the render preference to auto-detection
  ('auto'/None now restore it).
- F06-008: set_interactive_backend('plotly') was silently ignored when
  plotly is not installed (now warns; tested via a real subprocess whose
  import machinery blocks plotly).

Real calls only, no mocks; headless (Agg).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import hypertools as hyp
from hypertools._shared.exceptions import HypertoolsBackendError
from hypertools.plot import backend as B


REPO_ROOT = Path(__file__).resolve().parents[1]

# a syntactically valid matplotlib backend spec ("module://..." paths are
# legal backend names) that is guaranteed to fail to import on any machine:
# lets us exercise the switch-failure path deterministically
UNSWITCHABLE_BACKEND = "module://hypertools_nonexistent_backend_xyz"


def _traj():
    return np.random.default_rng(0).standard_normal((30, 4))


def _run_subprocess(code, extra_env=None):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.pop("HYPERTOOLS_BACKEND", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=180,
    )


@pytest.fixture(autouse=True)
def _restore_backend_state():
    """Snapshot/restore backend-module globals so tests can't pollute
    each other (these tests intentionally poke at failure paths)."""
    saved = {
        name: getattr(B, name)
        for name in (
            "HYPERTOOLS_BACKEND",
            "BACKEND_WARNING",
            "IN_SET_CONTEXT",
            "PREFERRED_RENDER_BACKEND",
        )
    }
    saved_depth = getattr(B, "_SET_CONTEXT_DEPTH", None)
    yield
    for name, value in saved.items():
        setattr(B, name, value)
    if saved_depth is not None:
        B._SET_CONTEXT_DEPTH = saved_depth


# ---------------------------------------------------------------------------
# F06-001: $HYPERTOOLS_BACKEND env var must never crash import
# ---------------------------------------------------------------------------

_IMPORT_PROBE = """
import json, warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    import hypertools
from hypertools.plot import backend as B
print(json.dumps({
    "backend": str(B.HYPERTOOLS_BACKEND),
    "warnings": [str(x.message) for x in w],
}))
"""


@pytest.mark.parametrize("env_value", ["TkAgg", "tkagg", "MacOSX", "QtAgg"])
def test_env_var_in_candidate_list_does_not_crash_import(env_value):
    # regression: these exact values crashed `import hypertools` with
    # "ValueError: tuple.index(x): x not in tuple" masked by
    # "UnboundLocalError: ... 'working_backend' ..."
    result = _run_subprocess(_IMPORT_PROBE, {"HYPERTOOLS_BACKEND": env_value})
    assert result.returncode == 0, (
        f"import hypertools crashed with HYPERTOOLS_BACKEND={env_value!r}:\n"
        f"{result.stderr}"
    )
    assert "UnboundLocalError" not in result.stderr
    assert "Traceback" not in result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    # the env backend must either be honored or produce the documented
    # fallback warning naming the requested value
    if payload["backend"].lower() != env_value.lower():
        assert any(
            env_value in w and "backend" in w for w in payload["warnings"]
        ), (
            f"HYPERTOOLS_BACKEND={env_value!r} neither honored "
            f"(got {payload['backend']!r}) nor warned about: "
            f"{payload['warnings']}"
        )


def test_env_var_honored_for_available_backend():
    # learn which backend this machine actually supports...
    baseline = _run_subprocess(_IMPORT_PROBE)
    assert baseline.returncode == 0, baseline.stderr
    default_backend = json.loads(baseline.stdout.strip().splitlines()[-1])[
        "backend"
    ]

    # ...then request it explicitly via the env var, in its canonical case
    # AND lowercased (the lowercase form exercises the case-normalized
    # candidate-list lookup that used to crash)
    for env_value in (default_backend, default_backend.lower()):
        result = _run_subprocess(_IMPORT_PROBE, {"HYPERTOOLS_BACKEND": env_value})
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload["backend"].lower() == default_backend.lower()
        assert not any(
            "failed to set matplotlib backend" in w for w in payload["warnings"]
        ), payload["warnings"]


# ---------------------------------------------------------------------------
# F06-002: a failed switch must leave module state exactly as before
# ---------------------------------------------------------------------------

def test_failed_context_switch_restores_state():
    before = (
        B.IN_SET_CONTEXT,
        str(B.HYPERTOOLS_BACKEND),
        B.BACKEND_WARNING,
    )
    with pytest.raises(HypertoolsBackendError):
        with hyp.set_interactive_backend(UNSWITCHABLE_BACKEND):
            pass  # pragma: no cover - never reached
    assert B.IN_SET_CONTEXT is False
    assert str(B.HYPERTOOLS_BACKEND) == before[1]
    assert B.BACKEND_WARNING == before[2]


def test_failed_switch_during_plot_keeps_raising_not_silently_succeeding():
    # direct-call form: the user set a valid-looking backend that cannot be
    # switched to on this machine. An animated plot that WILL BE DISPLAYED
    # (show=True, no save_path -> genuinely needs the interactive backend)
    # must raise the SAME clear error every time -- the first failure must
    # not corrupt IN_SET_CONTEXT and make later plots silently render on the
    # wrong backend.
    #
    # NB: this deliberately uses show=True (not save_path=). Since the
    # 2026-07 release review (headless-backend fix), hypertools only switches
    # to an interactive backend when a live figure is actually displayed --
    # rendering an animation to a FILE needs no GUI backend and must NOT
    # attempt (or fail on) the switch. That no-switch-on-file-export contract
    # is covered by tests/test_backend_headless.py.
    hyp.set_interactive_backend(UNSWITCHABLE_BACKEND)
    data = _traj()
    with pytest.raises(HypertoolsBackendError):
        hyp.plot(data, animate=True, duration=1, frame_rate=5, show=True)
    assert B.IN_SET_CONTEXT is False, (
        "IN_SET_CONTEXT leaked True after a failed backend switch"
    )
    # the failure must repeat, not be masked by the corrupted state
    with pytest.raises(HypertoolsBackendError):
        hyp.plot(data, animate=True, duration=1, frame_rate=5, show=True)


# ---------------------------------------------------------------------------
# F06-003: eager argument validation
# ---------------------------------------------------------------------------

def test_unknown_backend_name_raises_valueerror_naming_choices():
    before = str(B.HYPERTOOLS_BACKEND)
    with pytest.raises(ValueError) as excinfo:
        hyp.set_interactive_backend("bogus")
    msg = str(excinfo.value)
    assert "bogus" in msg
    assert "matplotlib" in msg and "plotly" in msg and "auto" in msg
    # state untouched by the rejected call
    assert str(B.HYPERTOOLS_BACKEND) == before


@pytest.mark.parametrize("bad", [42, ["plotly"], b"plotly", 3.14])
def test_non_string_backend_raises_typeerror(bad):
    before = str(B.HYPERTOOLS_BACKEND)
    with pytest.raises(TypeError) as excinfo:
        hyp.set_interactive_backend(bad)
    msg = str(excinfo.value)
    assert repr(bad) in msg
    assert type(bad).__name__ in msg
    assert str(B.HYPERTOOLS_BACKEND) == before


def test_valid_matplotlib_backend_names_still_accepted():
    hyp.set_interactive_backend("Agg")
    assert str(B.HYPERTOOLS_BACKEND).lower() == "agg"
    hyp.set_interactive_backend("WebAgg")
    assert str(B.HYPERTOOLS_BACKEND).lower() == "webagg"


# ---------------------------------------------------------------------------
# F06-004: nested contexts
# ---------------------------------------------------------------------------

def test_nested_contexts_keep_in_set_context_true_until_outermost_exit():
    assert B.IN_SET_CONTEXT is False
    with hyp.set_interactive_backend("plotly"):
        assert B.IN_SET_CONTEXT is True
        with hyp.set_interactive_backend("matplotlib"):
            assert B.IN_SET_CONTEXT is True
        # regression: inner __exit__ used to reset this to False while the
        # outer context was still active
        assert B.IN_SET_CONTEXT is True
        assert B.PREFERRED_RENDER_BACKEND == "plotly"
    assert B.IN_SET_CONTEXT is False


# ---------------------------------------------------------------------------
# F06-007: resetting the render preference to auto-detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reset_value", ["auto", "AUTO", None])
def test_reset_render_preference_to_auto(reset_value):
    hyp.set_interactive_backend("plotly")
    assert B.PREFERRED_RENDER_BACKEND == "plotly"
    before_mpl_backend = str(B.HYPERTOOLS_BACKEND)
    hyp.set_interactive_backend(reset_value)
    assert B.PREFERRED_RENDER_BACKEND is None
    # the reset must not corrupt the matplotlib-backend state (None used to
    # set HYPERTOOLS_BACKEND to the string 'None')
    assert str(B.HYPERTOOLS_BACKEND) == before_mpl_backend
    # locally ('other' environment), auto-detection renders with matplotlib
    fig = hyp.plot(_traj(), show=False)
    assert type(fig).__module__.startswith("matplotlib")


def test_auto_reset_works_as_context_manager():
    hyp.set_interactive_backend("plotly")
    with hyp.set_interactive_backend("auto"):
        assert B.PREFERRED_RENDER_BACKEND is None
    # outer preference restored on exit
    assert B.PREFERRED_RENDER_BACKEND == "plotly"


# ---------------------------------------------------------------------------
# F06-008: plotly preference must warn when plotly is unavailable
# ---------------------------------------------------------------------------

_NOPLOTLY_PROBE = """
import json, sys, warnings

class _BlockPlotly:
    def find_spec(self, name, path=None, target=None):
        if name == "plotly" or name.startswith("plotly."):
            raise ImportError("plotly blocked for test")
        return None

sys.meta_path.insert(0, _BlockPlotly())
for mod in list(sys.modules):
    if mod == "plotly" or mod.startswith("plotly."):
        del sys.modules[mod]

import hypertools as hyp
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    hyp.set_interactive_backend("plotly")
print(json.dumps([str(x.message) for x in w]))
"""


def test_set_plotly_preference_warns_when_plotly_missing():
    # real subprocess whose import machinery genuinely cannot import plotly
    result = _run_subprocess(_NOPLOTLY_PROBE)
    assert result.returncode == 0, result.stderr
    messages = json.loads(result.stdout.strip().splitlines()[-1])
    assert any(
        "plotly" in m and "pip install" in m for m in messages
    ), (
        "set_interactive_backend('plotly') issued no warning although "
        f"plotly is not importable; warnings: {messages}"
    )


# ---------------------------------------------------------------------------
# F06-005: the docstring must document both accepted forms
# ---------------------------------------------------------------------------

def test_docstring_documents_render_backend_form():
    doc = hyp.set_interactive_backend.__doc__
    assert "'plotly'" in doc, (
        "set_interactive_backend docstring does not document the "
        "'plotly'/'matplotlib' render-backend form"
    )
    assert "'auto'" in doc, (
        "set_interactive_backend docstring does not document the "
        "'auto' reset form"
    )
