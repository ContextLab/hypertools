"""set_interactive_backend() must select the RENDER backend (matplotlib vs
plotly), not just the matplotlib backend name (QC 2026-07 release hardening).

Before: `hyp.set_interactive_backend('plotly')` stored 'plotly' as if it were a
matplotlib backend name; a subsequent animated plot raised HypertoolsBackendError
trying to switch matplotlib to a nonexistent 'plotly' backend, and static plots
silently kept rendering with matplotlib. Now 'plotly'/'matplotlib' set a render
preference consulted by resolve_backend(), while real matplotlib backend names
(TkAgg/Agg/...) still switch matplotlib's backend as before.

Real rendering, no mocks; headless (Agg).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.figure
import numpy as np
import pytest

import hypertools as hyp

plotly = pytest.importorskip('plotly')
import plotly.graph_objects as go  # noqa: E402  (must follow the importorskip guard)


def _traj():
    return np.random.default_rng(0).normal(size=(20, 3)).cumsum(axis=0)


@pytest.fixture(autouse=True)
def _reset_backend():
    # ensure each test starts/ends on the default (auto) render preference
    hyp.set_interactive_backend('matplotlib')
    yield
    from hypertools.plot import backend as _b
    _b.PREFERRED_RENDER_BACKEND = None


def test_set_plotly_then_animate_does_not_crash():
    # the exact reported crash
    hyp.set_interactive_backend('plotly')
    out = hyp.plot(_traj(), ndims=3, animate='spin', duration=1, show=False)
    assert isinstance(out, go.Figure)  # plotly renders + embeds frames


def test_set_plotly_switches_static_renderer():
    hyp.set_interactive_backend('plotly')
    assert isinstance(hyp.plot(_traj(), ndims=3, show=False), go.Figure)


def test_set_matplotlib_forces_matplotlib():
    hyp.set_interactive_backend('plotly')
    hyp.set_interactive_backend('matplotlib')
    out = hyp.plot(_traj(), ndims=3, show=False)
    assert isinstance(out, matplotlib.figure.Figure)
    # matplotlib animations still return a HyperAnimation
    from hypertools import HyperAnimation
    anim = hyp.plot(_traj(), ndims=3, animate='spin', duration=1, show=False)
    assert isinstance(anim, HyperAnimation)


def test_context_manager_form_scopes_and_restores():
    from hypertools.plot import backend as _b
    _b.PREFERRED_RENDER_BACKEND = None
    with hyp.set_interactive_backend('plotly'):
        assert isinstance(hyp.plot(_traj(), ndims=3, show=False), go.Figure)
    # restored to the prior (auto/matplotlib) preference outside the block
    assert isinstance(hyp.plot(_traj(), ndims=3, show=False),
                      matplotlib.figure.Figure)


def test_explicit_backend_kwarg_still_overrides():
    hyp.set_interactive_backend('matplotlib')
    assert isinstance(hyp.plot(_traj(), ndims=3, backend='plotly', show=False),
                      go.Figure)


def test_real_matplotlib_backend_name_still_switches():
    # a genuine matplotlib backend name is NOT treated as a render preference
    hyp.set_interactive_backend('Agg')
    assert isinstance(hyp.plot(_traj(), ndims=3, show=False),
                      matplotlib.figure.Figure)


@pytest.mark.parametrize('name', ['Plotly', 'PLOTLY'])
def test_render_backend_name_is_case_insensitive(name):
    # a capitalized render-backend name used to be treated as an mpl backend and
    # (with animate) reproduced the HypertoolsBackendError this routing prevents
    hyp.set_interactive_backend('matplotlib')
    hyp.set_interactive_backend(name)
    try:
        assert isinstance(hyp.plot(_traj(), ndims=3, show=False), go.Figure)
    finally:
        hyp.set_interactive_backend('matplotlib')


def test_explicit_backend_kwarg_case_insensitive():
    hyp.set_interactive_backend('matplotlib')
    assert isinstance(hyp.plot(_traj(), ndims=3, backend='Plotly', show=False),
                      go.Figure)
