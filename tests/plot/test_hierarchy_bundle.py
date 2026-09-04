"""What `return_model=True` returns, and what it must NOT redefine.

FLAT inputs only -- the hierarchical bundle assertions live in
tests/plot/test_column_multiindex.py (Task 5), so this module passes
standalone the moment Task 4's keys exist.

There is no `return_data=` parameter (verified: `def plot(` at plot.py:517,
`return_model=False` at plot.py:579, no `return_data` anywhere in
hypertools/). Every test here uses `return_model=True, show=False`.
"""
import matplotlib
matplotlib.use("Agg")

import inspect

import numpy as np
import pytest

import hypertools as hyp

# Asking for 5 components while the display path allows 3 legitimately warns
# -- twice per plot() call, since the spec is applied by analyze() and again
# by the display reducer before the IncrementalPCA fallback. The two tests
# that use FIVE_D assert it rather than letting it leak: this suite is held
# to zero unasserted warnings.
DIMS_WARNING = 'Unequal values passed to dims and n_components'

# A reduce spec pinning MORE than three components, so the display-only
# projection at plot.py:2886-2919 actually runs and `xform_data` (captured
# at plot.py:2827, BEFORE it) keeps the 5-D arrays. Verified in this repo:
# xform_data[0].shape == (60, 5) while the drawn artist is 3-D.
FIVE_D = {'model': 'PCA', 'args': [], 'kwargs': {'n_components': 5}}


def flat_data(n=2, T=40, k=12, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(T, k)) for _ in range(n)]


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def test_no_return_data_parameter_exists():
    """v2 of this plan invented one; plot() takes **kwargs, so passing it
    would silently leak into backend kwargs instead of failing."""
    assert 'return_data' not in inspect.signature(hyp.plot).parameters


def test_bundle_keys_are_stable():
    out = hyp.plot(flat_data(), '-', return_model=True, show=False)
    assert set(out) == {'fig', 'xform_data', 'trace_data', 'trace_metadata',
                        'animation', 'pipeline', 'models', 'predict'}


def test_flat_input_bundle_is_unchanged():
    out = hyp.plot(flat_data(), '-', return_model=True, show=False)
    assert len(out['xform_data']) == 2
    assert out['trace_metadata'] is None


def test_flat_input_trace_data_is_xform_data_when_no_display_projection():
    """The COMMON case -- and the only one in which the two keys may be the
    same object. Contract 5 makes this conditional, not universal."""
    out = hyp.plot(flat_data(), '-', return_model=True, show=False)
    assert out['trace_data'] is out['xform_data']


def test_display_projection_makes_trace_data_diverge_from_xform_data():
    """Contract 5, the counterexample -- on FLAT input.

    `xform_data = copy.copy(xform)` (plot.py:2827) happens BEFORE the
    display-dimensionality enforcement (plot.py:2886-2919), which REBINDS
    `xform` to a new list. So a reduce spec pinning 5 components leaves
    `xform_data` 5-D while the plotted trajectory is 3-D.
    """
    with pytest.warns(UserWarning, match=DIMS_WARNING):
        out = hyp.plot(flat_data(n=1), '-', reduce=FIVE_D, return_model=True,
                       show=False)
    assert np.asarray(out['xform_data'][0]).shape == (40, 5)
    assert np.asarray(out['trace_data'][0]).shape[1] == 3
    assert out['trace_data'] is not out['xform_data']
    assert np.asarray(_ax(out['fig']).lines[0].get_data_3d()).shape[0] == 3


def test_bundled_forecasts_correspond_to_trace_data_not_xform_data():
    """Contract 5's headline: forecasts follow `trace_data`, always."""
    with pytest.warns(UserWarning, match=DIMS_WARNING):
        out = hyp.plot(flat_data(n=1), '-', reduce=FIVE_D, predict='Kalman',
                       t=2, return_model=True, show=False)
    forecast = np.asarray(out['predict']['forecasts'][0], dtype=float)
    from_trace = np.asarray(
        hyp.predict(np.asarray(out['trace_data'][0]), model='Kalman', t=2),
        dtype=float)
    assert forecast.shape[1] == 3
    assert np.allclose(forecast, from_trace, rtol=1e-6, atol=1e-6)
    from_xform = np.asarray(
        hyp.predict(np.asarray(out['xform_data'][0]), model='Kalman', t=2),
        dtype=float)
    assert from_xform.shape[1] == 5, 'the two spaces genuinely differ here'
