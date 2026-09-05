# -*- coding: utf-8 -*-
"""``Normalize(mode='isotropic')`` (GH #284, item B): centre a table on its
centroid and divide EVERY column by one shared scalar, so the data's shape
is preserved -- the "centre and scale a point cloud into the unit cube"
recipe the gallery examples used to hand-roll.

Contract under test (see `hypertools.manip.normalize.Normalize`):

- centroid (``data.mean(axis=0)``) lands at ``(min + max) / 2``;
- every column is divided by the SAME scalar,
  ``abs(data - centroid).max()``, so pairwise-distance ratios (the shape)
  are unchanged and a rotated copy is rescaled by the same scalar;
- output lies in ``[min, max]`` with at least one coordinate on a bound;
- ``min=-1, max=1`` reproduces ``(x - mean) / abs(x - mean).max()``
  exactly;
- works through `hyp.manip` kwargs, the dict spec, ``manip=`` inside
  `hyp.plot`, lists (one shared centre/scale), ``return_model=True``
  reuse via `transform`, and `inverse_transform`;
- the default ``mode='minmax'`` is untouched.

All computations are real (numpy / pandas / matplotlib) -- no mocks.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

import hypertools as hyp
from hypertools.manip import Normalize


def _cloud(seed=0, n=60):
    """An anisotropic, off-centre point cloud so per-column min-max and
    isotropic normalization give visibly different results."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) * np.array([1.0, 5.0, 0.2]) + np.array([3.0, -2.0, 10.0])


def _recipe(x):
    """The hand-rolled reference from examples/animate_morph_zoo.py."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=0)
    return x / np.abs(x).max()


def _rotation(seed=1):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


# --------------------------------------------------------------------------
# core contract
# --------------------------------------------------------------------------
def test_isotropic_matches_hand_rolled_recipe_exactly():
    x = _cloud()
    out = np.asarray(hyp.manip(x, model='Normalize', mode='isotropic', min=-1, max=1))
    assert np.allclose(out, _recipe(x))


def test_isotropic_preserves_shape_up_to_one_scalar():
    x = _cloud()
    out = np.asarray(Normalize(mode='isotropic').fit_transform(x))
    d_in, d_out = pdist(x), pdist(out)
    ratios = d_out / d_in
    # every pairwise distance is scaled by the SAME constant
    assert np.allclose(ratios, ratios[0])
    # ... and that constant is 1 / (largest abs deviation from the centroid)
    # times the half-width of the target range ([0, 1] -> 0.5)
    expected = 0.5 / np.abs(x - x.mean(axis=0)).max()
    assert np.isclose(ratios[0], expected)


def test_per_column_minmax_does_not_preserve_shape():
    """Regression guard for the motivating gap: the default mode really is
    anisotropic on this cloud, so the isotropic test above is not vacuous."""
    x = _cloud()
    out = np.asarray(Normalize().fit_transform(x))
    ratios = pdist(out) / pdist(x)
    assert not np.allclose(ratios, ratios[0])


@pytest.mark.parametrize('lo, hi', [(0, 1), (-1, 1), (-3.5, 2.0)])
def test_centroid_lands_at_midpoint_and_range_holds(lo, hi):
    x = _cloud()
    out = np.asarray(Normalize(mode='isotropic', min=lo, max=hi).fit_transform(x))
    assert np.allclose(out.mean(axis=0), (lo + hi) / 2.0)
    assert out.min() >= lo - 1e-12
    assert out.max() <= hi + 1e-12
    # the farthest coordinate sits exactly on a face of the cube
    assert np.isclose(out.min(), lo) or np.isclose(out.max(), hi)


def test_rotated_copy_is_rescaled_by_the_same_scalar():
    x = _cloud()
    r = _rotation()
    centred = x - x.mean(axis=0)
    rotated = centred @ r.T + np.array([100.0, -50.0, 7.0])
    m_a = Normalize(mode='isotropic', min=-1, max=1)
    m_b = Normalize(mode='isotropic', min=-1, max=1)
    m_a.fit(x)
    m_b.fit(rotated)
    # the fitted scale is a single float in isotropic mode ...
    assert isinstance(m_a.peak, float) and isinstance(m_b.peak, float)
    # ... equal to the max-abs deviation, which is NOT rotation invariant in
    # general; so compare the shape-preserving guarantee instead: the two
    # outputs are the same cloud up to the rotation and a scalar
    out_a = np.asarray(m_a.transform(x))
    out_b = np.asarray(m_b.transform(rotated))
    ratios = pdist(out_b) / pdist(out_a)
    assert np.allclose(ratios, ratios[0])
    # and un-rotating out_b recovers out_a up to that scalar
    assert np.allclose((out_b @ r) / ratios[0], out_a)


def test_rotation_about_the_axes_uses_identical_scalar():
    """A rotation that permutes/negates axes keeps the max-abs deviation, so
    the fitted scalar is bit-for-bit the same."""
    x = _cloud()
    perm = x[:, [2, 0, 1]] * np.array([1.0, -1.0, 1.0])
    m_a = Normalize(mode='isotropic')
    m_b = Normalize(mode='isotropic')
    m_a.fit(x)
    m_b.fit(perm)
    assert m_a.peak == m_b.peak


def test_degenerate_single_point_is_centred_not_nan():
    x = np.array([[3.0, -2.0, 10.0]])
    out = np.asarray(Normalize(mode='isotropic', min=-1, max=1).fit_transform(x))
    assert np.allclose(out, 0.0)
    assert not np.isnan(out).any()


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------
def test_dict_spec_entry_point():
    x = _cloud()
    spec = {'model': 'Normalize', 'kwargs': {'mode': 'isotropic', 'min': -1, 'max': 1}}
    out = np.asarray(hyp.manip(x, model=spec))
    assert np.allclose(out, _recipe(x))


def test_list_shares_one_centre_and_scale():
    a, b = _cloud(seed=0), _cloud(seed=1) * 3.0 + 20.0
    outs, model = hyp.manip([a, b], model='Normalize', mode='isotropic', min=-1, max=1,
                            return_model=True)
    assert isinstance(outs, list) and len(outs) == 2
    stacked = np.vstack([a, b])
    centroid = stacked.mean(axis=0)
    scale = np.abs(stacked - centroid).max()
    assert np.allclose(np.asarray(model.baseline), centroid)
    assert np.isclose(model.peak, scale)
    for raw, out in zip((a, b), outs):
        assert np.allclose(np.asarray(out), (raw - centroid) / scale)
    # the joint cloud fills the cube; each member alone need not
    joint = np.vstack([np.asarray(o) for o in outs])
    assert np.isclose(np.abs(joint).max(), 1.0)


def test_return_model_reuse_transforms_new_data_with_fitted_params():
    x = _cloud()
    _, model = hyp.manip(x, model='Normalize', mode='isotropic', min=-1, max=1,
                         return_model=True)
    assert model.is_fitted
    new = _cloud(seed=5) + 1.0
    centroid = x.mean(axis=0)
    scale = np.abs(x - centroid).max()
    expected = (new - centroid) / scale
    # via the Manipulator directly ...
    assert np.allclose(np.asarray(model.transform(new)), expected)
    # ... and via hyp.manip's fitted-model routing (no re-fit)
    reused = np.asarray(hyp.manip(new, model=model))
    assert np.allclose(reused, expected)
    assert np.isclose(model.peak, scale)  # untouched by the reuse


def test_inverse_transform_round_trips():
    x = _cloud()
    for lo, hi in [(0, 1), (-1, 1), (2.0, 9.0)]:
        model = Normalize(mode='isotropic', min=lo, max=hi)
        out = model.fit_transform(pd.DataFrame(x, columns=list('xyz')))
        back = model.inverse_transform(out)
        assert np.allclose(back, x)
        # also from a plain array (what a Pipeline hands between steps)
        assert np.allclose(model.inverse_transform(np.asarray(out)), x)


def test_manip_kwarg_inside_plot(tmp_path):
    x = _cloud()
    spec = {'model': 'Normalize', 'kwargs': {'mode': 'isotropic', 'min': -1, 'max': 1}}
    fig = hyp.plot(x, manip=spec, show=False)
    assert fig is not None
    png = tmp_path / 'iso.png'
    fig.savefig(png)
    assert png.stat().st_size > 0
    # the plotted trajectory IS the isotropically normalized cloud (3-D
    # input, so the reduce stage is a no-op): `return_model=True` hands the
    # per-trace coordinates back in `trace_data`
    bundle = hyp.plot(x, manip=spec, show=False, return_model=True)
    assert isinstance(bundle, dict)
    (trace,) = bundle['trace_data']
    assert np.asarray(trace).shape == x.shape
    assert np.allclose(np.asarray(trace), _recipe(x))
    # and the manip stage is recorded in the returned pipeline
    assert 'manip' in repr(bundle['pipeline'])


def test_plot_manip_string_plus_kwargs_via_pipeline_matches_manip():
    """`hyp.analyze` (the pipeline plot uses) with the same manip spec gives
    the same coordinates as a bare `hyp.manip` call."""
    x = _cloud()
    spec = {'model': 'Normalize', 'kwargs': {'mode': 'isotropic'}}
    via_analyze = np.asarray(hyp.analyze(x, manip=spec))
    via_manip = np.asarray(hyp.manip(x, model=spec))
    assert np.allclose(via_analyze, via_manip)


# --------------------------------------------------------------------------
# regression: the default mode is unchanged
# --------------------------------------------------------------------------
def test_default_minmax_mode_unchanged():
    x = _cloud()
    out = np.asarray(Normalize().fit_transform(x))
    assert np.allclose(out.min(axis=0), 0.0)
    assert np.allclose(out.max(axis=0), 1.0)
    manual = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    assert np.allclose(out, manual)
    model = Normalize()
    model.fit(x)
    assert model.mode == 'minmax'
    assert isinstance(model.peak, pd.Series) and len(model.peak) == 3


def test_default_minmax_axis1_and_inverse_unchanged():
    x = _cloud()
    row = np.asarray(Normalize(axis=1).fit_transform(x))
    assert np.allclose(row.min(axis=1), 0.0)
    assert np.allclose(row.max(axis=1), 1.0)
    model = Normalize(min=-2, max=3)
    out = model.fit_transform(x)
    assert np.allclose(model.inverse_transform(out), x)


def test_get_params_exposes_mode_and_clone_round_trips():
    from sklearn.base import clone
    model = Normalize(mode='isotropic', min=-1, max=1)
    assert model.get_params() == {'axis': 0, 'max': 1, 'min': -1, 'mode': 'isotropic'}
    twin = clone(model)
    assert twin.get_params() == model.get_params()
    x = _cloud()
    assert np.allclose(np.asarray(twin.fit_transform(x)), np.asarray(model.fit_transform(x)))


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="invalid Normalize mode 'iso'"):
        Normalize(mode='iso').fit_transform(_cloud())


def test_isotropic_with_axis1_raises():
    with pytest.raises(ValueError, match='only supports axis=0'):
        Normalize(mode='isotropic', axis=1).fit_transform(_cloud())


def test_isotropic_min_ge_max_raises():
    with pytest.raises(ValueError, match='strictly less'):
        Normalize(mode='isotropic', min=1, max=1).fit_transform(_cloud())
