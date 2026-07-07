# -*- coding: utf-8 -*-

import numpy as np
import pytest

from hypertools.align.procrustes import procrustes
from hypertools.io.load import load


def test_procrustes_func():
    target = load('spiral')[0]
    rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
           [-0.43426149,  0.87492975, -0.21427761],
           [-0.10761949,  0.18578133,  0.97667976]])
    source = np.dot(target, rot)
    source_aligned = procrustes(source,target)
    assert np.allclose(target,source_aligned)


def _rotation(rng, m, det_sign=1.0):
    rot, _ = np.linalg.qr(rng.rand(m, m))
    if np.sign(np.linalg.det(rot)) != det_sign:
        rot[:, 0] *= -1
    return rot


@pytest.mark.parametrize('k', [3.7, 0.2])
def test_procrustes_scaling_false_preserves_input_norm(k):
    # scaling=False should return a norm-preserving (orthogonal-only)
    # projection: proj = T with no rescaling applied, so
    # ||source @ T|| == ||source|| exactly (T is orthogonal), whereas
    # scaling=True rescales the result to match target's norm instead.
    rng = np.random.RandomState(0)
    target = rng.rand(20, 3)
    rot = _rotation(rng, 3, det_sign=1.0)
    source = k * (target @ rot)

    out_scaled = procrustes(source, target, scaling=True)
    out_unscaled = procrustes(source, target, scaling=False)

    # scaling=True recovers target (up to the applied scale correction)
    assert np.allclose(out_scaled, target, atol=1e-6)
    # scaling=False leaves the norm of the *input* untouched
    assert np.isclose(np.linalg.norm(out_unscaled), np.linalg.norm(source), atol=1e-6)
    # the two branches must actually produce different numeric results
    assert not np.allclose(out_scaled, out_unscaled, atol=1e-3)


def test_procrustes_reflection_false_cannot_recover_a_reflection():
    # Build a genuine reflection (det = -1) relating source to target.
    # reflection=True can recover it exactly; reflection=False is
    # constrained to proper rotations (det = +1) and therefore cannot,
    # so its residual error must be much larger.
    rng = np.random.RandomState(0)
    target = rng.rand(20, 3)
    rot = _rotation(rng, 3, det_sign=1.0)
    refl = rot.copy()
    refl[:, 0] *= -1
    assert np.linalg.det(refl) < 0

    source = target @ refl

    out_reflect_true = procrustes(source, target, reflection=True)
    out_reflect_false = procrustes(source, target, reflection=False)

    err_true = np.linalg.norm(out_reflect_true - target)
    err_false = np.linalg.norm(out_reflect_false - target)

    assert np.allclose(out_reflect_true, target, atol=1e-6)
    assert not np.allclose(out_reflect_false, target, atol=1e-2)
    # reflection=False must do meaningfully worse than reflection=True
    assert err_false > 100 * err_true


def test_procrustes_oblique_true_recovers_non_orthogonal_map():
    # A shear is an invertible but non-orthogonal linear map. oblique=True
    # solves the (least-squares) linear system directly and should recover
    # it essentially exactly; the default orthogonal-only fit cannot.
    rng = np.random.RandomState(0)
    target = rng.rand(20, 3)
    shear = np.array([[1.0, 0.0, 0.0],
                       [0.7, 1.3, 0.0],
                       [0.0, 0.2, 0.9]])
    assert not np.allclose(shear.T @ shear, np.eye(3))
    source = target @ shear

    out_oblique = procrustes(source, target, oblique=True)
    out_orthogonal = procrustes(source, target, oblique=False)

    assert np.allclose(out_oblique, target, atol=1e-6)
    assert not np.allclose(out_orthogonal, target, atol=1e-2)
    assert np.linalg.norm(out_oblique - target) < 1e-6 * np.linalg.norm(out_orthogonal - target)


def test_procrustes_oblique_rcond_changes_solution_for_ill_conditioned_source():
    # oblique_rcond controls the cutoff used by np.linalg.lstsq. With a
    # near-collinear (ill-conditioned) source, a looser vs. tighter cutoff
    # must yield numerically different least-squares solutions.
    rng = np.random.RandomState(0)
    target = rng.rand(20, 3)
    source = target.copy()
    source[:, 2] = source[:, 0] + 1e-12 * rng.rand(20)

    out_default_rcond = procrustes(source, target, oblique=True)
    out_tight_rcond = procrustes(source, target, oblique=True, oblique_rcond=1e-3)

    assert not np.allclose(out_default_rcond, out_tight_rcond, atol=1e-6)


def test_procrustes_reduction_true_projects_to_lower_dimension():
    # When the source has more features than the target, reduction=False
    # must raise, while reduction=True allows the (lower-dimensional)
    # mapping and returns output with the target's dimensionality.
    rng = np.random.RandomState(0)
    source = rng.rand(20, 3)
    rot = _rotation(rng, 3, det_sign=1.0)
    target = (source @ rot)[:, :2]

    with pytest.raises(ValueError):
        procrustes(source, target, reduction=False, format_data=False)

    out = procrustes(source, target, reduction=True, format_data=False)
    out = np.asarray(out)

    assert out.shape == target.shape
    err = np.linalg.norm(out - target)
    # a real (non-tautological) quality bound: reduction should recover
    # the target far better than an unrelated matrix of the same shape
    baseline_err = np.linalg.norm(rng.rand(*target.shape) - target)
    assert err < 0.25 * baseline_err
