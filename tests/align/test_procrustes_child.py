import numpy as np
import pandas as pd
import pytest
from hypertools.align.procrustes import procrustes, Procrustes
from hypertools.align.null import NullAlign


def test_procrustes_function_recovers_rotation():
    rng = np.random.RandomState(0)
    target = rng.rand(20, 3)
    rot, _ = np.linalg.qr(rng.rand(3, 3))
    source = target @ rot
    out = procrustes(source, target)
    assert np.allclose(out, target, atol=1e-6)


@pytest.mark.parametrize('scaling', [True, False])
def test_procrustes_child_scaling_param(scaling):
    # Procrustes(scaling=False) should skip the norm-matching rescale, so
    # aligning a rescaled copy of target no longer recovers target exactly,
    # while scaling=True (default) still does.
    rng = np.random.RandomState(3)
    target = rng.rand(15, 3)
    rot, _ = np.linalg.qr(rng.rand(3, 3))
    a = pd.DataFrame(target)
    b = pd.DataFrame(2.5 * (target @ rot))
    out = Procrustes(scaling=scaling).fit_transform([a, b])
    matches_target = np.allclose(np.asarray(out[1]), target, atol=1e-6)
    assert matches_target == scaling


def test_procrustes_child_reflection_false_rejects_reflected_data():
    # A reflected copy of target is perfectly recoverable when
    # reflection=True (default), but reflection=False restricts the fit to
    # proper rotations, which cannot represent a det=-1 reflection.
    rng = np.random.RandomState(4)
    target = rng.rand(15, 3)
    rot, _ = np.linalg.qr(rng.rand(3, 3))
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1
    refl = rot.copy()
    refl[:, 0] *= -1
    assert np.linalg.det(refl) < 0

    a = pd.DataFrame(target)
    b = pd.DataFrame(target @ refl)

    out_true = Procrustes(reflection=True).fit_transform([a, b])
    out_false = Procrustes(reflection=False).fit_transform([a, b])

    err_true = np.linalg.norm(np.asarray(out_true[1]) - target)
    err_false = np.linalg.norm(np.asarray(out_false[1]) - target)

    assert err_true < 1e-6
    assert err_false > 100 * err_true


def test_procrustes_child_oblique_recovers_shear():
    # oblique=True lets Procrustes fit a non-orthogonal (shear) map exactly;
    # the default orthogonal-only fit leaves substantial residual error.
    rng = np.random.RandomState(5)
    target = rng.rand(15, 3)
    shear = np.array([[1.0, 0.0, 0.0],
                       [0.4, 1.2, 0.0],
                       [0.0, 0.3, 0.8]])
    a = pd.DataFrame(target)
    b = pd.DataFrame(target @ shear)

    out_oblique = Procrustes(oblique=True).fit_transform([a, b])
    out_default = Procrustes(oblique=False).fit_transform([a, b])

    err_oblique = np.linalg.norm(np.asarray(out_oblique[1]) - target)
    err_default = np.linalg.norm(np.asarray(out_default[1]) - target)

    assert err_oblique < 1e-6
    assert err_default > 100 * err_oblique


def test_procrustes_child_aligns_list_of_dataframes():
    rng = np.random.RandomState(1)
    target = rng.rand(15, 3)
    rot, _ = np.linalg.qr(rng.rand(3, 3))
    a = pd.DataFrame(target)
    b = pd.DataFrame(target @ rot)
    out = Procrustes().fit_transform([a, b])
    assert np.allclose(np.asarray(out[0]), np.asarray(out[1]), atol=1e-6)


def test_null_align_returns_input_rows_cols():
    a = pd.DataFrame(np.random.RandomState(2).rand(8, 4))
    out = NullAlign().fit_transform([a, a])
    assert out[0].shape == (8, 4)
