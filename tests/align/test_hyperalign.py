import numpy as np
import pandas as pd
from hypertools.align.hyperalign import HyperAlign


def test_hyperalign_recovers_rotation_of_two_datasets():
    rng = np.random.RandomState(0)
    base = rng.rand(20, 4)
    rot, _ = np.linalg.qr(rng.rand(4, 4))
    a = pd.DataFrame(base)
    b = pd.DataFrame(base @ rot)
    out = HyperAlign(n_iter=10).fit_transform([a, b])
    # aligned datasets should be close (hyperalignment of a pure rotation)
    assert np.corrcoef(np.asarray(out[0]).ravel(),
                       np.asarray(out[1]).ravel())[0, 1] > 0.95


def test_hyperalign_preserves_scale_across_iterations():
    rng = np.random.RandomState(1)
    data = [pd.DataFrame(rng.rand(15, 3)) for _ in range(3)]
    out = HyperAlign(n_iter=10).fit_transform(data)
    norms = [np.linalg.norm(np.asarray(o)) for o in out]
    # rescaling keeps magnitudes on the original order (not collapsed to ~0)
    assert min(norms) > 1e-3


def test_hyperalign_recovers_pure_rotation_tightly():
    rng = np.random.RandomState(3)
    base = rng.rand(40, 3)
    rot, _ = np.linalg.qr(rng.rand(3, 3))          # orthonormal rotation
    a, b = pd.DataFrame(base), pd.DataFrame(base @ rot)
    out = HyperAlign(n_iter=10).fit_transform([a, b])
    assert np.allclose(np.asarray(out[0]), np.asarray(out[1]), atol=1e-4)
