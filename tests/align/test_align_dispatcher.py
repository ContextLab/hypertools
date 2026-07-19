import numpy as np
import pandas as pd
from hypertools.align.align import align


def test_dispatcher_hyperalign_by_name():
    rng = np.random.RandomState(0)
    base = rng.rand(20, 3)
    rot, _ = np.linalg.qr(rng.rand(3, 3))
    out = align([pd.DataFrame(base), pd.DataFrame(base @ rot)], model='HyperAlign')
    assert isinstance(out, list) and len(out) == 2


def test_dispatcher_null_by_name():
    a = pd.DataFrame(np.random.RandomState(1).rand(10, 4))
    out = align([a, a], model='NullAlign')
    assert out[0].shape == (10, 4)


def test_dispatcher_accepts_arrays():
    rng = np.random.RandomState(2)
    base = rng.rand(12, 3)
    out = align([base, base.copy()], model='Procrustes')
    assert np.allclose(np.asarray(out[0]), np.asarray(out[1]), atol=1e-6)
