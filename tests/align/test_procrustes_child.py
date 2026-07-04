import numpy as np
import pandas as pd
from hypertools.align.procrustes import procrustes, Procrustes
from hypertools.align.null import NullAlign


def test_procrustes_function_recovers_rotation():
    rng = np.random.RandomState(0)
    target = rng.rand(20, 3)
    rot, _ = np.linalg.qr(rng.rand(3, 3))
    source = target @ rot
    out = procrustes(source, target)
    assert np.allclose(out, target, atol=1e-6)


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
