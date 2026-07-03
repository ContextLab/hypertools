import numpy as np
import pandas as pd
from hypertools.align.common import Aligner, pad, trim_and_pad


def test_pad_widens_to_c_columns():
    df = pd.DataFrame(np.ones((3, 2)))
    out = pad(df, c=5)
    assert out.shape == (3, 5)
    assert np.allclose(out.iloc[:, :2].to_numpy(), 1.0)
    assert np.allclose(out.iloc[:, 2:].to_numpy(), 0.0)


def test_trim_and_pad_aligns_shapes():
    a = pd.DataFrame(np.random.RandomState(0).rand(5, 3))
    b = pd.DataFrame(np.random.RandomState(1).rand(4, 2))
    out = trim_and_pad([a, b])
    assert out[0].shape[1] == out[1].shape[1] == 3
    assert out[0].shape[0] == out[1].shape[0] == 4  # common rows


def test_aligner_null_fit_transform_returns_data():
    # a null aligner (no fitter/transformer) returns its (trim_and_padded) data
    a = pd.DataFrame(np.random.RandomState(0).rand(6, 3))
    m = Aligner(fitter=lambda data, **k: {}, transformer=lambda data, **k: data,
                required=[], data=None)
    out = m.fit_transform([a, a])
    assert isinstance(out, list) and out[0].shape == (6, 3)
