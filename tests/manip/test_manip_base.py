import numpy as np
import pandas as pd
from hypertools.manip.common import Manipulator


def _mean_center_fitter(data, **kwargs):
    return {"mean": data.mean(axis=0)}


def _mean_center_transformer(data, **kwargs):
    return data - kwargs["mean"]


def test_manipulator_fit_transform_roundtrip():
    df = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3), columns=list("abc"))
    m = Manipulator(fitter=_mean_center_fitter, transformer=_mean_center_transformer,
                    required=["mean"], data=None)
    out = m.fit_transform(df)
    assert np.allclose(out.mean(axis=0).to_numpy(), 0.0)
