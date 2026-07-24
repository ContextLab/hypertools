import numpy as np
import pandas as pd
import pytest

from hypertools.align.srm import RobustSharedResponseModel


def _rotated_pair(seed=0, k=4):
    rng = np.random.RandomState(seed)
    base = rng.rand(30, k)
    rot, _ = np.linalg.qr(rng.rand(k, k))
    return [pd.DataFrame(base), pd.DataFrame(base @ rot)]


def test_rsrm_aligns_to_shared_space():
    out = RobustSharedResponseModel(features=3).fit_transform(_rotated_pair())
    assert isinstance(out, list) and len(out) == 2
    assert np.asarray(out[0]).shape[1] == 3


def test_rsrm_via_dispatcher():
    import hypertools as hyp
    # align= is the deliberately-exercised legacy model-spec kwarg; assert
    # its deprecation notice fires alongside the result
    with pytest.warns(DeprecationWarning, match='align= is deprecated'):
        out = hyp.align(_rotated_pair(1), align='RSRM')
    assert isinstance(out, list) and len(out) == 2
