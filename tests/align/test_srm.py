import numpy as np
import pandas as pd
from hypertools.align.srm import SharedResponseModel, DeterministicSharedResponseModel


def _rotated_pair(seed=0, k=4):
    rng = np.random.RandomState(seed)
    base = rng.rand(30, k)
    rot, _ = np.linalg.qr(rng.rand(k, k))
    return [pd.DataFrame(base), pd.DataFrame(base @ rot)]


def test_srm_aligns_to_shared_space():
    out = SharedResponseModel(features=3).fit_transform(_rotated_pair())
    assert isinstance(out, list) and len(out) == 2
    # shared responses should be correlated across the two views
    assert np.corrcoef(np.asarray(out[0]).ravel(),
                       np.asarray(out[1]).ravel())[0, 1] > 0.5


def test_detsrm_runs_and_shapes():
    out = DeterministicSharedResponseModel(features=3).fit_transform(_rotated_pair(1))
    assert len(out) == 2 and np.asarray(out[0]).shape[1] == 3


def test_rsrm_now_exported():
    # external.brainiak now vendors SRM + DetSRM + RSRM (see tests/align/test_rsrm.py
    # for full RobustSharedResponseModel coverage); this guard test previously
    # asserted the pre-vendoring state and has been flipped to match.
    # NB: use `from hypertools.align import srm` (not `import hypertools.align.srm
    # as srm`) — the classic `hyp.align` callable owns the `hypertools.align`
    # attribute, so the chained-attribute import form is shadowed by design. The
    # submodule still resolves via the import machinery / sys.modules.
    from hypertools.align import srm
    assert hasattr(srm, 'RobustSharedResponseModel')
