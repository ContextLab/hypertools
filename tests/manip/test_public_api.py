import numpy as np
import hypertools as hyp


def test_hyp_manip_exposed():
    out = hyp.manip(np.random.RandomState(0).rand(20, 3), model="ZScore")
    assert np.allclose(np.asarray(out).mean(axis=0), 0.0, atol=1e-9)


def test_hyp_normalize_compat_still_present():
    # dev-2.0 array/mode API must be unchanged
    out = hyp.normalize(np.random.RandomState(0).rand(10, 4), normalize="across")
    assert np.asarray(out).shape == (10, 4)
