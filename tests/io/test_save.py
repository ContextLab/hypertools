import numpy as np
import os


def test_save_roundtrips_array(tmp_path):
    from hypertools.io.save import save
    import pickle
    arr = np.random.RandomState(0).rand(6, 3)
    fname = str(tmp_path / "data.pkl")
    save(arr, fname)
    assert os.path.exists(fname)
    with open(fname, "rb") as f:
        loaded = pickle.load(f)
    assert np.allclose(np.asarray(loaded), arr)


def test_hyp_save_and_io_exposed(tmp_path):
    import hypertools as hyp
    assert callable(hyp.save)
    # hyp.io is the subpackage (no competing classic callable)
    assert hasattr(hyp.io, "load") and hasattr(hyp.io, "save")
    fname = str(tmp_path / "x.pkl")
    hyp.save(np.arange(5), fname)
    assert os.path.exists(fname)
