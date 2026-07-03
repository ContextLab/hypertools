import numpy as np
from hypertools.core.model import apply_model


def test_apply_model_accepts_fork_dict_form():
    # fork spec: {'model', 'args', 'kwargs'} must work the same as {'model','params'}
    data = np.random.RandomState(0).rand(20, 5)
    out = apply_model(data, {"model": "PCA", "args": [], "kwargs": {"n_components": 2}},
                      format_data=False)
    assert np.asarray(out).shape == (20, 2)


def test_apply_model_devtwo_dict_form_still_works():
    data = np.random.RandomState(0).rand(20, 5)
    out = apply_model(data, {"model": "PCA", "params": {"n_components": 2}},
                      format_data=False)
    assert np.asarray(out).shape == (20, 2)
