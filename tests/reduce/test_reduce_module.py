import numpy as np
import pandas as pd


def test_reduce_new_path_importable_and_reduces():
    from hypertools.reduce.reduce import reduce
    data = [np.random.RandomState(0).rand(10, 6) for _ in range(2)]
    out = reduce(data, ndims=3)
    assert isinstance(out, list) and out[0].shape == (10, 3)


def test_reduce_shim_is_same_function():
    from hypertools.reduce.reduce import reduce as new_reduce
    from hypertools.tools.reduce import reduce as old_reduce
    assert new_reduce is old_reduce


def test_reduce_registry_models_still_exposed_via_tools():
    # core.model._build_registry imports `from ..tools.reduce import models`
    from hypertools.tools.reduce import models
    assert 'PCA' in models and 'IncrementalPCA' in models


def test_reduce_accepts_dataframe():
    from hypertools.reduce.reduce import reduce
    df = pd.DataFrame(np.random.RandomState(1).rand(12, 5))
    out = reduce(df, ndims=2)
    assert np.asarray(out).shape == (12, 2)


def test_describe_new_path():
    from hypertools.reduce.describe import describe
    data = np.random.RandomState(2).rand(20, 8)
    result = describe(data, max_dims=4, show=False)
    assert 'average' in result and 'individual' in result
