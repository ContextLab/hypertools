"""Real numerical regressions for release review findings 1, 3 and 4."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import LinearSegmentedColormap

import hypertools as hyp
from hypertools.manip import Delay, Normalize, Resample, Smooth, ZScore
from hypertools.plot.colors import MatrixColormap


MODELS = [(ZScore, {}), (Normalize, {}), (Smooth, {'kernel_width': 5}),
          (Delay, {'dims': 2}), (Resample, {'n_samples': 7})]


@pytest.mark.parametrize('cls,kwargs', MODELS)
@pytest.mark.parametrize('container', [np.asarray, list, tuple])
def test_one_dimensional_input_has_one_feature_everywhere(cls, kwargs, container):
    x = np.arange(12.) ** 2
    raw = container(x)
    expected = cls(**kwargs).fit_transform(pd.DataFrame(x))
    calls = [hyp.manip(raw, model=cls.__name__, **kwargs),
             cls(**kwargs).fit_transform(raw),
             hyp.Pipeline([cls(**kwargs)]).fit_transform(raw)]
    for actual in calls:
        pd.testing.assert_frame_equal(actual, expected)
    # A nonconstant signal must not silently become one all-zero row.
    assert len(expected) > 1
    assert np.ptp(np.asarray(expected)) > 0


@pytest.mark.parametrize('cls,kwargs', MODELS)
def test_returned_manipulator_reuses_one_dimensional_inputs(cls, kwargs):
    train = np.arange(12.) ** 2
    new = train + 100
    _, model = hyp.manip(train, model=cls.__name__, return_model=True, **kwargs)
    reference = cls(**kwargs).fit(pd.DataFrame(train))
    expected = reference.transform(pd.DataFrame(new))
    for actual in [model.transform(new), model.transform(new.tolist()),
                   hyp.manip(new, model=model)]:
        pd.testing.assert_frame_equal(actual, expected)
    if cls is ZScore:
        np.testing.assert_allclose(expected.to_numpy().ravel(),
                                   (new - train.mean()) / train.std(ddof=1))
    elif cls is Normalize:
        np.testing.assert_allclose(expected.to_numpy().ravel(),
                                   (new - train.min()) / np.ptp(train))


@pytest.mark.parametrize('cls,kwargs', MODELS)
def test_array_datasets_keep_boundaries_and_column_counts(cls, kwargs):
    x = np.arange(24.).reshape(12, 2)
    datasets = (x, x + 100)
    expected = cls(**kwargs).fit_transform([pd.DataFrame(d) for d in datasets])
    for actual in [cls(**kwargs).fit_transform(datasets),
                   hyp.Pipeline([cls(**kwargs)]).fit_transform(datasets),
                   hyp.manip(datasets, model=cls.__name__, **kwargs)]:
        assert len(actual) == 2
        for result, reference in zip(actual, expected):
            pd.testing.assert_frame_equal(result, reference)


@pytest.mark.parametrize('cls', [ZScore, Normalize])
@pytest.mark.parametrize('different_widths', [False, True])
def test_rowwise_lists_keep_their_own_statistics_and_metadata(cls, different_widths):
    a = pd.DataFrame([[1., 3., 8.], [10., 20., 50.]],
                     columns=['a', 'b', 'c'], index=['same', 'same'])
    b = pd.DataFrame([[2., 4., 9.], [20., 30., 70.], [3., 5., 10.]],
                     columns=['x', 'y', 'z'], index=[5, 8, 13])
    if different_widths:
        b = b.iloc[:, :2]
    data = [a, b]
    model = cls(axis=1)
    for actual in [model.fit_transform(data), model.transform(),
                   hyp.manip(data, model=cls.__name__, axis=1),
                   hyp.Pipeline([cls(axis=1)]).fit_transform(data)]:
        for result, original in zip(actual, data):
            values = original.to_numpy()
            if cls is ZScore:
                expected = ((values - values.mean(axis=1, keepdims=True))
                            / values.std(axis=1, ddof=1, keepdims=True))
            else:
                expected = ((values - values.min(axis=1, keepdims=True))
                            / np.ptp(values, axis=1, keepdims=True))
            pd.testing.assert_index_equal(result.index, original.index)
            pd.testing.assert_index_equal(result.columns, original.columns)
            np.testing.assert_allclose(result, expected)
    with pytest.raises(NotImplementedError, match='row-wise'):
        model.transform([a + 1, b + 1])


@pytest.mark.parametrize('gamma', [0.5, 2., 3.])
def test_matrix_colormap_gamma_matches_its_parent_and_exact_curve(gamma):
    anchors = [[0., 0., 0.], [1., 1., 1.]]
    cmap = MatrixColormap('gamma', anchors, N=4096)
    parent = LinearSegmentedColormap.from_list('parent', anchors, N=4096)
    for obj in (cmap, parent):
        obj.set_gamma(gamma)
        obj.set_under('red', alpha=.2)
        obj.set_over('blue', alpha=.4)
        obj.set_bad('green', alpha=.6)
    x = np.array([-.1, .1, .25, .5, .75, .9, 1.1, np.nan])
    np.testing.assert_allclose(cmap(x), parent(x), atol=.002)
    np.testing.assert_allclose(cmap(x), cmap(np.ma.array(x)), atol=.002)
    np.testing.assert_allclose(cmap(.5), [.5 ** gamma] * 3 + [1.])
    cmap.set_gamma(1.)
    np.testing.assert_allclose(cmap(.5), [.5, .5, .5, 1.])
