import numpy as np
import pandas as pd
import hypertools as hyp


weights = hyp.load('weights')
spiral = hyp.load('spiral')


def compare_alignments(a1, a2, tol=1e-5):
    def get_alignment(x):
        if type(x) is pd.DataFrame:
            return x.values
        elif type(x) in [np.array, np.ndarray]:
            return x
        elif type(x) is list:
            return x[0]
        elif hasattr(x, 'proj'):
            return get_alignment(x.proj)
    return np.allclose(get_alignment(a1), get_alignment(a2), atol=tol)


def spiral_alignment_checker(model, known_rot=True, relax=False, tol=1e-5):
    def get_vals(x):
        if hasattr(x, 'values'):
            return x.values
        else:
            return x

    def test_all_close(unaligned, aligned):
        if not relax:
            return np.allclose(aligned[0], aligned[1], atol=tol)
        else:
            d1 = np.max(np.abs(get_vals(unaligned[0]), get_vals(unaligned[1])))
            d2 = np.max(np.abs(get_vals(aligned[0]), get_vals(aligned[1])))
            return d2 - d1 <= tol  # either d1 and d2 are within tol of each other, or d2 < d1

    rot = np.array([[-0.50524616, -0.48383773, -0.71458195],
                    [-0.86275536, 0.26450786, 0.43091621],
                    [-0.01948098, 0.83422817, -0.55107518]])

    aligned_spirals1 = [spiral[0], np.dot(spiral[1], rot)]
    aligned_spirals2 = hyp.align(spiral, model=model)
    aligned_spirals3, fitted_model = hyp.align(spiral, model=model, return_model=True)

    # noinspection DuplicatedCode
    assert test_all_close(spiral, aligned_spirals1)
    assert test_all_close(spiral, aligned_spirals2)
    assert all([test_all_close(spiral, [a, b]) for a, b in zip(aligned_spirals2, aligned_spirals3)])

    if known_rot:
        assert all([np.allclose(a, b, atol=1e-5) for a, b in zip(aligned_spirals1, aligned_spirals3)])
        assert compare_alignments(np.eye(3), fitted_model['model'].proj[0])
        assert compare_alignments(rot, fitted_model['model'].proj[1])


def weights_alignment_checker(model):
    def dists(x, y):
        return np.sqrt(np.sum(np.power(x - y, 2), axis=1))

    def get_mean_dists(x):
        dist_sum = 0
        for i, dx in enumerate(x):
            if hasattr(dx, 'values'):
                dx = dx.values
            for dy in x[:i]:
                if hasattr(dy, 'values'):
                    dy = dy.values
                # if shapes are identical, one may be "flipped" relative to the other
                dist_sum += np.min([np.mean(dists(dx, dy)), np.mean(dists(dx[::-1, :], dy))])
        return dist_sum

    d1 = get_mean_dists(weights)
    d2 = get_mean_dists(hyp.align(weights, model=model))
    assert d1 > d2


def test_procrustes():
    spiral_alignment_checker('Procrustes')


def test_hyperalign():
    spiral_alignment_checker('HyperAlign')
    weights_alignment_checker('HyperAlign')


def test_shared_response_model():
    spiral_alignment_checker('SharedResponseModel', known_rot=False, tol=1e-2)
    weights_alignment_checker('SharedResponseModel')


def test_robust_shared_response_model():
    spiral_alignment_checker('RobustSharedResponseModel', known_rot=False, tol=1e-2, relax=True)
    weights_alignment_checker('RobustSharedResponseModel')


def test_deterministic_shared_response_model():
    spiral_alignment_checker('DeterministicSharedResponseModel', known_rot=False, tol=1e-2, relax=True)
    weights_alignment_checker('DeterministicSharedResponseModel')


def test_null_align():
    spiral2 = hyp.align(spiral, model='NullAlign')
    weights2 = hyp.align(weights, model='NullAlign')

    assert all([np.allclose(x, y) for x, y in zip(spiral, spiral2)])
    assert all([np.allclose(x, y) for x, y in zip(weights, weights2)])


def test_pad():
    a = pd.DataFrame(np.random.randn(10, 3))
    b = pd.DataFrame(np.random.randn(100, 5))

    a_padded = hyp.pad(a, 20)
    assert a_padded.shape == (10, 20)

    auto_padded = hyp.pad([a, b])
    assert auto_padded[0].shape[1] == auto_padded[1].shape[1]
    assert auto_padded[0].shape[1] == 5
    assert auto_padded[0].shape[0] == 10
    assert auto_padded[1].shape[0] == 100

    assert np.allclose(auto_padded[0].iloc[:, :a.shape[1]], a)
    assert np.allclose(auto_padded[0].iloc[:, a.shape[1]:], 0)
    assert np.allclose(auto_padded[1], b)


def test_trim_and_pad():
    a = pd.DataFrame(np.random.randn(15, 20))
    b = pd.DataFrame(np.random.randn(5, 30), index=np.arange(5, 10))

    padded = hyp.trim_and_pad([a, b])

    assert padded[0].shape == padded[1].shape
    assert np.allclose(padded[0].index.values, padded[0].index.values)

    assert np.allclose(a.loc[padded[0].index], padded[0].iloc[:, :a.shape[1]])
    assert np.allclose(padded[0].iloc[:, a.shape[1]:], 0)

    assert np.allclose(b, padded[1])
