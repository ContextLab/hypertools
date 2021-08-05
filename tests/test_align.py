import numpy as np
import hypertools as hyp
from scipy.spatial.distance import cdist

weights = hyp.load('weights')
spiral = hyp.load('spiral')


def test_procrustes():
    rot = np.array([[-0.50524616, -0.48383773, -0.71458195],
                    [-0.86275536,  0.26450786,  0.43091621],
                    [-0.01948098,  0.83422817, -0.55107518]])

    aligned_spirals1 = [spiral[0], np.dot(spiral[1], rot)]
    aligned_spirals2 = hyp.align(spiral, model='Procrustes')
    aligned_spirals3, procrustes = hyp.align(spiral, model='Procrustes', return_model=True)

    # noinspection DuplicatedCode
    assert not np.allclose(aligned_spirals1[0], aligned_spirals1[1])
    assert np.allclose(aligned_spirals2[0], aligned_spirals2[1])
    assert all([np.allclose(a, b, atol=1e-5) for a, b in zip(aligned_spirals2, aligned_spirals3)])
    assert all([np.allclose(a, b, atol=1e-5) for a, b in zip(aligned_spirals1, aligned_spirals3)])

    assert np.allclose(procrustes['model'].proj[0], np.eye(3))
    assert np.allclose(procrustes['model'].proj[1], rot)


def test_hyperalign():
    def get_mean_dists(x):
        dist_sum = 0
        for i, dx in enumerate(x):
            for dy in x[:i]:
                dist_sum += np.mean(cdist(dx, dy))
        return dist_sum

    rot = np.array([[-0.50524616, -0.48383773, -0.71458195],
                    [-0.86275536, 0.26450786, 0.43091621],
                    [-0.01948098, 0.83422817, -0.55107518]])
    aligned_spirals1 = [spiral[0], np.dot(spiral[1], rot)]
    aligned_spirals2 = hyp.align(spiral, model='HyperAlign')
    aligned_spirals3, hyper = hyp.align(spiral, model='HyperAlign', return_model=True)

    # noinspection DuplicatedCode
    assert np.allclose(aligned_spirals1[0], aligned_spirals1[1])
    assert np.allclose(aligned_spirals2[0], aligned_spirals2[1])
    assert all([np.allclose(a, b) for a, b in zip(aligned_spirals2, aligned_spirals3)])
    assert all([np.allclose(a, b) for a, b in zip(aligned_spirals1, aligned_spirals3)])
    assert np.allclose(hyper['model'].proj, rot)

    d1 = get_mean_dists(weights)
    d2 = get_mean_dists(hyp.align(weights, model='HyperAlign'))
    assert d1 > d2



def test_shared_response_model():
    rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
           [-0.43426149,  0.87492975, -0.21427761],
           [-0.10761949,  0.18578133,  0.97667976]])
    data2 = np.dot(data1, rot)
    result = align([data1,data2], align='SRM')
    assert np.allclose(result[0],result[1], rtol=1)


def test_robust_shared_response_model():
    pass


def test_null_align():
    pass


def test_deterministic_shared_response_model():
    pass


def test_pad():
    pass


def test_trim_and_pad():
    pass


test_procrustes()
test_hyperalign()