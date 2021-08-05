import numpy as np
import hypertools as hyp

weights = hyp.load('weights')
spiral = hyp.load('spiral')


def test_procrustes():
    rot = np.array([[-0.50524616, -0.48383773, -0.71458195],
                    [-0.86275536,  0.26450786,  0.43091621],
                    [-0.01948098,  0.83422817, -0.55107518]])

    aligned_spirals1 = np.dot(spiral[1], rot)
    aligned_spirals2 = hyp.align(spiral, model='Procrustes')
    aligned_spirals3, procrustes = hyp.align(spiral, model='Procrustes', return_model=True)

    assert np.allclose(aligned_spirals2[0], aligned_spirals2[1])
    assert all([np.allclose(a, b) for a, b in zip(aligned_spirals2, aligned_spirals3)])
    assert np.allclose(procrustes['model'].proj, rot)


def test_hyperalign():
    rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
           [-0.43426149,  0.87492975, -0.21427761],
           [-0.10761949,  0.18578133,  0.97667976]])
    data2 = np.dot(data1, rot)
    result = align([data1,data2], align='hyper')
    assert np.allclose(result[0],result[1], rtol=1)


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