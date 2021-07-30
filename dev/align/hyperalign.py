import numpy as np

from .procrustes import Procrustes
from .common import Aligner


def fitter(data, n_iter=10):
    assert type(data) == list, "data must be specified as a list"

    n = len(data)
    if n <= 1:
        return data

    if n_iter == 0:
        return data

    x = data.copy()

    p = None
    for i in range(n_iter):
        # STEP 1: TEMPLATE
        template = np.copy(x[0])
        for j in range(1, len(x)):
            p = Procrustes()
            template += p.fit_transform(x[j], template / (j + 1))
        template /= len(x)

        # STEP 2: NEW COMMON TEMPLATE
        #  - align each subj to template
        template2 = np.zeros_like(template)
        for j in range(0, len(x)):
            p = Procrustes()
            template2 += p.fit_transform(x[j], template)
        template2 /= len(x)

        # align each subj to template2
        p = [Procustes() for _ in x]
        x = [m.fit(i, template2) for i in x]
    return {'proj': p}


def transformer(data, **kwargs):
    assert 'proj' in kwargs.keys(), "Transformer needs to be trained before use."
    assert type(proj) is list, "Projection must be a list"
    assert type(data) is list, "Data must be a list"
    assert len(proj) == len(data), "Must have one projection per data matrix"
    return [p.transform(d) for p, d in zip(proj, data)]


class HyperAlign(Aligner):
    def __init__(self, n_iter=10):
        assert n_iter >= 0, 'Number of iterations must be non-negative'
        super().__init__(n_iter=n_iter, required=['proj'], fitter=fitter, transformer=transformer, data=None)
