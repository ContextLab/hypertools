# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np

from .procrustes import Procrustes
from .common import Aligner

from ..core import get_default_options


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
            p = Procrustes(target=template / j)
            template += p.fit_transform(x[j])
        template /= len(x)

        # STEP 2: NEW COMMON TEMPLATE
        #  - align each subj to template
        template2 = np.zeros_like(template)
        for j in range(0, len(x)):
            p = Procrustes(target=template)
            template2 += p.fit_transform(x[j])
        template2 /= len(x)

        # align each subj to template2
        p = [Procustes(target=template2) for _ in x]
        x = [m.fit(i) for m, i in zip(p, x)]

    p = [Procustes(target=template2) for _ in data]
    _ = [m.fit(d) for m, d in zip(p, data)]
    return {'proj': p}


def transformer(data, **kwargs):
    assert 'proj' in kwargs.keys(), "Transformer needs to be trained before use."
    assert type(proj) is list, "Projection must be a list"
    assert type(data) is list, "Data must be a list"
    assert len(proj) == len(data), "Must have one projection per data matrix"
    return [p.transform(d) for p, d in zip(proj, data)]


class HyperAlign(Aligner):
    """
    Base class for HyperAlign objects.  Takes a single keyword argument, n_iter, which specifies how many iterations
    to run (default: 10).
    """
    def __init__(self, **kwargs):
        opts = dw.core.update_dict(get_default_options()['HyperAlign'], kwargs)
        assert opts['n_iter'] >= 0, 'Number of iterations must be non-negative'
        required = ['proj', 'n_iter']
        super().__init__(required=required, fitter=fitter, transformer=transformer, data=None, **opts)

        for k, v in opts.items():
            setattr(self, k, v)
        self.required = required
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
