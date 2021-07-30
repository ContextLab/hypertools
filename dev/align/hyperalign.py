import numpy as np
from .procrustes import Procrustes


class Hyperalign:
    def __init__(self, n_iter=10):
        assert n_iter >= 0, 'Number of iterations must be non-negative'

        self.n_iter = n_iter
        self.proj = None
        self.data = None

    def fit(self, data):
        assert type(data) == list, "data must be specified as a list"
        self.data = data

        n = len(self.data)
        if n <= 1:
            return self.data

        if self.n_iter == 0:
            return self.data

        x = self.data.copy()

        p = None
        for i in range(self.n_iter):
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
        self.proj = p

    def transform(self):
        assert self.proj is not None, "Mapper needs to be trained before use."
        assert self.data is not None, "Need to add data before transforming it."

        return [p.transform(d) for p, d in zip(self.proj, self.data)]

    def fit_transform(self, data):
        self.fit(data)
        return self.transform()
