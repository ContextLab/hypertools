#!/usr/bin/env python
from __future__ import division
from builtins import zip
from builtins import range

import numpy as np

class Procrustes:
    def __init__(self, scaling=True, reflection=True, reduction=False, oblique=False, oblique_rcond=-1):
        self.scaling = scaling
        self.reflection = reflection
        self.reduction = reduction
        self.oblique = oblique
        self.oblique_rcond = oblique_rcond
        self.proj = None

    # copied from pymvpa2 toolbox...could be cleaned up
    def fit(self, source, target):
        def align(source, target):
            datas = (source, target)
            sn, sm = source.shape
            tn, tm = target.shape

            # Check the sizes
            if sn != tn:
                raise ValueError("Data for both spaces should have the same number of samples. \
                                  Got %d in template and %d in target space" % (sn, tn))

            # Sums of squares
            ssqs = [np.sum(d ** 2, axis=0) for d in datas]

            # TODO: check for being invariant?
            #       needs to be tuned up properly and not raise but handle
            for i in range(2):
                if np.all(ssqs[i] <= np.abs((np.finfo(datas[i].dtype).eps
                                             * sn) ** 2)):
                    raise ValueError("For now do not handle invariant in time datasets")

            norms = [np.sqrt(np.sum(ssq)) for ssq in ssqs]
            normed = [data / norm for (data, norm) in zip(datas, norms)]

            # add new blank dimensions to template space if needed
            if sm < tm:
                normed[0] = np.hstack((normed[0], np.zeros((sn, tm - sm))))

            if sm > tm:
                if self.reduction:
                    normed[1] = np.hstack((normed[1], np.zeros((sn, sm - tm))))
                else:
                    raise ValueError("reduction=False, so mapping from \
                                      higher dimensionality \
                                      template space is not supported. template space had %d \
                                      while target %d dimensions (features)" % (sm, tm))

            source, target = normed
            if self.oblique:
                # Just do silly linear system of equations ;) or naive
                # inverse problem
                if sn == sm and tm == 1:
                    T = np.linalg.solve(source, target)
                else:
                    T = np.linalg.lstsq(source, target, rcond=self.oblique_rcond)[0]
                ss = 1.0
            else:
                # Orthogonal transformation
                # figure out optimal rotation
                U, s, Vh = np.linalg.svd(np.dot(target.T, source),
                                         full_matrices=False)
                T = np.dot(Vh.T, U.T)

                if not self.reflection:
                    # then we need to assure that it is only rotation
                    # "recipe" from
                    # http://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
                    # for more and info and original references, see
                    # http://dx.doi.org/10.1007%2FBF02289451
                    nsv = len(s)
                    s[:-1] = 1
                    s[-1] = np.linalg.det(T)
                    T = np.dot(U[:, :nsv] * s, Vh)

                # figure out scale and final translation
                # XXX with reflection False -- not sure if here or there or anywhere...
                ss = sum(s)

            # if we were to collect standardized distance
            # std_d = 1 - sD**2

            # select out only relevant dimensions
            if sm != tm:
                T = T[:sm, :tm]

            # Assign projection
            if self.scaling:
                scale = ss * norms[1] / norms[0]
                proj = scale * T
            else:
                proj = T

            return proj

        if type(source) == list:
            self.proj = [align(s, target) for s in source]
        else:
            self.proj = align(source, target)

    def transform(self, data, index=0):
        def xform(data, proj):
            if proj is None:
                raise RuntimeError("Mapper needs to be trained before use.")
            d = np.asmatrix(data)

            # Do projection
            res = (d * proj).A
            return res

        if type(self.proj) == list:
            if type(data) == list:
                assert len(self.proj) == len(data), "Data must either be passed in as an individual matrix, or must be of the same length as the fitted list of projections"
                return [xform(d, p) for zip(data, self.proj)]
            else:
                return xform(data, self.proj[index])
        else:
            if type(data) == list:
                return [xform(d, self.proj) for d in data]
            else:
                return xform(data, self.proj)
    
    def fit_transform(self, source, target):
        self.fit(source, target)
        return self.transform(source)
