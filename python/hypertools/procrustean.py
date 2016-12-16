# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Procrustean rotation mapper"""

# __docformat__ = 'restructuredtext'

import numpy as np
# from mvpa2.base import externals
# from mvpa2.base.param import Parameter
# from mvpa2.base.constraints import EnsureChoice
# from mvpa2.base.types import is_datasetlike
# from mvpa2.mappers.projection import ProjectionMapper

# from mvpa2.base import warning
# if __debug__:
#     from mvpa2.base import debug



class Procrustes(object):
    """Mapper to project from one space to another using Procrustean
    transformation (shift + scaling + rotation).

    Training this mapper requires data for both source and target space to be
    present in the training dataset. The source space data is taken from the
    training dataset's ``samples``, while the target space is taken from a
    sample attribute corresponding to the ``space`` setting of the
    ProcrusteanMapper.

    See: http://en.wikipedia.org/wiki/Procrustes_transformation
    """
    # scaling = Parameter(True, constraints='bool',
    #             doc="""Estimate a global scaling factor for the transformation
    #                    (no longer rigid body)""")
    # reflection = Parameter(True, constraints='bool',
    #              doc="""Allow for the data to be reflected (so it might not be
    #                  a rotation. Effective only for non-oblique transformations.
    #                  """)
    # reduction = Parameter(True, constraints='bool',
    #              doc="""If true, it is allowed to map into lower-dimensional
    #                  space. Forward transformation might be suboptimal then and
    #                  reverse transformation might not recover all original
    #                  variance.""")
    # oblique = Parameter(False, constraints='bool',
    #              doc="""Either to allow non-orthogonal transformation -- might
    #                  heavily overfit the data if there is less samples than
    #                  dimensions. Use `oblique_rcond`.""")
    # oblique_rcond = Parameter(-1, constraints='float',
    #              doc="""Cutoff for 'small' singular values to regularize the
    #                  inverse. See :class:`~numpy.linalg.lstsq` for more
    #                  information.""")
    # svd = Parameter('numpy', constraints=EnsureChoice('numpy', 'scipy', 'dgesvd'),
    #              doc="""Implementation of SVD to use. dgesvd requires ctypes to
    #              be available.""")

    def __init__(self, scaling=True, reflection=True, reduction=True, oblique=False, oblique_rcond=-1):
        self._scale = None
        self._demean = False
        def params():
            self.scaling = scaling
            self.reflection = reflection
            self.reduction = reduction
            self.oblique = oblique
            self.oblique_rcond = oblique_rcond
            self.svd = 'numpy'
            return self
        self.params = params()

    def fit(self, source, target):
        params = self.params
        # Since it is unsupervised, we don't care about labels
        datas = ()
        odatas = ()
        means = ()
        shapes = ()

        assess_residuals = False

        # target = source.sa[self.get_space()].value

        for i, ds in enumerate((source, target)):
            data = ds
            if assess_residuals:
                odatas += (data,)
            if self._demean:
                if i == 0:
                    mean = self._offset_in
                else:
                    mean = data.mean(axis=0)
                data = data - mean
            else:
                # no demeaning === zero means
                mean = np.zeros(shape=data.shape[1:])
            means += (mean,)
            datas += (data,)
            shapes += (data.shape,)

        # shortcuts for sizes
        sn, sm = shapes[0]
        tn, tm = shapes[1]

        # Check the sizes
        if sn != tn:
            raise ValueError, "Data for both spaces should have the same " \
                  "number of samples. Got %d in source and %d in target space" \
                  % (sn, tn)

        # Sums of squares
        ssqs = [np.sum(d**2, axis=0) for d in datas]

        # XXX check for being invariant?
        #     needs to be tuned up properly and not raise but handle
        for i in xrange(2):
            if np.all(ssqs[i] <= np.abs((np.finfo(datas[i].dtype).eps
                                       * sn * means[i] )**2)):
                raise ValueError, "For now do not handle invariant in time datasets"

        norms = [ np.sqrt(np.sum(ssq)) for ssq in ssqs ]
        normed = [ data/norm for (data, norm) in zip(datas, norms) ]

        # add new blank dimensions to source space if needed
        if sm < tm:
            normed[0] = np.hstack( (normed[0], np.zeros((sn, tm-sm))) )

        if sm > tm:
            if params.reduction:
                normed[1] = np.hstack( (normed[1], np.zeros((sn, sm-tm))) )
            else:
                raise ValueError, "reduction=False, so mapping from " \
                      "higher dimensionality " \
                      "source space is not supported. Source space had %d " \
                      "while target %d dimensions (features)" % (sm, tm)

        source, target = normed
        if params.oblique:
            # Just do silly linear system of equations ;) or naive
            # inverse problem
            if sn == sm and tm == 1:
                T = np.linalg.solve(source, target)
            else:
                T = np.linalg.lstsq(source, target, rcond=params.oblique_rcond)[0]
            ss = 1.0
        else:
            # Orthogonal transformation
            # figure out optimal rotation
            U, s, Vh = np.linalg.svd(np.dot(target.T, source),
                               full_matrices=False)
            T = np.dot(Vh.T, U.T)

            if not params.reflection:
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

        self._scale = scale = ss * norms[1] / norms[0]
        # Assign projection
        if self.params.scaling:
            proj = scale * T
        else:
            proj = T
        self._proj = proj

        if self._demean:
            self._offset_out = means[1]

    def transform(self,data):
        if self._proj is None:
            raise RuntimeError, "Mapper needs to be train before used."

        # local binding
        demean = self._demean

        d = np.asmatrix(data)

        # Remove input offset if present
        if demean and self._offset_in is not None:
            d = d - self._offset_in

        # Do projection
        res = (d * np.linalg.pinv(self._proj)).A

        # Add output offset if present
        if demean and self._offset_out is not None:
            res += self._offset_out

        return res
