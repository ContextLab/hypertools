#!/usr/bin/env python

##PACKAGES##
from __future__ import division
from builtins import zip
from builtins import range
import numpy as np

##MAIN FUNCTION##
def procrustes(source, target, scaling=True, reflection=True, reduction=False,
               oblique=False, oblique_rcond=-1):
    """Function to project from one space to another using Procrustean
    transformation (shift + scaling + rotation + reflection).

    The implementation of this function was based on the ProcrusteanMapper in
    pyMVPA: https://github.com/PyMVPA/PyMVPA

    See also: http://en.wikipedia.org/wiki/Procrustes_transformation

    Parameters
    ----------
    source : Numpy array
        Array to be aligned to target's coordinate system.

    target: Numpy array
        Source is aligned to this target space

    scaling : bool
        Estimate a global scaling factor for the transformation
        (no longer rigid body)

    reflection : bool
        Allow for the data to be reflected (so it might not be
        a rotation. Effective only for non-oblique transformations.

    reduction : bool
        If true, it is allowed to map into lower-dimensional
        space. Forward transformation might be suboptimal then and
        reverse transformation might not recover all original
        variance.

    oblique : bool
        Either to allow non-orthogonal transformation -- might
        heavily overfit the data if there is less samples than
        dimensions. Use `oblique_rcond`.

    oblique_rcond : float
        Cutoff for 'small' singular values to regularize the
        inverse. See :class:`~numpy.linalg.lstsq` for more
        information.

    Returns
    ----------
    aligned_source : Numpy array
        The array source is aligned to target and returned

    """

    _scale = None
    _demean = False

    def fit(source, target):
        # Since it is unsupervised, we don't care about labels
        datas = ()
        means = ()
        shapes = ()

        for i, ds in enumerate((source, target)):
            data = ds
            if _demean:
                if i == 0:
                    mean = _offset_in
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
            pass
            raise ValueError("Data for both spaces should have the same " \
                  "number of samples. Got %d in template and %d in target space" \
                  % (sn, tn))

        # Sums of squares
        ssqs = [np.sum(d**2, axis=0) for d in datas]

        # XXX check for being invariant?
        #     needs to be tuned up properly and not raise but handle
        for i in range(2):
            if np.all(ssqs[i] <= np.abs((np.finfo(datas[i].dtype).eps
                                       * sn * means[i] )**2)):
                raise ValueError("For now do not handle invariant in time datasets")

        norms = [ np.sqrt(np.sum(ssq)) for ssq in ssqs ]
        normed = [ data/norm for (data, norm) in zip(datas, norms) ]

        # add new blank dimensions to template space if needed
        if sm < tm:
            normed[0] = np.hstack( (normed[0], np.zeros((sn, tm-sm))) )

        if sm > tm:
            if reduction:
                normed[1] = np.hstack( (normed[1], np.zeros((sn, sm-tm))) )
            else:
                raise ValueError("reduction=False, so mapping from " \
                      "higher dimensionality " \
                      "template space is not supported. template space had %d " \
                      "while target %d dimensions (features)" % (sm, tm))

        source, target = normed
        if oblique:
            # Just do silly linear system of equations ;) or naive
            # inverse problem
            if sn == sm and tm == 1:
                T = np.linalg.solve(source, target)
            else:
                T = np.linalg.lstsq(source, target, rcond=oblique_rcond)[0]
            ss = 1.0
        else:
            # Orthogonal transformation
            # figure out optimal rotation
            U, s, Vh = np.linalg.svd(np.dot(target.T, source),
                                     full_matrices=False)
            T = np.dot(Vh.T, U.T)

            if not reflection:
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

        _scale = scale = ss * norms[1] / norms[0]
        # Assign projection
        if scaling:
            proj = scale * T
        else:
            proj = T
        return proj

        if _demean:
            _offset_out = means[1]

    def transform(data,proj):
        if proj is None:
            raise RuntimeError("Mapper needs to be train before used.")

        # local binding
        demean = _demean

        d = np.asmatrix(data)

        # Remove input offset if present
        if demean and _offset_in is not None:
            d = d - _offset_in

        # Do projection
        res = (d * proj).A

        # Add output offset if present
        if demean and _offset_out is not None:
            res += _offset_out

        return res

    # Fit and transform
    proj = fit(source, target)
    return transform(source, proj)
