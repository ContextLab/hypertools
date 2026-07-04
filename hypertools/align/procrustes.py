#!/usr/bin/env python

# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd

from .common import Aligner
from ..tools.format_data import format_data as formatter


def procrustes(source, target, scaling=True, reflection=True, reduction=False,
               oblique=False, oblique_rcond=-1, format_data=True):
    """
    Function to project from one space to another using Procrustean
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

    def fit(source, target):

        datas = (source, target)
        sn, sm = source.shape
        tn, tm = target.shape

        # Check the sizes
        if sn != tn:
            raise ValueError("Data for both spaces should have the same " \
                  "number of samples. Got %d in template and %d in target space" \
                  % (sn, tn))

        # Sums of squares
        ssqs = [np.sum(d**2, axis=0) for d in datas]

        # XXX check for being invariant?
        #     needs to be tuned up properly and not raise but handle
        for i in range(2):
            if np.all(ssqs[i] <= np.abs((np.finfo(datas[i].dtype).eps
                                       * sn )**2)):
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

        # Assign projection
        if scaling:
            scale = ss * norms[1] / norms[0]
            proj = scale * T
        else:
            proj = T
        return proj

    def transform(data, proj):
        if proj is None:
            raise RuntimeError("Mapper needs to be trained before use.")

        d = np.asmatrix(data)

        # Do projection
        res = (d * proj).A

        return res

    if format_data:
        source, target = formatter([source, target])

    # fit and transform
    proj = fit(source, target)
    return transform(source, proj)


def align(source, target, scaling=True, reflection=True, reduction=False, oblique=False, oblique_rcond=-1):
    if hasattr(source, 'values'):
        source = getattr(source, 'values')
    if hasattr(target, 'values'):
        target = getattr(target, 'values')

    datas = (source, target)
    sn, sm = source.shape
    tn, tm = target.shape

    if np.allclose(source.shape, target.shape) and np.allclose(source, target):
        return np.eye(source.shape[1])

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
        if reduction:
            normed[1] = np.hstack((normed[1], np.zeros((sn, sm - tm))))
        else:
            raise ValueError("reduction=False, so mapping from \
                              higher dimensionality \
                              template space is not supported. template space had %d \
                              while target %d dimensions (features)" % (sm, tm))

    source, target = normed
    if oblique:
        # Just do silly linear system of equations ;) or naive
        # inverse problem
        if sn == sm and tm == 1:
            t = np.linalg.solve(source, target)
        else:
            t = np.linalg.lstsq(source, target, rcond=oblique_rcond)[0]
        ss = 1.0
    else:
        # Orthogonal transformation
        # figure out optimal rotation
        u, s, vh = np.linalg.svd(np.dot(target.T, source),
                                 full_matrices=False)
        t = np.dot(vh.T, u.T)

        if not reflection:
            # then we need to assure that it is only rotation
            # "recipe" from
            # http://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
            # for more and info and original references, see
            # http://dx.doi.org/10.1007%2FBF02289451
            nsv = len(s)
            s[:-1] = 1
            s[-1] = np.linalg.det(t)
            t = np.dot(u[:, :nsv] * s, vh)

        # figure out scale and final translation
        # XXX with reflection False -- not sure if here or there or anywhere...
        ss = sum(s)

    # if we were to collect standardized distance
    # std_d = 1 - sD**2

    # select out only relevant dimensions
    if sm != tm:
        t = t[:sm, :tm]

    # Assign projection
    if scaling:
        scale = ss * norms[1] / norms[0]
        proj = scale * t
    else:
        proj = t

    return proj


def xform(data, proj):
    if proj is None:
        raise RuntimeError("Mapper needs to be trained before use.")

    res = np.asarray(data) @ np.asarray(proj)
    return pd.DataFrame(data=res, index=data.index)


def fitter(data, **kwargs):
    target = kwargs.pop('target', None)
    index = kwargs.pop('index', 0)

    if type(data) is list:
        if len(data) == 0:
            proj = []
        else:
            if target is None:
                target = data[0]
            proj = [align(d, target, **kwargs) for d in data]
    elif target is not None:
        proj = align(data, target, **kwargs)
    else:
        proj = align(data, data, **kwargs)

    return dw.core.update_dict(kwargs, {'index': index, 'proj': proj})


def transformer(data, **kwargs):
    proj = kwargs.pop('proj', None)
    assert proj is not None, 'Need to fit model before transforming data'

    if type(proj) is list:
        if len(proj) == 0:
            return data
        if type(data) is list:
            assert len(proj) == len(data), "Data must either be passed in as an individual matrix, or must be of the" \
                                           "same length as the fitted list of projections"
            return [xform(d, p) for d, p in zip(data, proj)]
        else:
            index = kwargs.pop('index', 0)
            assert index < len(proj), IndexError(f'Index {index} is outside the range of list length ({len(proj)}')
            return xform(data, proj[index])
    if type(data) is list:
        return [xform(d, proj) for d in data]
    else:
        return xform(data, proj)


class Procrustes(Aligner):
    """
    Base class for Procrustes objects.  Takes several keyword arguments that specify which transformations are allowed:

    :param scaling: True or False (default: True)
    :param reflection: True or False (default: True)
    :param reduction: True or False (default: False)
    :param oblique: Are oblique transformations allowed?  (default: False)
    :param target: Optional argument for specifying a target dataset to align data to.  If not specified, data are
      aligned to the first DataFrame in the given list.
    """
    def __init__(self, target=None, scaling=True, reflection=True,
                 reduction=False, oblique=False, oblique_rcond=-1, index=0, **kwargs):
        required = ['proj', 'index']
        super().__init__(required=required, fitter=fitter, transformer=transformer,
                         data=None, target=target, scaling=scaling, reflection=reflection,
                         reduction=reduction, oblique=oblique, oblique_rcond=oblique_rcond,
                         index=index, **kwargs)
