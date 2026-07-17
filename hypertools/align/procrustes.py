#!/usr/bin/env python

# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd

from .common import Aligner, reject_unknown_kwargs
from ..tools.format_data import format_data as formatter


def procrustes(source, target, scaling=True, reflection=True, reduction=False,
               oblique=False, oblique_rcond=-1, format_data=True):
    """
    Function to project from one space to another using Procrustean
    transformation (shift + scaling + rotation + reflection).

    The implementation of this function was based on the ProcrusteanMapper in
    pyMVPA: https://github.com/PyMVPA/PyMVPA

    See also: https://en.wikipedia.org/wiki/Procrustes_transformation

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
    -------
    aligned_source : Numpy array
        The array source is aligned to target and returned

    """
    if format_data:
        source, target = formatter([source, target])

    # fit (single shared implementation: `align`, below) and transform
    proj = align(source, target, scaling=scaling, reflection=reflection,
                 reduction=reduction, oblique=oblique,
                 oblique_rcond=oblique_rcond)
    return np.asarray(source) @ np.asarray(proj)


def align(source, target, scaling=True, reflection=True, reduction=False, oblique=False, oblique_rcond=-1):
    """Compute a Procrustes projection matrix mapping `source` onto `target`.

    Lower-level counterpart to `procrustes()` that works directly on
    (already-formatted) arrays and returns only the projection matrix
    (not the aligned data). Normalizes both inputs by their Frobenius
    norm, pads the lower-dimensional one with zero columns (or raises if
    `reduction=False` and `source` has more columns than `target`),
    solves for the optimal linear map (SVD-based orthogonal solution, or
    a least-squares oblique solution if `oblique=True`), and rescales by
    the norm ratio if `scaling=True`.

    Parameters
    ----------
    source : numpy.ndarray or pandas.DataFrame
        Data to be aligned to `target`'s coordinate system. If it has a
        `.values` attribute (e.g. a DataFrame), that is used.
    target : numpy.ndarray or pandas.DataFrame
        Data defining the target coordinate system.
    scaling : bool, optional
        Estimate a global scaling factor (default: True).
    reflection : bool, optional
        Allow reflections in the (non-oblique) rotation (default: True).
    reduction : bool, optional
        Allow mapping into a lower-dimensional space when `source` has
        more columns than `target` (default: False).
    oblique : bool, optional
        Use a non-orthogonal (least-squares) transformation instead of
        an orthogonal one (default: False).
    oblique_rcond : float, optional
        Cutoff for small singular values, passed to `numpy.linalg.lstsq`
        when `oblique=True` (default: -1).

    Returns
    -------
    numpy.ndarray
        The projection matrix mapping `source` onto `target`, or the
        identity matrix if `source` and `target` are already identical.

    Raises
    ------
    ValueError
        If `source` and `target` do not have the same number of rows, or
        if either dataset is invariant (near-zero variance), or if
        `reduction=False` but `source` has more columns than `target`.
    """
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
        raise ValueError(
            "source and target must have the same number of rows (samples); "
            "got %d (source) vs %d (target). Trim or resample the datasets "
            "to matching lengths before aligning." % (sn, tn))

    # Sums of squares
    ssqs = [np.sum(d ** 2, axis=0) for d in datas]

    for i, which in enumerate(('source', 'target')):
        if np.all(ssqs[i] <= np.abs((np.finfo(datas[i].dtype).eps
                                     * sn) ** 2)):
            raise ValueError(
                "cannot align a dataset with (near-)zero variance: the %s "
                "dataset is constant/invariant across rows. Remove constant "
                "datasets or add variability before aligning." % which)

    norms = [np.sqrt(np.sum(ssq)) for ssq in ssqs]
    normed = [data / norm for (data, norm) in zip(datas, norms)]

    # add new blank dimensions to template space if needed
    if sm < tm:
        normed[0] = np.hstack((normed[0], np.zeros((sn, tm - sm))))

    if sm > tm:
        if reduction:
            normed[1] = np.hstack((normed[1], np.zeros((sn, sm - tm))))
        else:
            raise ValueError(
                "the source dataset has more columns (%d) than the target "
                "(%d), and reduction=False disallows mapping into a "
                "lower-dimensional space. Pass reduction=True to allow it, "
                "or choose a target with at least as many columns as the "
                "source." % (sm, tm))

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
            # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
            # for more and info and original references, see
            # https://doi.org/10.1007/BF02289451
            nsv = len(s)
            s[:-1] = 1
            s[-1] = np.linalg.det(t)
            t = np.dot(u[:, :nsv] * s, vh)

        # figure out scale and final translation
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
    """Apply a fitted Procrustes projection matrix to a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to project; its index is preserved on the result.
    proj : numpy.ndarray or None
        Projection matrix returned by `align`.

    Returns
    -------
    pandas.DataFrame
        `data @ proj`, indexed like `data`.

    Raises
    ------
    RuntimeError
        If `proj` is `None` (mapper has not been fit).
    """
    if proj is None:
        raise RuntimeError("Mapper needs to be trained before use.")

    res = np.asarray(data) @ np.asarray(proj)
    return pd.DataFrame(data=res, index=data.index)


def fitter(data, **kwargs):
    """Fit Procrustes projection matrix/matrices for the `Procrustes` Aligner.

    Parameters
    ----------
    data : DataFrame or list of DataFrame
        Dataset(s) to align. If a list, each is aligned to a common
        `target` (or, when `target` is None, to `data[index]` --
        `data[0]` by default).
    **kwargs
        `target` : DataFrame or None, optional dataset to align to.
        `index` : int, default 0. When `target` is None and `data` is a
        (non-empty) list, `data[index]` is used as the alignment target
        (negative indices follow the usual Python convention). Also
        stored for transformer-side lookups (see `transformer`).
        Remaining kwargs (`scaling`, `reflection`, `reduction`,
        `oblique`, `oblique_rcond`) are forwarded to `align`.

    Returns
    -------
    dict
        `{'index': index, 'proj': proj, ...other kwargs}`, where `proj`
        is a single projection matrix (if `data` is a single dataset) or
        a list of projection matrices (one per dataset in `data`).

    Raises
    ------
    TypeError
        If `index` is not an integer.
    IndexError
        If `index` is out of range for the datasets in `data` (when it
        is used to select the default target).
    """
    target = kwargs.pop('target', None)
    index = kwargs.pop('index', 0)

    if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
        raise TypeError(
            f'index= must be an integer dataset index; got {index!r} '
            f'({type(index).__name__}). Pass the (0-based) position of the '
            'dataset to align the others to, or pass target= explicitly.')

    if isinstance(data, list):
        if len(data) == 0:
            proj = []
        else:
            if target is None:
                if not -len(data) <= index < len(data):
                    raise IndexError(
                        f'index={index} is out of range for {len(data)} '
                        f'dataset(s); pass an index between 0 and '
                        f'{len(data) - 1} (negative indices also work) to '
                        'choose the default alignment target, or pass '
                        'target= explicitly.')
                target = data[index]
            proj = [align(d, target, **kwargs) for d in data]
    elif target is not None:
        proj = align(data, target, **kwargs)
    else:
        proj = align(data, data, **kwargs)

    return dw.core.update_dict(kwargs, {'index': index, 'proj': proj})


def transformer(data, **kwargs):
    """Apply fitted Procrustes projection matrix/matrices for the `Procrustes` Aligner.

    Parameters
    ----------
    data : DataFrame or list of DataFrame
        Dataset(s) to project, using the projection(s) from `fitter`.
    **kwargs
        `proj` : the fitted projection matrix, or list of matrices, from
        `fitter`. `index` : int, used to select which fitted projection
        to apply when `data` is a single (non-list) dataset and `proj`
        is a list.

    Returns
    -------
    DataFrame or list of DataFrame
        The projected dataset(s), matching the list/single-item shape
        implied by `data` and `proj`.

    Raises
    ------
    RuntimeError
        If `proj` is `None` (the model has not been fit).
    ValueError
        If `data` and `proj` are both lists of mismatched length.
    IndexError
        If `index` is outside the range of `proj` (when `proj` is a
        list and `data` is a single dataset).
    """
    proj = kwargs.pop('proj', None)
    if proj is None:
        raise RuntimeError('Need to fit model before transforming data: no '
                           'fitted Procrustes projection found. Call fit '
                           '(or fit_transform) first.')

    if isinstance(proj, list):
        if len(proj) == 0:
            return data
        if isinstance(data, list):
            if len(proj) != len(data):
                raise ValueError(
                    f'cannot transform {len(data)} dataset(s) with '
                    f'{len(proj)} fitted projection(s). Data must either '
                    'be passed in as an individual matrix, or must be of '
                    'the same length as the fitted list of projections.')
            return [xform(d, p) for d, p in zip(data, proj)]
        else:
            index = kwargs.pop('index', 0)
            if not -len(proj) <= index < len(proj):
                raise IndexError(
                    f'index={index} is outside the range of the fitted '
                    f'projection list (length {len(proj)}); pass an index '
                    f'between 0 and {len(proj) - 1} to select which fitted '
                    'projection to apply.')
            return xform(data, proj[index])
    if isinstance(data, list):
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
      aligned to ``data[index]`` (the first DataFrame in the given list by default).
    :param index: Position (0-based) of the dataset used as the default alignment target when ``target`` is not
      given (default: 0). Negative indices follow the usual Python convention.
    """
    def __init__(self, target=None, scaling=True, reflection=True,
                 reduction=False, oblique=False, oblique_rcond=-1, index=0, **kwargs):
        reject_unknown_kwargs('Procrustes', kwargs,
                              ['target', 'scaling', 'reflection', 'reduction',
                               'oblique', 'oblique_rcond', 'index'])
        required = ['proj', 'index']
        super().__init__(required=required, fitter=fitter, transformer=transformer,
                         data=None, target=target, scaling=scaling, reflection=reflection,
                         reduction=reduction, oblique=oblique, oblique_rcond=oblique_rcond,
                         index=index)
