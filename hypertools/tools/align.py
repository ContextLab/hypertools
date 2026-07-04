#!/usr/bin/env python
"""Classic array/mode ``align`` API (HyperTools <2.0).

Thin compatibility wrapper over the new class-based dispatcher in
:mod:`hypertools.align.align`. The classic string/dict ``align`` argument is
translated to a dispatcher ``model`` name, ``n_iter`` is threaded into
hyperalignment, and (when ``format_data=True``) the input -- a DataGeometry,
text, arrays, or a mix -- is first funneled through
:func:`hypertools.tools.format_data.format_data` into a list of numpy arrays,
mirroring dev-2.0's ``format_data=True`` behavior. Output is a list of numpy
arrays, as the classic API promised.
"""
import numpy as np

from ..align.align import align as _align_dispatch
from .format_data import format_data as formatter

_ALIAS = {'hyper': 'HyperAlign', 'HyperAlign': 'HyperAlign',
          'SRM': 'SharedResponseModel',
          'SharedResponseModel': 'SharedResponseModel',
          'DetSRM': 'DeterministicSharedResponseModel',
          'DeterministicSharedResponseModel': 'DeterministicSharedResponseModel',
          'Procrustes': 'Procrustes', 'NullAlign': 'NullAlign'}


def align(data, align='hyper', n_iter=10, format_data=True):
    """
    Aligns a list of arrays

    This function takes a list of high dimensional arrays and 'hyperaligns' them
    to a 'common' space, or coordinate system following the approach outlined by
    Haxby et al, 2011. Hyperalignment uses linear transformations (rotation,
    reflection, translation, scaling) to register a group of arrays to a common
    space. This can be useful when two or more datasets describe an identical
    or similar system, but may not be in same coordinate system. For example,
    consider the example of fMRI recordings (voxels by time) from the visual
    cortex of a group of subjects watching the same movie: The brain responses
    should be highly similar, but the coordinates may not be aligned.

    Haxby JV, Guntupalli JS, Connolly AC, Halchenko YO, Conroy BR, Gobbini
    MI, Hanke M, and Ramadge PJ (2011)  A common, high-dimensional model of
    the representational space in human ventral temporal cortex.  Neuron 72,
    404 -- 416. (used to implement hyperalignment, see https://github.com/PyMVPA/PyMVPA)

    Brain Imaging Analysis Kit, http://brainiak.org. (used to implement Shared Response Model [SRM], see https://github.com/IntelPNI/brainiak)

    Parameters
    ----------
    data : numpy array, pandas df, or list of arrays/dfs
        A list of Numpy arrays or Pandas Dataframes

    align : str or dict
        If str, either 'hyper' or 'SRM'.  If 'hyper', alignment algorithm will be
        hyperalignment. If 'SRM', alignment algorithm will be shared response
        model.  You can also pass a dictionary for finer control, where the 'model'
        key is a string that specifies the model and the params key is a dictionary
        of parameter values (default : 'hyper').

    n_iter : int
        Number of hyperalignment iterations: the common template is
        re-estimated from the aligned data and all datasets are re-aligned
        to it, repeatedly. More iterations give a more stable common space
        (default: 10). Only used when align='hyper'; may also be passed via
        the dict form, e.g. align={'model': 'hyper',
        'params': {'n_iter': 10}}.

    format_data : bool
        Whether or not to first call the format_data function (default: True).

    Returns
    ----------
    aligned : list
        An aligned list of numpy arrays

    """
    # if model is None, just return data unchanged
    if align is None:
        return data
    if align is True:
        # retired in 2.0 (previously deprecated): boolean form was ambiguous --
        # require an explicit algorithm name
        raise ValueError("align=True was removed in hypertools 2.0; specify the "
                         "algorithm instead, e.g. align='hyper' or align='SRM'.")

    if isinstance(align, dict):
        params = dict(align.get('params', {}))
        model = align['model']
        if model is None:
            return data
        n_iter = params.get('n_iter', n_iter)
    else:
        model, params = align, {}

    model = _ALIAS.get(model, model)
    if model == 'HyperAlign':
        params.setdefault('n_iter', n_iter)

    # funnel any classic input (geo / text / arrays / mixed) into a list of
    # numpy arrays before handing off to the (funnel-wrapped) dispatcher
    if format_data:
        data = formatter(data, ppca=True)

    out = _align_dispatch(data, model=model, **params)
    if not isinstance(out, list):
        out = [out]
    return [np.asarray(o) for o in out]
