#!/usr/bin/env python

from .._externals.srm import SRM
from .procrustes import procrustes
import numpy as np
from .format_data import format_data as formatter
import warnings

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

    # if model is None, just return data
    if align is None:
        return data
    else:
        if isinstance(align, dict):
            params = dict(align.get('params', {}))
            align = align['model']
            if align is None:
                return data
            n_iter = params.get('n_iter', n_iter)

        if align is True:
            # retired in 2.0 (previously deprecated): boolean form was
            # ambiguous -- require an explicit algorithm name
            raise ValueError("align=True was removed in hypertools 2.0; "
                             "specify the algorithm instead, e.g. "
                             "align='hyper' or align='SRM'.")

        # common format
        if format_data:
            data = formatter(data, ppca=True)

        if len(data) == 1:
            warnings.warn('Data in list of length 1 can not be aligned. '
                 'Skipping the alignment.')

        if data[0].shape[1] >= data[0].shape[0]:
            warnings.warn('The number of features exceeds number of samples. This can lead \
                 to overfitting.  We recommend reducing the dimensionality to be \
                 less than the number of samples prior to hyperalignment.')

        if align == 'hyper':

            ##STEP 0: STANDARDIZE SIZE AND SHAPE##
            sizes_0 = [x.shape[0] for x in data]
            sizes_1 = [x.shape[1] for x in data]

            #find the smallest number of rows
            R = min(sizes_0)
            C = max(sizes_1)

            m = [np.empty((R,C), dtype=np.ndarray)] * len(data)

            for idx,x in enumerate(data):
                y = x[0:R,:]
                missing = C - y.shape[1]
                add = np.zeros((y.shape[0], missing))
                y = np.append(y, add, axis=1)
                m[idx]=y

            # REPEATED APPLICATION OF HYPERALIGNMENT (n_iter passes): each
            # pass runs the full classic procedure -- build a sequential
            # template, refine it, align every dataset to it -- and the
            # ALIGNED datasets become the input to the next pass, so
            # convergence toward the common space compounds across passes.
            # procrustes' optimal scaling factor is < 1 whenever alignment
            # is imperfect, so repeated passes geometrically shrink the
            # data; rescale after each pass to preserve the original scale
            # (relative scales within a pass are preserved)
            aligned = m
            orig_norm = np.mean([np.linalg.norm(x) for x in m])
            for _ in range(max(1, int(n_iter))):
                aligned = _hyperalign_pass(aligned)
                cur_norm = np.mean([np.linalg.norm(np.asarray(a))
                                    for a in aligned])
                if cur_norm > 0:
                    aligned = [np.asarray(a) * (orig_norm / cur_norm)
                               for a in aligned]
            return aligned

        elif align == 'SRM':
            # n_iter repeated applications, mirroring the classic
            # smooth-and-align recipe (each pass re-fits SRM on the
            # previous pass's aligned output)
            aligned = data
            for _ in range(max(1, int(n_iter))):
                transposed = [np.asarray(i).T for i in aligned]
                srm = SRM(features=np.min([i.shape[0]
                                           for i in transposed]))
                srm.fit(transposed)
                aligned = [i.T for i in srm.transform(transposed)]
            return aligned

def _hyperalign_pass(m):
    """One full pass of classic hyperalignment (Haxby et al., 2011):
    sequentially build a template, refine it by aligning every dataset to
    it, then align all datasets to the refined template.

    The template is rescaled to the datasets' mean Frobenius norm at each
    step: procrustes maps datasets onto the template's scale, and averaging
    shrinks the template, so without rescaling repeated passes collapse all
    datasets geometrically toward zero (eventually tripping procrustes'
    invariant-data check)."""
    mean_norm = np.mean([np.linalg.norm(x) for x in m])

    def rescale(t):
        norm = np.linalg.norm(t)
        return t * (mean_norm / norm) if norm > 0 else t

    ##STEP 1: INITIAL TEMPLATE##
    for x in range(0, len(m)):
        if x == 0:
            template = np.copy(m[x])
        else:
            template += procrustes(m[x], template / (x + 1))
    template = rescale(template / len(m))

    ##STEP 2: REFINED TEMPLATE##
    template2 = np.zeros(template.shape)
    for x in range(0, len(m)):
        template2 += procrustes(m[x], template)
    template2 = rescale(template2 / len(m))

    ##STEP 3: ALIGN TO REFINED TEMPLATE##
    return [procrustes(m[x], template2) for x in range(0, len(m))]
