#!/usr/bin/env python

##PACKAGES##
from __future__ import division
from builtins import range
from .._externals.srm import SRM
from .procrustes import procrustes
import numpy as np
from .._shared.helpers import format_data, memoize
from .normalize import normalize as normalizer
import warnings

@memoize
def align(data, align='hyper', method=None):
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
    data : list
        A list of Numpy arrays or Pandas Dataframes

    align : str or dict
        If str, either 'hyper' or 'SRM'.  If 'hyper', alignment algorithm will be
        hyperalignment. If 'SRM', alignment algorithm will be shared response
        model.  You can also pass a dictionary for finer control, where the 'model'
        key is a string that specifies the model and the params key is a dictionary
        of parameter values (default : 'hyper').

    Returns
    ----------
    aligned : list
        An aligned list of numpy arrays

    """

    # if model is None, just return data
    if align is None:
        return data
    else:
        if method is not None:
            warnings.warn('The method arg is deprecated.  Please use align.')
            align = method

        # common format
        data = format_data(data)

        if len(data) is 1:
            warn('Data in list of length 1 can not be aligned. '
                 'Skipping the alignment.')

        if data[0].shape[1]>=data[0].shape[0]:
            warnings.warn('The number of features exceeds number of samples. This can lead \
                 to overfitting.  We recommend reducing the dimensionality to be \
                 less than the number of samples prior to hyperalignment.')

        if (align is 'hyper') or (method is 'hyper'):

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

            ##STEP 1: TEMPLATE##
            for x in range(0, len(m)):
                if x==0:
                    template = np.copy(m[x])
                else:
                    next = procrustes(m[x], template / (x + 1))
                    template += next
            template /= len(m)

            ##STEP 2: NEW COMMON TEMPLATE##
            #align each subj to the template from STEP 1
            template2 = np.zeros(template.shape)
            for x in range(0, len(m)):
                next = procrustes(m[x], template)
                template2 += next
            template2 /= len(m)

            #STEP 3 (below): ALIGN TO NEW TEMPLATE
            aligned = [np.zeros(template2.shape)] * len(m)
            for x in range(0, len(m)):
                next = procrustes(m[x], template2)
                aligned[x] = next
            return aligned

        elif (align is 'SRM') or (method is 'SRM'):
            data = [i.T for i in data]
            srm = SRM(features=np.min([i.shape[0] for i in data]))
            fit = srm.fit(data)
            return [i.T for i in srm.transform(data)]
