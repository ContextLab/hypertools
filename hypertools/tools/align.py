#!/usr/bin/env python

"""
Implements the "hyperalignment" algorithm described by the
following paper:

Haxby JV, Guntupalli JS, Connolly AC, Halchenko YO, Conroy BR, Gobbini
MI, Hanke M, and Ramadge PJ (2011)  A common, high-dimensional model of
the representational space in human ventral temporal cortex.  Neuron 72,
404 -- 416.

INPUTS:
-numpy array(s)
-list of numpy arrays

OUTPUTS:
-numpy array
-list of aligned numpy arrays
"""

##PACKAGES##
from __future__ import division
from builtins import range
from .._externals.srm import SRM
from .procrustes import procrustes
import numpy as np
from .._shared.helpers import format_data
from .normalize import normalize as normalizer
from warnings import warn

##MAIN FUNCTION##
def align(data, method='hyper', normalize=False, ndims=None):
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

    method : str
        Either 'hyper' or 'SRM'.  If 'hyper', alignment algorithm will be
        hyperalignment. If 'SRM', alignment algorithm will be shared response
        model (default : 'hyper')

    normalize : str or False
        If set to 'across', the columns of the input data will be z-scored
        across lists. If set to 'within', the columns will be
        z-scored within each list that is passed. If set to 'row', each row of
        the input data will be z-scored. If set to False, the input data will
        be returned (default is False).

    ndims : int
        Number of dimensions to reduce the dataset to *prior* to alignment

    Returns
    ----------
    aligned : list
        An aligned list of numpy arrays

    """

    data = format_data(data)

    if data[0].shape[1]>=data[0].shape[0]:
        warn('The number of features exceeds number of samples. This can lead \
             to overfitting.  We recommend reducing the dimensionality to be \
             less than the number of samples prior to hyperalignment.')

    # normalize data
    if normalize:
        x = normalizer(x, normalize=normalize)

    # reduce if ndims is specified
    if ndims is not None:
        # Import is here to avoid circular imports with align.py
        from .reduce import reduce as reducer
        data = reducer(data, ndims, internal=True)

    if method=='hyper':

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

    elif method=='SRM':
        data = [i.T for i in data]
        srm = SRM(features=np.min([i.shape[0] for i in data]))
        fit = srm.fit(data)
        return [i.T for i in srm.transform(data)]
