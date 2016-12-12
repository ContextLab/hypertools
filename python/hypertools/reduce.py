#!/usr/bin/env python

"""
Implements PCA (wrapper for scikit-learn.decomposition.PCA)

INPUTS:
-numpy array(s)
-list of numpy arrays

OUTPUTS:
-numpy array (or list of arrays) with dimensions reduced
"""

##PACKAGES##
from .helpers import reduceD

##MAIN FUNCTION##
def reduce(arr,ndims=3):
    if type(arr) is list:
        return reduceD(arr,ndims)
    else:
        return reduceD(srr,ndims)
