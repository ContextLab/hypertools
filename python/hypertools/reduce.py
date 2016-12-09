#!/usr/bin/env python

"""
Implements PCA (wrapper for scikit-learn.decomposition.PCA)

INPUTS:
-numpy array(s)
-list of numpy arrays

OUTPUTS:
-numpy array (or list of arrays) with dimensions reduced
"""

from .helpers import reduceD, reduceD_list

def reduce(arr,ndims=3):
    if type(arr) is list:
        return reduceD_list(arr,ndims)
    else:
        return reduceD(srr,ndims)
