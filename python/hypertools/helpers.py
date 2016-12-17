#!/usr/bin/env python

"""
Helper functions
"""

##PACKAGES##
from __future__ import division
import numpy as np
from scipy.interpolate import PchipInterpolator as pchip
import seaborn as sns
import itertools

##HELPER FUNCTIONS##
def center(x):
	x_stacked = np.vstack(x)
	return x - np.mean(x_stacked, 0)

def group_by_category(vals):
	if any(isinstance(el, list) for el in vals):
		vals = list(itertools.chain(*vals))
	val_set = list(set(vals))
	return [val_set.index(val) for val in vals]

def vals2colors(vals,cmap='GnBu_d',res=100):
	"""Maps values to colors
	Args:
	values (list or list of lists) - list of values to map to colors
	cmap (str) - color map (default is 'husl')
	res (int) - resolution of the color map (default: 100)
	Returns:
	list of rgb tuples
	"""
	# flatten if list of lists
	if any(isinstance(el, list) for el in vals):
		vals = list(itertools.chain(*vals))

	# get palette from seaborn
	palette = sns.color_palette(cmap, res)

	# rank the values and then normalize
	ranks = list(map(lambda x: sum([val <= x for val in vals]),vals))
	ranks = list(map(lambda rank: int(res*rank/len(vals)),ranks))
	return [palette[rank-1] for rank in ranks]

# this will be moved to utils.py
def is_list(x):
    if type(x[0][0])==np.ndarray:
        return True
    elif type(x[0][0])==np.int64 or type(x[0][0])==int or type(x[0][0])==np.float32:
        return False

# #  this will be moved to utils.py
def interp_array(arr,interp_val=10):
    x=np.arange(0, len(arr), 1)
    xx=np.arange(0, len(arr)-1, 1/interp_val)
    q=pchip(x,arr)
    return q(xx)

# #  this will be moved to utils.py
def interp_array_list(arr_list,interp_val=10):
    smoothed= [np.zeros(arr_list[0].shape) for item in arr_list]
    for idx,arr in enumerate(arr_list):
        smoothed[idx] = interp_array(arr,interp_val)
    return smoothed

def check_data(data):
    assert all([data[0].shape[1]==x.shape[1] for x in data]), 'Arrays must have the same shape.'

    ##FUNCTIONS##
def is_list(x):
	if type(x[0][0])==np.ndarray:
		return True
	elif type(x[0][0])==np.int64:
		return False

def _getAplus(A):
	eigval, eigvec = np.linalg.eig(A)
	Q = np.matrix(eigvec)
	xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
	return Q*xdiag*Q.T

def _getPs(A, W=None):
	W05 = np.matrix(W**.5)
	return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
	Aret = np.array(A.copy())
	Aret[W > 0] = np.array(W)[W > 0]
	return np.matrix(Aret)

def nearPD(A, nit=10):
	n = A.shape[0]
	W = np.identity(n)
	# W is the matrix used for the norm (assumed to be Identity matrix here)
	# the algorithm should work for any diagonal W
	deltaS = 0
	Yk = A.copy()
	for k in range(nit):
		Rk = Yk - deltaS
		Xk = _getPs(Rk, W=W)
		deltaS = Xk - Rk
		Yk = _getPu(Xk, W=W)
	return Yk

def is_pos_def(x):
	return np.all(np.linalg.eig(x)>0)

def make_pos_def(x):
	if is_pos_def(x):
		return x
	else:
		return nearPD(x)
