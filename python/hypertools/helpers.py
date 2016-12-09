#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.interpolate import PchipInterpolator as pchip
from sklearn.decomposition import PCA as PCA
import seaborn as sns
import itertools

def center(x):
	x_stacked = np.vstack(x)
	return x - np.mean(x_stacked, 0)

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
	ranks = list(map(lambda rank: int(round(res*rank/len(vals))),ranks))

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

def reduceD(x, ndim):
	#if more than 3d, reduce and re-run
	m = PCA(n_components=ndim, whiten=True)
	m.fit(x)
	return m.transform(x)

def reduceD_list(x, ndim):
	m=PCA(n_components=ndim, whiten=True)
	m.fit(np.vstack(x))

	r=[]
	for i in x:
		r.append(m.transform(i))
	return r
