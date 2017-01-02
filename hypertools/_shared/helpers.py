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
import pandas as pd

##HELPER FUNCTIONS##
def center(x):
    assert type(x) is list, "Input data to center must be list"
    x_stacked = np.vstack(x)
    return [i - np.mean(x_stacked, 0) for i in x]

def normalize(x):
    assert type(x) is list, "Input data to center must be list"
    x_stacked = np.vstack(x)
    m1 = np.min(x_stacked)
    m2 = np.max(x_stacked - m1)    
    f = lambda(x): 2*((x - m1) / m2) - 1
    return [f(i) for i in x]

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
	palette = np.array(sns.color_palette(cmap, res))
	ranks = np.digitize(vals, np.linspace(np.min(vals), np.max(vals)+1, res+1)) - 1
	return [tuple(i) for i in palette[ranks, :]]

def vals2bins(vals,res=100):
	"""Maps values to bins
	Args:
	values (list or list of lists) - list of values to map to colors
	res (int) - resolution of the color map (default: 100)
	Returns:
	list of numbers representing bins
	"""
	# flatten if list of lists
	if any(isinstance(el, list) for el in vals):
		vals = list(itertools.chain(*vals))
	return list(np.digitize(vals, np.linspace(np.min(vals), np.max(vals)+1, res+1)) - 1)

def interp_array(arr,interp_val=10):
    x=np.arange(0, len(arr), 1)
    xx=np.arange(0, len(arr)-1, 1/interp_val)
    q=pchip(x,arr)
    return q(xx)

def interp_array_list(arr_list,interp_val=10):
    smoothed= [np.zeros(arr_list[0].shape) for item in arr_list]
    for idx,arr in enumerate(arr_list):
        smoothed[idx] = interp_array(arr,interp_val)
    return smoothed

def check_data(data):
	if type(data) is list:
		assert all([data[0].shape[1]==x.shape[1] for x in data]), 'Arrays must have the same shape.'
		return 'list'
	elif isinstance(data, np.ndarray):
		return 'array'
	elif isinstance(data, pd.DataFrame):
		return 'df'
	else:
		raise ValueError("Data must be numpy array, list of numpy array or pandas dataframe.")

def parse_args(x,args):
	args_list = []
	for i,item in enumerate(x):
		tmp = []
		for ii,arg in enumerate(args):
			if type(arg) is tuple or type(arg) is list:
				if len(arg) == len(x):
					tmp.append(arg[i])
				else:
					print('Error: arguments must be a list of the same length as x')
					sys.exit(1)
			else:
				tmp.append(arg)
		args_list.append(tuple(tmp))
	return args_list

def parse_kwargs(x,kwargs):
	kwargs_list = []
	for i,item in enumerate(x):
		tmp = {}
		for kwarg in kwargs:
			if type(kwargs[kwarg]) is tuple or type(kwargs[kwarg]) is list:
				if len(kwargs[kwarg]) == len(x):
					tmp[kwarg]=kwargs[kwarg][i]
				else:
					print('Error: keyword arguments must be a list of the same length as x')
					sys.exit(1)
			else:
				tmp[kwarg]=kwargs[kwarg]
		kwargs_list.append(tmp)
	return kwargs_list

def reshape_data(x,labels):
	categories = list(set(np.sort(labels)))
	x_stacked = np.vstack(x)
	x_reshaped = [[] for i in categories]
	for idx,point in enumerate(labels):
		x_reshaped[categories.index(point)].append(x_stacked[idx])
	return [np.vstack(i) for i in x_reshaped]

def pandas_to_list(data):
	df_str = data.select_dtypes(include=['object'])
	df_num = data.select_dtypes(exclude=['object'])
	for colname in df_str.columns:
		df_num = df_num.join(pd.get_dummies(data[colname], prefix=colname))
	plot_data = df_num.as_matrix()
	return plot_data
