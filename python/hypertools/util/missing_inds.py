#!/usr/bin/env python

##PACKAGES##
import numpy as np

def missing_inds(x):
	if type(x) is not list:
		x = [x]
	inds = [[(i,j) for i in range(arr.shape[0]) for j in range(arr.shape[1]) if np.isnan(arr[i,j])] for arr in x]
	if len(inds)>1:
		return inds
	else:
		return inds[0]
