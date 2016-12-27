#!/usr/bin/env python

##PACKAGES##
import numpy as np

def missing_inds(x):
	if type(x) is not list:
		x = [x]
	inds = [[idx for idx,row in enumerate(arr) if any(np.isnan(row))] for arr in x]
	if len(inds)>1:
		return inds
	else:
		return inds[0]
