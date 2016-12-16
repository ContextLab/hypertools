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
from srm import SRM
from procrustean import Procrustes
import numpy as np

##MAIN FUNCTION##
def align(data, method='HYP'):
	"""Implements hyperalignment"""

	assert all(isinstance(i, np.ndarray) for i in data) and type(data) is list and len(data)>1, "Input must be list of arrays"

	if method=='HYP':
		for item in data:
			pc = Procrustes()
			pc._train(w1[0],w1[1].T)




	elif method=='SRM':
		data = [i.T for i in data]
		srm = SRM(features=np.min([i.shape[0] for i in data]))
		fit = srm.fit(data)
		return [i.T for i in srm.transform(data)]
