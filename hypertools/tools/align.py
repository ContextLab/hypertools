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
from .._externals.srm import SRM
from .procrustes import procrustes
import numpy as np
from .._shared.helpers import format_data

##MAIN FUNCTION##
def align(data, method='HYPER'):
	"""Implements hyperalignment"""

	data = format_data(data)

	if method=='HYPER':

		##STEP 0: STANDARDIZE SIZE AND SHAPE##
		sizes_0 = map(lambda x: x.shape[0], data)
		sizes_1 = map(lambda x: x.shape[1], data)

		#find the smallest number of rows
		R = min(sizes_0)
		C = max([3, max(sizes_1)])

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
