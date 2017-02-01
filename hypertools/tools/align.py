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
def align(data, method='hyper'):
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

	Parameters
	----------
	data : list
			A list of Numpy arrays or Pandas Dataframes

	method : str
			Either 'hyper' or 'SRM'.  If 'hyper' (default),

	Returns
	----------
	aligned : list
			An aligned list of numpy arrays

	"""

	data = format_data(data)

	if method=='hyper':

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
