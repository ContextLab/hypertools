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
import numpy as np
import numpy as np,numpy.linalg
from scipy.spatial import procrustes

##MAIN FUNCTION##
def align(data):
	"""Implements hyperalignment"""

	assert all(isinstance(i, np.ndarray) for i in data) and type(data) is list and len(data)>1, "Input must be list of arrays"

	##STEP 0: STANDARDIZE SIZE AND SHAPE##
	sizes_0=np.zeros(len(data))
	sizes_1=np.zeros(len(data))

	for x in range(0, len(data)):
		sizes_0[x]=data[x].shape[0]
		sizes_1[x]=data[x].shape[1]

	#find the smallest number of rows
	R=min(sizes_0)

	#find max columns; if max columns less than 3, add cols of zeros to make 3
	if max(sizes_1) < 3:
		C=3
	else:
		C=max(sizes_1)

	k=np.empty((R,C), dtype=np.ndarray)
	m=[k]*len(data)

	for idx,x in enumerate(data):
		y=x[0:R,:]
		missing=C-y.shape[1]
		add=np.zeros((y.shape[0], missing))
		y=np.append(y, add, axis=1)
		m[idx]=y

	##STEP 1: TEMPLATE##

	for x in range(0, len(m)):
		if x==0:
			template=m[x]
		else:
			_,next,_ = procrustes(np.transpose(template/x), np.transpose(m[x]))
			template = template + np.transpose(next)
	template= template/len(m)

	##STEP 2: NEW COMMON TEMPLATE##
	#align each subj to the template from STEP 1

	template2= np.zeros(template.shape)
	for x in range(0, len(m)):
		_,next,_ = procrustes(np.transpose(template),np.transpose(m[x]))
		template2 = template2 + np.transpose(next)
	template2=template2/len(m)

	#STEP 3 (below): ALIGN TO NEW TEMPLATE

	empty= np.zeros(template2.shape)
	aligned=[empty]*(len(m))
	for x in range(0, len(m)):
		_,next,_ = procrustes(np.transpose(template2),np.transpose(m[x]))
		aligned[x] = np.transpose(next)
	return aligned
