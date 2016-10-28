#!/usr/bin/env python

"""
input: TxD matrix of observations
		   T-number of coords
		   D-dimensionality of each observation
output: Tx3 matrix of observations
		   reduced via PCA
"""

##PACKAGES###
import numpy as np 
from sklearn.decomposition import PCA

##META##
__authors__ = ["Jeremy Manning", "Kirsten Ziman"]
__version__ = "1.0.0"
__maintainers__ = ["Jeremy Manning", "Kirsten Ziman"] 
__emails__ = ["Jeremy.R.Manning@dartmouth.edu", "kirstenkmbziman@gmail.com", "contextualdynamics@gmail.com"]
#__copyright__ = ""
#__license__ = ""  

##FUNCTION##
def reduc(x, ndim):

##SUB FUNCTIONS##
	def is_list(x):
		if type(x[0][0])==np.ndarray:
			return True
		elif type(x[0][0])==np.int64:
			return False

	def reduceD(x, ndim):	
		#if more than 3d, reduce and re-run
		m = PCA(n_components=ndim, whiten=True)
		m.fit(x)
		return m.transform(x)

	def reduceD_list(x, ndim):
		m=PCA(n_components=ndim, whiten=True)
		m.fit(x[0])

		r=[]
		for i in x:
			r.append(m.transform(i))
		return r

##FUNCTION BODY##
	if is_list(x):
		return reduceD_list(x, 3)
	else: 
		return reduceD(x, 3)