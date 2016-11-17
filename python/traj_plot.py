#!/usr/bin/env python

"""

"""

##PACKAGES##
import numpy as np
import PCA as PCA
from scipy.interpolate import PchipInterpolator as pchip
import plot_coords as plot
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

##META##
__authors__ = ["Jeremy Manning", "Kirsten Ziman"]
__version__ = "1.0.0"
__maintainers__ = ["Jeremy Manning", "Kirsten Ziman"] 
__emails__ = ["Jeremy.R.Manning@dartmouth.edu", "kirstenkmbziman@gmail.com", "contextualdynamics@gmail.com"]
#__copyright__ = ""
#__credits__ = [""]
#__license__ = ""

#NEXT STEPS
#++++++++++++++++
#-smooth to 100 samples per window (p chip to scale up by ten)
#send to plot_coords in "chunks"
#use lazy evaluation to stream

##MAIN FUNCTION##
def traj(x, window):

	##SUB FUNCTIONS##
	def is_list(x):
		if type(x[0][0])==np.ndarray:
			return True
		elif type(x[0][0])==np.int64:
			return False

	def col_match(j):
		sizes_1=np.zeros(len(j))
		for x in range(0, len(j)):
			sizes_1[x]=j[x].shape[1]

		if len(np.unique(sizes_1)) == 1:
			return True
		else:
			return False

##MAIN FUNCTION##
	if is_list(x) and x[0].shape[-1]>3 or not is_list(x) and x.shape[-1]>3:
		data=PCA.reduce(x, 3)

	if not col_match(x):
		print "Inputted arrays must have the same number of columns"

	else: data=x

	new=np.zeros(data.shape)

	if not is_list(data):
		#n=data.shape[0]
		#for col in range(0, data.shape[1]):
		#	new[:,col]=scipy.interpolate.pchip_interploate(np.arange(0, n, 1), data[:,col], np.arange(0, n, .1))

		chunks=len(data/w) 
		extra=(data % w) 


		for z in range(0, (len(data)/w)):
			plot()





