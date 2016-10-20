#!/usr/bin/env python

"""
inputs: TxD matrix of observations
		   T-number of coords
		   D-dimensionality of each observation
		   *Nans treated as missing observations
		type (specify the type of plot)
		   see http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.plot for available options
outputs: 1-, 2-, or 3-dimensional representation of the data

		to edit color map, change both instances of cm.plasma to cm.DesiredColorMap
"""

##PACKAGES##
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

##META##
__authors__ = ["Jeremy Manning", "Kirsten Ziman"]
__version__ = "1.0.0"
__maintainers__ = ["Jeremy Manning", "Kirsten Ziman"] 
__emails__ = ["Jeremy.R.Manning@dartmouth.edu", "kirstenkmbziman@gmail.com", "contextualdynamics@gmail.com"]
#__copyright__ = ""
#__credits__ = [""]
#__license__ = ""  

##MAIN FUNCTION##
def plot_coords(x, *args, **kwargs):
	"""
	implements plotting
	"""

	##SUB FUNCTIONS##
	def is_list(x):
		if type(x[0][0])==np.ndarray:
			return True
		elif type(x[0][0])==np.int64:
			return False

	def col_match(k):
		sizes_1=np.zeros(len(j[0]))
		for x in range(0, len(j[0])):
			sizes_1[x]=j[0][x].shape[1]

		if len(np.unique(sizes_1)) == 1:
			return True
		else:
			return False

	#def resize(k):
	#	sizes_1=np.zeros(len(k))

	#	for x in range(0, len(k)):
	#		sizes_1[x]=k[x].shape[1]

	#	C=max(sizes_1)
		#find largest # of columns from all inputted arrays

	#	m=[]
	#	for idx,x in enumerate(k):			
	#		missing=C-x.shape[1]
	#		add=np.zeros((x.shape[0], missing))
	#		y=np.append(x, add, axis=1)

	#		m.append(y)
	#	return m

	def dispatch(x):
		#determine how many dimensions (number of columns)
		if x.shape[-1]==1:
			plot1D(x)
		elif x.shape[-1]==2:
			plot2D(x)
		elif x.shape[-1]==3:
			plot3D(x)
		elif x.shape[-1]>3:
			plot3D(reduceD(x, 3))

	def dispatch_list(x):
		if x[0].shape[-1]==1:
			plot1D_list(x)
		elif x[0].shape[-1]==2:
			plot2D_list(x)
		elif x[0].shape[-1]==3:
			plot3D_list(x)
		elif x[0].shape[-1]>3:
			plot3D_list(reduceD_list(x, 3))

	def plot1D(data):
		x=np.arange(len(data)).reshape((len(data),1))
		plot2D(np.hstack((x, data)))

	def plot1D_list(data):
		x=[]
		for i in range(0, len(data)):
			x.append(np.arange(len(data[i])).reshape(len(data[i]),1))
		plot_1to2_list(np.hstack((x, data)))

	def plot2D(data):
		# type: (object) -> object
		#if 2d, make a scatter
		plt.plot(data[:,0], data[:,1], *args, **kwargs)

	def plot_1to2_list(data):
		n=len(data)
		color=iter(cm.plasma(np.linspace(0,1,n)))
		fig, ax = plt.subplots()
		for i in range(n):
			c=next(color)
			m=len(data[i])
			half=m/2
			ax.plot(data[i][0:half,0], data[i][half:m+1,0], c=c)

	def plot2D_list(data):
		# type: (object) -> object
		#if 2d, make a scatter
		n=len(data)
		color=iter(cm.plasma(np.linspace(0,1,n)))
		fig, ax = plt.subplots()
		for i in range(n):
			c=next(color)
			ax.plot(data[i][:,0], data[i][:,1], c=c, *args, **kwargs)

	def plot3D(data):
		#if 3d, make a 3d scatter
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(data[:,0], data[:,1], data[:,2], *args, **kwargs)

	def plot3D_list(data):
		#if 3d, make a 3d scatter
		n=len(data)
		color=iter(cm.plasma(np.linspace(0,1,n)))
		fig = plt.figure() 
		ax = fig.add_subplot(111, projection='3d')
		for i in range(n):
			c=next(color)
			ax.plot(data[i][:,0], data[i][:,1], data[i][:,2], c=c, *args, **kwargs)

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

	##MAIN FUNCTION##
	if is_list(x):
		#dispatch_list(resize(x))
		if col_match(x):
			dispatch_list(x)
			plt.show()
		else:
			print "Inputted arrays must have the same number of columns"

	else:
		dispatch(x)
		plt.show()
	

