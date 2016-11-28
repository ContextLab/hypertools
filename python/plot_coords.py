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
import sys
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns

##MAIN FUNCTION##
def plot_coords(x, *args, **kwargs):
	"""
	implements plotting
	"""

	##STYLING##
	if 'style' in kwargs:
		sns.set(style=kwargs['style'])
		del kwargs['style']
	else:
		sns.set(style="whitegrid")

	if 'palette' in kwargs:
		sns.set_palette(palette=kwargs['palette'], n_colors=len(x))
		del kwargs['palette']
	else:
		sns.set_palette(palette="hls", n_colors=len(x))

	##PARSE PLOT_COORDS SPECIFIC ARGS##
	if 'ndims' in kwargs:
		ndims=kwargs['ndims']
		del kwargs['ndims']

	if 'labels' in kwargs:
		labels=kwargs['labels']
		del kwargs['labels']
	else:
		labels=False

	if 'explore' in kwargs:
		kwargs['picker']=True
		del kwargs['explore']
		explore=True
	else:
		explore=False

	##PARSE ARGS##
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

	##PARSE KWARGS##
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

	def distance(point, event):
		"""Return distance between mouse position and given data point

		Args:
			point (np.array): np.array of shape (3,), with x,y,z in data coords
			event (MouseEvent): mouse event (which contains mouse position in .x and .xdata)
		Returns:
			distance (np.float64): distance (in screen coords) between mouse pos and data point
		"""
		assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape

		# Project 3d data space to 2d data space
		x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
		# Convert 2d data space to 2d screen space
		x3, y3 = ax.transData.transform((x2, y2))

		return np.sqrt ((x3 - event.x)**2 + (y3 - event.y)**2)

	def calcClosestDatapoint(X, event):
		""""Calculate which data point is closest to the mouse position.

		Args:
			X (np.array) - array of points, of shape (numPoints, 3)
			event (MouseEvent) - mouse event (containing mouse position)
		Returns:
			smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
		"""
		distances = [distance (X[i, 0:3], event) for i in range(X.shape[0])]
		return np.argmin(distances)

	global labels_and_points
	labels_and_points = []

	def annotate_plot(X, labels):
		"""Create popover label in 3d chart

		Args:
			X (np.array) - array of points, of shape (numPoints, 3)
			index (int) - index (into points array X) of item which should be printed
		Returns:
			None
		"""
		proj = ax.get_proj()
		for idx,x in enumerate(X):
			if labels[idx] is not None:
				x2, y2, _ = proj3d.proj_transform(x[0], x[1], x[2], proj)
				label = plt.annotate(
				labels[idx],
				xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
				bbox = dict(boxstyle = 'round,pad=0.5', fc = 'gray', alpha = 0.5),
				arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
				labels_and_points.append((label,x[0],x[1],x[2]))
		fig.canvas.draw()

	def update_position(e):
		proj = ax.get_proj()
		for label, x, y, z in labels_and_points:
			x2, y2, _ = proj3d.proj_transform(x, y, z, proj)
			label.xy = x2,y2
			label.update_positions(fig.canvas.renderer)
		fig.canvas.draw()

	def annotate_plot_explore(X, index, labels=False):
		"""Create popover label in 3d chart

		Args:
			X (np.array) - array of points, of shape (numPoints, 3)
			index (int) - index (into points array X) of item which should be printed
		Returns:
			None
		"""
		# save clicked points
		if not hasattr(annotate_plot_explore, 'clicked'):
			annotate_plot_explore.clicked = []

		# If we have previously displayed another label, remove it first
		if hasattr(annotate_plot_explore, 'label'):
			if index not in annotate_plot_explore.clicked:
				annotate_plot_explore.label.remove()

		# Get data point from array of points X, at position index
		x2, y2, _ = proj3d.proj_transform(X[index, 0], X[index, 1], X[index, 2], ax.get_proj())

		if type(labels) is list:
			label = labels[index]
		else:
			label = "Index " + str(index) + ": (" + "{0:.2f}, ".format(X[index, 0]) + "{0:.2f}, ".format(X[index, 1]) + "{0:.2f}".format(X[index, 2]) + ")"

		annotate_plot_explore.label = plt.annotate(
		label,
		xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
		bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
		arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
		fig.canvas.draw()

	def onMouseMotion(event):
		"""Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
		closestIndex = calcClosestDatapoint(X, event)
		if type(labels) is list:
			annotatePlot (X, closestIndex, labels)
		else:
			annotatePlot (X, closestIndex)

	def onMouseClick(event):
		"""Event that is triggered when mouse is clicked. Preserves text annotation when mouse is clicked on datapoint."""
		closestIndex = calcClosestDatapoint(X, event)
		annotate_plot_explore.clicked.append(closestIndex)

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
			return plot1D(x)
		elif x.shape[-1]==2:
			return plot2D(x)
		elif x.shape[-1]==3:
			return plot3D(x)
		elif x.shape[-1]>3:
			return plot3D(reduceD(x, 3))

	def dispatch_list(x):
		if x[0].shape[-1]==1:
			return plot1D_list(x)
		elif x[0].shape[-1]==2:
			return plot2D_list(x)
		elif x[0].shape[-1]==3:
			return plot3D_list(x)
		elif x[0].shape[-1]>3:
			return plot3D_list(reduceD_list(x, 3))

	def plot1D(data):
		fig, ax = plt.subplots()
		x=np.arange(len(data)).reshape((len(data),1))
		plot2D(np.hstack((x, data)))
		return fig, ax, data

	def plot1D_list(data):
		fig, ax = plt.subplots()
		x=[]
		for i in range(0, len(data)):
			x.append(np.arange(len(data[i])).reshape(len(data[i]),1))
		plot_1to2_list(np.hstack((x, data)))
		return fig, ax, data

	def plot2D(data):
		fig, ax = plt.subplots()
		plt.plot(data[:,0], data[:,1], *args, **kwargs)
		return fig, ax, data

	def plot_1to2_list(data):
		n=len(data)
		fig, ax = plt.subplots()
		for i in range(n):
			m=len(data[i])
			half=m/2
			iargs = args_list[i]
			ikwargs = kwargs_list[i]
			ax.plot(data[i][0:half,0], data[i][half:m+1,0], *iargs, **ikwargs)
		return fig, ax, data

	def plot2D_list(data):
		# type: (object) -> object
		#if 2d, make a scatter
		n=len(data)
		fig, ax = plt.subplots()
		for i in range(n):
			iargs = args_list[i]
			ikwargs = kwargs_list[i]
			ax.plot(data[i][:,0], data[i][:,1], *iargs, **ikwargs)
		return fig, ax, data

	def plot3D(data):
		#if 3d, make a 3d scatter
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(data[:,0], data[:,1], data[:,2], *args, **kwargs)
		return fig, ax, data

	def plot3D_list(data):
		#if 3d, make a 3d scatter
		n=len(data)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for i in range(n):
			iargs = args_list[i]
			ikwargs = kwargs_list[i]
			ax.plot(data[i][:,0], data[i][:,1], data[i][:,2], *iargs, **ikwargs)
		return fig, ax, data

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

	def add_labels(data,labels):
		if explore:
			X = np.vstack(data)
			labels = list(itertools.chain(*labels))
			fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion
			fig.canvas.mpl_connect('button_press_event', onMouseClick)  # on mouse click
		elif labels:
			X = np.vstack(data)
			labels = list(itertools.chain(*labels))
			annotate_plot(X,labels)
			fig.canvas.mpl_connect('button_release_event', update_position)

	##MAIN FUNCTION##
	if is_list(x):
		if col_match(x):
			fig,ax,data = dispatch_list(x)
			add_labels(data,labels)
			plt.show()
		else:
			print "Inputted arrays must have the same number of columns"
	else:
		fig,ax,data = dispatch(x)
		add_labels(data,labels)
		plt.show()
