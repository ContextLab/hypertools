#!/usr/bin/env python

"""
INPUTS:
-numpy array(s)
-list of numpy arrays

OUTPUTS:
-matplotlib plt, fig, ax handles as tuple
"""

##PACKAGES##
import sys
import warnings
import itertools
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns
from .._shared.helpers import *

matplotlib.rcParams['pdf.fonttype'] = 42

##MAIN FUNCTION##
def static_plot(x, *args, **kwargs):
	"""
	implements plotting
	"""

	##PARSE HYPERTOOLS SPECIFIC ARGUMENTS##

	# optionally return data
	if 'return_data' in kwargs:
		return_data=kwargs['return_data']
		del kwargs['return_data']
	else:
		return_data=False

	# handle legend
	if 'legend' in kwargs:
		kwargs['label'] = kwargs['legend']
		del kwargs['legend']
		legend=True
	else:
		legend=False

	# save path
	if 'save_path' in kwargs:
		save=True
		save_path = kwargs['save_path']
		del kwargs['save_path']
	else:
		save=False

    # handle labels flag
	if 'labels' in kwargs:
		labels=kwargs['labels']
		del kwargs['labels']
	else:
		labels=False

    # handle explore flag
	if 'explore' in kwargs:
		kwargs['picker']=True
		del kwargs['explore']
		explore=True
	else:
		explore=False

	# handle dims flag
	if 'ndims' in kwargs:
		assert (kwargs['ndims'] in [1,2,3]), 'ndims must be 1,2 or 3.'
		x = reduceD(x,kwargs['ndims'])
		del kwargs['ndims']

	# handle show flag
	if 'show' in kwargs:
		show=kwargs['show']
		del kwargs['show']
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
	else:
		show=True
		import matplotlib.pyplot as plt

	##PARSE LEFTOVER MATPLOTLIB ARGS##
	args_list = parse_args(x,args)

	##PARSE LEFTOVER MATPLOTLIB KWARGS##
	kwargs_list = parse_kwargs(x,kwargs)

	##SUB FUNCTIONS##
	def dispatch(x):
		if x[0].shape[-1]==1:
			return plot1D(x)
		elif x[0].shape[-1]==2:
			return plot2D(x)
		elif x[0].shape[-1]==3:
			return plot3D(x)

	# plot data in 1D
	def plot1D(data):
		n=len(data)
		fig, ax = plt.subplots()
		for i in range(n):
			iargs = args_list[i]
			ikwargs = kwargs_list[i]
			ax.plot(data[i][:,0], *iargs, **ikwargs)
		return fig, ax, data

	# plot data in 2D
	def plot2D(data):
		n=len(data)
		fig, ax = plt.subplots()
		for i in range(n):
			iargs = args_list[i]
			ikwargs = kwargs_list[i]
			ax.plot(data[i][:,0], data[i][:,1], *iargs, **ikwargs)
		return fig, ax, data

	# plot data in 3D
	def plot3D(data):
		n=len(data)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for i in range(n):
			iargs = args_list[i]
			ikwargs = kwargs_list[i]
			ax.plot(data[i][:,0], data[i][:,1], data[i][:,2], *iargs, **ikwargs)
		return fig, ax, data

	##LABELS##
	def annotate_plot(data, labels):
		"""Create labels in 3d chart
		Args:
			X (np.array) - array of points, of shape (numPoints, 3)
			labels (list) - list of labels of shape (numPoints,1)
		Returns:
			None
		"""

		global labels_and_points
		labels_and_points = []

		if data[0].shape[-1]>2:
			proj = ax.get_proj()

		for idx,x in enumerate(data):
			if labels[idx] is not None:
				if data[0].shape[-1]>2:
					x2, y2, _ = proj3d.proj_transform(x[0], x[1], x[2], proj)
					label = plt.annotate(
					labels[idx],
					xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
					bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.5),
					arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'),family='serif')
					labels_and_points.append((label,x[0],x[1],x[2]))
				elif data[0].shape[-1]==2:
					x2, y2 = x[0], x[1]
					label = plt.annotate(
					labels[idx],
					xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
					bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.5),
					arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'),family='serif')
					label.draggable()
					labels_and_points.append((label,x[0],x[1]))
		fig.canvas.draw()

	def update_position(e):
		"""Update label positions in 3d chart
		Args:
			e (mouse event) - event handle to update on
		Returns:
			None
		"""

		proj = ax.get_proj()
		for label, x, y, z in labels_and_points:
			x2, y2, _ = proj3d.proj_transform(x, y, z, proj)
			label.xy = x2,y2
			label.update_positions(fig.canvas.renderer)
			label._visible=True
		fig.canvas.draw()

	def hide_labels(e):
		"""Hides labels on button press
		Args:
			e (mouse event) - event handle to update on
		Returns:
			None
		"""

		for label in labels_and_points:
			label[0]._visible=False

	def add_labels(data,labels=False):
		"""Add labels to graph if available
		Args:
			data (np.ndarray) - Array containing the data points
			labels (list) - List containing labels
		Returns:
			None
		"""
		# if explore mode is activated, implement the on hover behavior
		if explore:
			X = np.vstack(data)
			if labels:
				if any(isinstance(el, list) for el in labels):
					labels = list(itertools.chain(*labels))
				fig.canvas.mpl_connect('motion_notify_event', lambda event: onMouseMotion(event, X, labels)) # on mouse motion
				# fig.canvas.mpl_connect('button_press_event', lambda event: onMouseClick(event, X, labels))  # on mouse click
			else:
				fig.canvas.mpl_connect('motion_notify_event', lambda event: onMouseMotion(event, X)) # on mouse motion
				# fig.canvas.mpl_connect('button_press_event', lambda event: onMouseClick(event, X, labels))  # on mouse click

		elif labels:
			X = np.vstack(data)
			if any(isinstance(el, list) for el in labels):
				labels = list(itertools.chain(*labels))
			annotate_plot(X,labels)
			fig.canvas.mpl_connect('button_press_event', hide_labels)
			fig.canvas.mpl_connect('button_release_event', update_position)

	##EXPLORE MODE##
	def distance(point, event):
		"""Return distance between mouse position and given data point

		Args:
			point (np.array) -  np.array of shape (3,), with x,y,z in data coords
			event (MouseEvent) - mouse event (which contains mouse position in .x and .xdata)
		Returns:
			distance (np.float64) - distance (in screen coords) between mouse pos and data point
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

	def annotate_plot_explore(X, index, labels=False):
		"""Create popover label in 3d chart

		Args:
			X (np.array) - array of points, of shape (numPoints, 3)
			index (int) - index (into points array X) of item which should be printed
			labels (list or False) - list of data point labels (default is False)
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



	def onMouseMotion(event,X,labels=False):
		"""Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse
		Args:
			event (event) - event triggered when the mous is moved
			X (np.ndarray) - coordinates by datapoints matrix
			labels (list or False) - list of data labels (default is False)
		Returns:
			None
		"""

		closestIndex = calcClosestDatapoint(X, event)

		if hasattr(onMouseMotion, 'first'):
			pass
		else:
			onMouseMotion.first = False
			onMouseMotion.closestIndex_prev = calcClosestDatapoint(X, event)

		if closestIndex!=onMouseMotion.closestIndex_prev:
			if type(labels) is list:
				annotate_plot_explore (X, closestIndex, labels)
				closestIndex_prev = closestIndex
			else:
				annotate_plot_explore (X, closestIndex)
				closestIndex_prev = closestIndex

	# def onMouseClick(event,X):
	# 	"""Event that is triggered when mouse is clicked. Preserves text annotation when mouse is clicked on datapoint."""
	# 	closestIndex = calcClosestDatapoint(X, event)
	# 	annotate_plot_explore.clicked.append(closestIndex)

	##MAIN##
	check_data(x) # throws error if the arrays are not the same shape
	fig,ax,data = dispatch(x)
	add_labels(data,labels)
	if legend:
		plt.legend()
	if save:
		plt.savefig(save_path)
	if show:
		plt.show()
	if return_data:
		return fig,ax,data
