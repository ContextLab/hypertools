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
import seaborn as sns

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
		sns.set_palette(palette="GnBu_d", n_colors=len(x))

	# set default colors
	# cargs_list = [];
	# palette = sns.color_palette()
	# for idx,i in enumerate(x):
	# 	cargs_list.append({
	# 		'color': palette[idx],
	# 	})

	# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'];
	# markers = ['.', 'o', 'x', '+', '*', 's', 'd', 'v', '^', '<', '>', 'p', 'h', 'square', 'diamond', 'pentagram', 'hexagram'];
	# linestyles = ['-', ':', '-.', '--'];
	# format_strs = list(itertools.chain(colors, markers, linestyles))

	# def is_format_str(arg):
	# 	format_str_set = []
	# 	for color in colors:
	# 		format_str_set.append(color)
	# 	for marker in markers:
	# 			format_str_set.append(marker)
	# 	for linestyle in linestyles:
	# 		format_str_set.append(linestyle)
	# 	for marker in markers:
	# 		format_str_set.append(color+marker)
	# 		for linestyle in linestyles:
	# 			format_str_set.append(color+linestyle)
	# 	if arg in format_str_set:
	# 		return True
	# 	else:
	# 		return False

	# def parse_format_str(arg):
	# 	cdict = {}
	# 	if arg in colors:
	# 		cdict['color']=arg
	# 	if arg in markers:
	# 		cdict['marker']=arg
	# 		cdict['linestyle']=''
	# 	if arg in linestyles:
	# 		cdict['linestyle']=arg
	# 	return cdict

	#PARSE ARGS##
	# def parse_args(args):
	# 	for idx,arg in enumerate(args):
	# 		if is_format_str(arg):
	# 			parsed_format_str = parse_format_str(arg)
	# 			del args[idx]
	# 			for key in parsed_format_str:
	# 				cargs_list[idx][key]=parsed_format_str[key]
	# 		elif type(arg) is tuple:
	# 			for idx,a in enumerate(arg):
	# 				parsed_format_str = parse_format_str(a)
	# 				for key in parsed_format_str:
	# 					cargs_list[idx][key]=parsed_format_str[key]
	#  	return tuple(x for x in args if type(x) is not tuple)
	#
	# args = parse_args(args)

	args_list = []
	for i,item in enumerate(x):
		tmp = []
		for ii,arg in enumerate(args):
			if type(arg) is tuple or type(arg) is list:
				tmp.append(arg[i])
			else:
				tmp.append(arg)
		args_list.append(tuple(tmp))

	##PARSE KWARGS##
	kwargs_list = []
	for i,item in enumerate(x):
		tmp = {}
		for kwarg in kwargs:
			if type(arg) is tuple or type(arg) is list:
				tmp[kwarg]=kwargs[kwarg][i]
			else:
				tmp[kwarg]=kwargs[kwarg]
		kwargs_list.append(tmp)

	# ##PARSE KWARGS COLORS##
	# if 'color' in kwargs:
	# 	if len(kwargs['color'])==len(x):
	# 		for idx,color in enumerate(kwargs['color']):
	# 			cargs_list[idx]['color']=color
	# 	else:
	# 		print('Error: colors must be same length as x.')
	# 		sys.exit(1)
	#
	# ##PARSE KWARGS LINESTYLES##
	# if 'linestyle' in kwargs:
	# 	if len(kwargs['linestyle'])==len(x):
	# 		for idx,linestyle in enumerate(kwargs['linestyle']):
	# 			cargs_list[idx]['linestyle']=linestyle
	# 	else:
	# 		print('Error: colors must be same length as x.')
	# 		sys.exit(1)


	##PARSE PLOT_COORDS SPECIFIC ARGS##
	if 'ndims' in kwargs:
		ndims=kwargs['ndims']
		del kwargs['ndims']

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
		plt.plot(data[:,0], data[:,1], c=colors, *args, **kwargs)

	def plot_1to2_list(data):
		n=len(data)
		fig, ax = plt.subplots()
		for i in range(n):
			m=len(data[i])
			half=m/2
			cargs = cargs_list[i]
			for key in cargs:
				kwargs[key]=cargs[key]
			ax.plot(data[i][0:half,0], data[i][half:m+1,0], *args, **kwargs)

	def plot2D_list(data):
		# type: (object) -> object
		#if 2d, make a scatter
		n=len(data)
		fig, ax = plt.subplots()
		for i in range(n):
			cargs = cargs_list[i]
			for key in cargs:
				kwargs[key]=cargs[key]
			ax.plot(data[i][:,0], data[i][:,1], *args, **kwargs)

	def plot3D(data):
		#if 3d, make a 3d scatter
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(data[:,0], data[:,1], data[:,2], c=color, ls=linestyle, *args, **kwargs)

	# def get_ikwargs(i):
	# 	cargs = cargs_list[i]
	# 	ikwargs=kwargs.copy()
	# 	for key in cargs:
	# 		ikwargs[key]=cargs[key]
	# 	return ikwargs

	def plot3D_list(data):
		#if 3d, make a 3d scatter
		n=len(data)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for i in range(n):
			iargs = args_list[i]
			ikwargs = kwargs_list[i]
			ax.plot(data[i][:,0], data[i][:,1], data[i][:,2], *iargs, **ikwargs)

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
