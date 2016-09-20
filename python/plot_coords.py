#FOR CLEANUP##################################################

#may want to add ability to read pandas data frames
#may want to add fancier plotting options

#FOR TESTING##################################################
	
#x=np.array([[1], [2], [3], [4]]) 
#x=np.array([[1, 11], [2, 12], [3, 13], [4, 14]])
#x=np.array([[1, 11, 21], [2, 12, 22], [3, 13, 33], [4, 14, 44]])
#x=np.array([[1, 11, 21, 31], [2, 12, 22, 32], [3, 13, 33, 43], [4, 14, 44, 54]])

##############################################################

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import lena
from sklearn.decomposition import PCA

def plot_coords(x):
	"""
		inputs: TxD matrix of observations
	           T-number of coords 
	           D-dimensionality of each observation
	           *Nans treated as missing observations
	    outputs: 1-, 2-, or 3-dimensional representation of the data
	"""
    
	def main_helper(x):
		#determine how many dimensions (number of columns)
		if x.shape[-1]==1:
			bar(x)
		if x.shape[-1]==2:
			scatter2D(x)
		if x.shape[-1]==3:
			scatter3d(x)
		if x.shape[-1]>3:
			lowD(x)

	def bar(data):
		#if 1d, make a bar graph
		pos=np.arange(len(data))
		plt.xticks(pos+0.4, pos)
		plt.bar(pos,data)
		plt.show()

	def scatter2D(data):
		#if 2d, make a scatter
		plt.scatter(data[:,0], data[:,1])
		plt.show()

	def scatter3d(data):
		#if 3d, make a 3d scatter
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(data[:,0], data[:,1], data[:,2], depthshade=True)
		plt.show()

	def lowD(x):
		#if more than 3d, reduce to 3 (PCA), then re-run
		m = PCA(n_components=3, whiten=True)
		#n_components=3--> reduce to 3 dimensions
		m.fit(x)
		z = m.transform(x)
		plot_coords(z)
		
	main_helper(x)
