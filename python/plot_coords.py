	#FOR CLEANUP##################################################
	#can't call main_helper before it is defined? thus, main_helper is at the bottom of the code
	# OPTION 5 won't work, not sure why..

	#may want to add ability to read pandas data frames
	#may want to add fancier plotting options

	#FOR TESTING##################################################

	#OPTION 1 - 1D matrix
	#x=np.array([[1], [2], [3], [4]])
    
    #OPTION 2 - 2D matrix 
    #x=np.array([[1, 11], [2, 12], [3, 13], [4, 14]])

    #OPTION 3 - 3d matrix
    #x=np.array([[1, 11, 21], [2, 12, 22], [3, 13, 33], [4, 14, 44]])

    #OPTION 4 - 4d matrix
    #x=np.array([[1, 11, 21, 31], [2, 12, 22, 32], [3, 13, 33, 43], [4, 14, 44, 54]])

    #OPTION 5 - 100d matrix
    #NOTE: this is a numpy array of numpy arrays, whereas the previous examples were numpy arrays of regular arrays. both will work.
    #x=np.array([[np.random.uniform(size=(100,))],[np.random.uniform(size=(100,))], [np.random.uniform(size=(100,))], [np.random.uniform(size=(100,))]])

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
		if x.shape[-1]==1:
			bar(x)
		if x.shape[-1]==2:
			scatter2D(x)
		if x.shape[-1]==3:
			scatter3d(x)
		if x.shape[-1]>3:
			lowD(x)

	def bar(data):
		pos=np.arange(len(data))
		plt.xticks(pos+0.4, pos)
		plt.bar(pos,data)
		plt.show()

	def scatter2D(data):
		plt.scatter(data[:,0], data[:,1])
		plt.show()

	def scatter3d(data):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(data[:,0], data[:,1], data[:,2], depthshade=True)
		plt.show()

	def lowD(x):
		m = PCA(n_components=3, whiten=True)
		#n_components=3--> reduce to 3 dimensions
		m.fit(x)
		z = m.transform(x)
		plot_coords(z)
		
	main_helper(x)
