	#FOR TESTING##################################################

	#OPTION 1 - 1D matrix
	#x=np.array([[1], [2], [3], [4]])
    
    #OPTION 2 - 2D matrix 
    #x=np.array([[1, 11], [2, 12], [3, 13], [4, 14]])

    #OPTION 3 - 3d matrix
    #x=np.array([[1, 11, 21] [2, 12, 22] [3, 13, 33] [4, 14, 44]])


    #OPTION 4 - use "weights" --> import weights.mat and save to 'data' as numpy.ndarray
	#import scipy.io 
	#data_0=scipy.io.loadmat('weights.mat')
    #x=data_0['weights']

    ##############################################################

def plot_coords(x):
	"""
		inputs: TxD matrix of observations
	           T-number of coords 
	           D-dimensionality of each observation
	           *Nans treated as missing observations
	    outputs: 1-, 2-, or 3-dimensional representation of the data
	"""
    
def main_helper(x=np.array([[1], [2], [3], [4]])):
	import numpy as np 
	import matplotlib.pyplot as plt 
	from plotting import *
	#from ggplot import *
	#import pandas as pd

	if x.shape[-1]==1:
	#shape gives (rows,columns) --> x.shape[-1] == # of columns 
	#in N-dimenional data, is the column # still displayed in the same position?
		bar(x)
		#^if #columns==1, make a bar graph (see 'bar')

	if x.shape[-1]==2:
		#if number of columns ==2
		scatter2D(x)
		#make a scatterplot

	if x.shape[-1]==3:
		# if 3 columns
		scatter3d(x)

def bar(data):
    pos=np.arange(len(data))
    #arange returns evenly spaced values within given interval
    plt.xticks(pos+0.4, pos)
    #put ticks half way between each bar
    plt.bar(pos,data)
    #make the plot
    plt.show()
    #display it

def scatter2D(data):
	plt.scatter(data[:,0], data[:,1])
	#uses first two columns of the input data
	plt.show()

def scatter3d(data):






