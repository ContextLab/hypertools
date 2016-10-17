#reload(plot_coords.py)
import numpy as np
import plot_coords as coords
import hyperalign as hyp
import scipy.io as sio
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib import pylab
import matplotlib.cm as cm
import matplotlib.colors as col

def parse(*args):

	if len(args)<=1:
		if all(isinstance(x, int) for x in args[0]):
			print "Only one dataset"


		elif all(isinstance(x, np.ndarray) for x in args[0][0]): #and all(isinstance(x, numpy.float32) for x in args[0][0][0]):
			print "array or list of arrays"


		elif all(isinstance(x, np.ndarray) for x in args[0]) and all(isinstance(x, int) for x in args[0][0]):
			print "single array"

		else: 
			print "Input argument elements are neither all ints nor all numpy arrays..."

	else:
		if all(isinstance(x, np.ndarray) for x in args):
			print "multiple arrays"


		else:
			print "Input datasets should be numpy arrays"
