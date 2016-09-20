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
from sklearn.decomposition import PCA

def plot_coords(x):
    """
    inputs: TxD matrix of observations
               T-number of coords
               D-dimensionality of each observation
               *Nans treated as missing observations
            type (specify the type of plot)
               if 'scatter', make a scatterplot
               if 'line', make a line plot
    outputs: 1-, 2-, or 3-dimensional representation of the data
    """
    typedict = dict()
    typedict['scatter'] = ('k.', 'Axes3D.scatter')
    typedict['line'] = ('k-', 'Axes3D.Line')

    def dispatch(x, type='scatter'):
        #determine how many dimensions (number of columns)
        if x.shape[-1]==1:
            plot1D(x, type)
        elif x.shape[-1]==2:
            plot2D(x, type)
        elif x.shape[-1]==3:
            plot3D(x, type)
        elif x.shape[-1]>3:
            plot3D(reduceD(x, 3), type)

    def plot1D(data, type):
        x=np.arange(len(data))
        plot2D(np.hstack((np.transpose(x), data), type))

    def plot2D(data, type):
        # type: (object) -> object
        #if 2d, make a scatter
        plt.plot(data[:,0], data[:,1], typedict[type][0])

    def plot3D(data, type):
        #if 3d, make a 3d scatter
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        c = [0, 0, 0]
        plotfun = eval('ax.' + typedict[type][1])
        plotfun(data[:,0], data[:,1], data[:,2], c=c, depthshade=True)

    def reduceD(x, ndim):
        #if more than 3d, reduce to 3 (PCA), then re-run
        m = PCA(n_components=ndim, whiten=True)
        #n_components=3--> reduce to 3 dimensions
        m.fit(x)
        return m.transform(x)
		
    dispatch(x)
    plt.show()


x=np.array([[1, 11, 21, 31], [2, 12, 22, 32], [3, 13, 33, 43], [4, 14, 44, 54]])
plot_coords(x)