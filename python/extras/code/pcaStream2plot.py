import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import time
import scipy.io as sio
import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator as pchip
import PCA as PCA
from mpl_toolkits.mplot3d import Axes3D
import hyperalign as hyp

fig=plt.figure()
ax1=fig.add_subplot(1, 1, 1)

    
def interp(z):
    x=np.arange(0, len(z), 1)
    xx=np.arange(0, len(z)-1, .1)
    q=pchip(x,z)
    return q(xx)

def animate(i):
	graph_data = open('samplefile.txt','r').read()
	lines=graph_data.split('\n')

	xs=[]
	ys=[]

	for line in lines:
		if len(line)>1:
			x,y=line.split(',')
			xs.append(x)
			ys.append(y)
	ax1.clear
	ax1.plot(xs, ys)
ani=animation.FuncAnimation(fig, animate, interval=2000)