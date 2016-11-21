#import sys
#sys.path.append('../')
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




    
#def interp(z):
#    x=np.arange(0, len(z), 1)
#    xx=np.arange(0, len(z)-1, .1)
#    q=pchip(x,z)
#    return q(xx)

def animate(i):
    ax1.set_color_cycle(['red','red','grey']) #'purple','purple', 'grey'])
    
    graph_data = open("test1.csv",'r').read()
    lines=graph_data.split('\n')
    
    X=np.array([])
    Y=np.array([])
    Z=np.array([])


    for line in lines:
        if len(line)>1:
            x,y,z=line.split(',')
            X=np.append(X,int(x))
            Y=np.append(Y,int(y))
            Z=np.append(Z,int(z))
    
    print X
    print type(X)        
    
    ax1.clear
    
    if i<= 15:
        ax1.plot(X[0:i], Y[0:i], Z[0:i])
    if i>15:
        ax1.plot(X[0:i], Y[0:i], Z[0:i])
        #ax1.plot(X[i-15:i], Y[i-15:i], Z[i-15:i])
        #ax1.plot(X[i-18:i-15],Y[i-18:i-15],Z[i-18:i-15], ":")
        #ax1.plot(X[0:i-18],Y[0:i-18],Z[0:i-18], ":")
        
        
        
fig1=plt.figure()
ax1=fig1.add_subplot(111,projection='3d')
ani=animation.FuncAnimation(fig1, animate, interval=8)

plt.show()