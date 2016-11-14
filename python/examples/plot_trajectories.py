import plot_coords as coords
import hyperalign as hyp
import scipy.io as sio
import numpy as np

data=sio.loadmat('../weights.mat')
w=data['weights'][0][:3]
coords.plot_coords((w[0],w[1]),color=['r','g'])
