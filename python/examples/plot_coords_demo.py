import plot_coords as coords
import scipy.io as sio

data=sio.loadmat('../weights.mat')
w=data['weights'][0][:3]
coords.plot_coords(w)
