import hypertools as hyp
import scipy.io as sio
import numpy as np

data=sio.loadmat('weights.mat')
w = data['weights'][0]
aligned = hyp.align(w)
hyp.plot(aligned,animate=True, save_path='test-movie.mp4')
