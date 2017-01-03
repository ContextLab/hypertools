import hypertools as hyp
import scipy.io as sio
import numpy as np
import os

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sample_data/')
data=sio.loadmat(datadir + 'weights.mat')
w = [i for i in data['weights'][0]]
aligned = hyp.align(w)
hyp.plot(aligned, animate=True, save_path='test-movie.mp4')
