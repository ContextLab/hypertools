import hypertools as hyp
import scipy.io as sio
import numpy as np

data=sio.loadmat('./weights.mat')
w = data['weights'][0]
w = [i for i in w]

aligned = hyp.align(w)
hyp.plot(w,animate=True)
