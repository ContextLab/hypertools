import hypertools as hyp
import scipy.io as sio
import numpy as np

data=sio.loadmat('./weights.mat')
w=data['weights'][0][0:3]
w = [i for i in w]

hyp.describe(w)
