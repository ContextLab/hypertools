import hypertools as hyp
import scipy.io as sio
import numpy as np

data=sio.loadmat('./weights.mat')
w=[i for i in data['weights'][0][0:3]]

hyp.describe(w)
