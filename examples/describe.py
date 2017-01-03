import hypertools as hyp
import scipy.io as sio
import numpy as np
import os

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sample_data/')
data=sio.loadmat(datadir + 'weights.mat')
w=[i for i in data['weights'][0][0:3]]

hyp.tools.describe_pca(w)
