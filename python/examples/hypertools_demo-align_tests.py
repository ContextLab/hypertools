import hypertools as hyp
import scipy.io as sio
import numpy as np

data = sio.loadmat('test_data.mat')
data1 = data['spiral']
data2 = data['randwalk']
hyp.plot([data1,data2])

hyp.plot(hyp.align([data1,data2]))
