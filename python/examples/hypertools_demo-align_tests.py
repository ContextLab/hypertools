import hypertools as hyp
import scipy.io as sio
import numpy as np

data = sio.loadmat('sample_data/test_data.mat')
data1 = data['spiral']
data2 = data['randwalk']
hyp.plot([data1, data2])

hyp.plot(hyp.util.align([data1, data2]))

# A random rotation matrix
rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
       [-0.43426149,  0.87492975, -0.21427761],
       [-0.10761949,  0.18578133,  0.97667976]])
# creating new spiral with some noise
data_rot = np.dot(data1, rot) + np.random.randn(data1.shape[0], data1.shape[1])*0.05
# before hyperalignment
hyp.plot([data1, data_rot])
# After hyperalignment
hyp.plot(hyp.util.align([data1, data_rot]))
