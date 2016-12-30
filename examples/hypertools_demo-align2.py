import hypertools as hyp
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = sio.loadmat('sample_data/test_data.mat')
data1 = data['spiral']

# A random rotation matrix
rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
       [-0.43426149,  0.87492975, -0.21427761],
       [-0.10761949,  0.18578133,  0.97667976]])
# creating new spiral with some noise
data_rot = np.dot(data1, rot) + np.random.randn(data1.shape[0], data1.shape[1])*0.05

# Before hyperalignment
fig,ax,data = hyp.plot([data1, data_rot], show=False)
ax.set_title('Before Alignment')
plt.show()

# After hyperalignment
fig,ax,data = hyp.plot(hyp.util.align([data1, data_rot]), show=False)
ax.set_title('After Alignment')
plt.show()
