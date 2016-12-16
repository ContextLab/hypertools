# from scipy.io import loadmat as load
# # from mvpa2.suite import *
# import hypertools as hyp
#
# data = load('test_data.mat')
# data1 = data['spiral']
# data2 = data['randwalk']
#
# hyp.plot([data1,data2])
#
# pc = hyp.Procrustes()
# pc.fit(data1,data2)
# hyp.plot([pc.transform(data1).T,data2])

# aligned1 = hyp.align(data)
#
# ha = Hyperalignment()
# mappers = ha(map(lambda x: Dataset(x), data))
# aligned2 = [m.forward(d) for m, d in zip(mappers, data)]
#

# import hypertools as hyp
# import scipy.io as sio
# import numpy as np
#
# data=sio.loadmat('weights.mat')
# w=data['weights'][0]
# w = [i for i in w]
#
# w1 = np.mean(w[:17],0)
# w2 = np.mean(w[18:],0)
#
# hyp.plot([w1[:100,:],w2[:100,:]])

# pc = hyp.Procrustes()
# pc.fit(w1[0].T,w1[1].T)
# hyp.plot([w1[0],pc.transform(w1[1].T)])

# data2=sio.loadmat('weights.mat')
# w2=data2['weights'][0]
# w2 = [i for i in w2]
# aligned_w2 = hyp.align(w2)

# import hypertools as hyp
# import scipy.io as sio
# import numpy as np
#
# data=sio.loadmat('weights.mat')
# w=data['weights'][0]
# w = [i for i in w]
#
# data1 = np.mean(w[:17],0)[:-100]
# data2 = np.mean(w[18:],0)[:-100]
#
# hyp.plot([data1,data2],animate=True)
#
# pc = hyp.Procrustes()
# pc.fit(data1,data2)
# hyp.plot([pc.transform(data1).T,data2],animate=True)

import hypertools as hyp
import scipy.io as sio
import numpy as np

data1 = np.array([[.1,.1,.1],[.2,.2,.2],[.3,.3,.3],[.4,.4,.4],[.5,.5,.5]]) + np.random.random((5,3))*.05
data2 = np.array([[-1,-1,1],[-2,-2,2],[-3,-3,3],[-4,-4,4],[-5,-5,5]]) + np.random.random((5,3))*.5
# data = sio.loadmat('test_data.mat')
# data1 = data['spiral']
# data2 = data['randwalk']
hyp.plot([data1,data2])

pc = hyp.Procrustes()
pc.fit(data1,data2)
hyp.plot([data1,pc.transform(data2)])
