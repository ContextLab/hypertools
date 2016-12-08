import hypertools as hyp
import scipy.io as sio
import numpy as np

data=sio.loadmat('./weights.mat')
# w1=np.mean(data['weights'][0][:11])
# w2=np.mean(data['weights'][0][11:24])
# w3=np.mean(data['weights'][0][24:])
# aligned = hyp.align([w1,w2,w3])
#
# hyp.plot(aligned,animate=True)

w = data['weights'][0]
aligned = hyp.align(w)
hyp.plot(aligned,animate=True)
