import hypertools as hyp
import numpy as np

w = np.cumsum(np.random.multivariate_normal(np.zeros(3), np.eye(3), size=1000),axis=0)
hyp.plot(w,animate=True)
