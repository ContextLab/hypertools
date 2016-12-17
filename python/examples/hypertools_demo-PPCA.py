import hypertools as hyp
import numpy as np

data1 = np.array(np.random.random((10,10)))
data1[1,0]=np.nan
data2 = np.array(np.random.random((10,10)))
data2[5,5]=np.nan
hyp.plot([data1,data2])
