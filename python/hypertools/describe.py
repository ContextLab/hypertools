from __future__ import division
import numpy as np
from sklearn import PCA
from scipy.spatial.distance import pdist

import seaborn as sns
sns.set(style="darkgrid")

def describe(x):

    ##SUB FUNCTIONS##

    def describe_align(x):
        if type(x) not list:
            print('Must pass a list of arrays.')
        

    def describe_PCA(x):
        if type(x) is list:
            x = np.vstack(x)
        cov_alldims = pdist(x,'correlation')
        cov_PCA =  [(pdist(reduceD(x,num),'correlation')) for num in range(2,x.shape[1]+1)]
        return [np.corrcoef(cov_alldims, cov_PCA_i)[0][1] for cov_PCA_i in cov_PCA]

    if type(x) is list:
        continue
    else:
        x = [x]

    attrs = {}

    attrs['PCA_summary'] = {}
    attrs['PCA_summary']['average'] = describe_PCA(x)
    attrs['PCA_summary']['individual'] = [describe_PCA(x_i) for x_i in x]
