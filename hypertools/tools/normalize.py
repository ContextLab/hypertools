#!/usr/bin/env python

from __future__ import division
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def normalize(x, normalize='across'):

    assert normalize in ['across','within','row', False], "scale_type must be across, within, row or none."

    if type(x) is not list:
        x = [x]

    zscore = lambda X,y: (y - np.mean(X)) / np.std(X)

    if normalize=='across':
        x_stacked=np.vstack(x)
        return [np.array([zscore(x_stacked[:,j], i[:,j]) for j in range(i.shape[1])]).T for i in x]

    elif normalize=='within':
        return [np.array([zscore(i[:,j], i[:,j]) for j in range(i.shape[1])]).T for i in x]

    elif normalize=='row':
        return [np.array([zscore(i[j,:], i[j,:]) for j in range(i.shape[0])]) for i in x]

    elif normalize==False:
        return x
