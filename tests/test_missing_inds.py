# -*- coding: utf-8 -*-

import numpy as np

data = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=10)
data[3,0]=np.nan
data[9,1]=np.nan
