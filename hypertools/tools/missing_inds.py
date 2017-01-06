#!/usr/bin/env python

##PACKAGES##
import numpy as np
from .._shared.helpers import format_data

def missing_inds(x):

	x = format_data(x)

	inds = [[idx for idx,row in enumerate(arr) if any(np.isnan(row))] for arr in x]
	if len(inds)>1:
		return inds
	else:
		return inds[0]
