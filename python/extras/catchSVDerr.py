import numpy as numpy


def is_pos_def(x):
		return np.all(np.linalg.eigenvals(x)>0)

def make_pos_def(x):
	if is_pos_def(x):
		return x
	else:
		return nearPD(x)

#get near PD from slack
