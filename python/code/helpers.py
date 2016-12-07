import numpy as np
from scipy.interpolate import PchipInterpolator as pchip
import seaborn as sns
import itertools

def vals2colors(vals,cmap='GnBu_d',res=100):
	"""Maps values to colors
	Args:
	values (list or list of lists) - list of values to map to colors
	cmap (str) - color map (default is 'husl')
	res (int) - resolution of the color map (default: 100)
	Returns:
	list of rgb tuples
	"""
	# flatten if list of lists
	if any(isinstance(el, list) for el in vals):
		vals = list(itertools.chain(*vals))

	# get palette from seaborn
	palette = sns.color_palette(cmap, res)

	# rank the values and then normalize
	ranks = list(map(lambda x: sum([val <= x for val in vals]),vals))
	ranks = list(map(lambda rank: int(round(res*rank/len(vals))),ranks))

	return [palette[rank-1] for rank in ranks]

# this will be moved to utils.py
def is_list(x):
    if type(x[0][0])==np.ndarray:
        return True
    elif type(x[0][0])==np.int64 or type(x[0][0])==int:
        return False

# #  this will be moved to utils.py
def interp_array(arr,interp_val=10):
    x=np.arange(0, len(arr), 1)
    xx=np.arange(0, len(arr)-1, 1/interp_val)
    q=pchip(x,arr)
    return q(xx)

# #  this will be moved to utils.py
def interp_array_list(arr_list,interp_val=10):
    smoothed= [np.zeros(arr_list[0].shape) for item in arr_list]
    for idx,arr in enumerate(arr_list):
        smoothed[idx] = interp_array(arr,interp_val)
    return smoothed

def get_cube_scale(x):
    x = np.vstack(x)
    x_square = np.square(x)
    x_ss = np.sum(x_square,axis=1)
    idx = [i for i in range(len(x_ss)) if x_ss[i]==np.max(x_ss)]
    print(np.linalg.norm(x[idx,:]))
    return np.linalg.norm(x[idx,:])
