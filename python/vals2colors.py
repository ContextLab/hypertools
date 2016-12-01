import seaborn as sns
import itertools

def vals2colors(vals,cmap='husl',res=100):
	"""Maps values to colors
	Args:
	values (list or list of lists) - list of values to map to colors
	cmap (str) - color map (default is 'husl')
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
