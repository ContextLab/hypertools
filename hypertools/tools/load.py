import requests
import pickle
import pandas as pd
import sys

def load(dataset):
	"""
	Load example data

    Parameters
    ----------
    dataset : string
        The name of the example dataset.  This can be weights, spiral or mushrooms.

    Returns
    ----------
    data : Numpy Array
        Example data

    """
	if sys.version_info[0]==3:
		pickle_options = {
			'encoding' : 'latin1'
		}
	else:
		pickle_options = {}

	if dataset is 'weights':
		fileid = '0B7Ycm4aSYdPPREJrZ2stdHBFdjg'
		url = 'https://docs.google.com/uc?export=download&id=' + fileid
		data = pickle.loads(requests.get(url, stream=True).content, **pickle_options)
	elif dataset is 'spiral':
		fileid = '0B7Ycm4aSYdPPQS0xN3FmQ1FZSzg'
		url = 'https://docs.google.com/uc?export=download&id=' + fileid
		data = pickle.loads(requests.get(url, stream=True).content, **pickle_options)
	elif dataset is 'mushrooms':
		fileid = '0B7Ycm4aSYdPPY3J0U2tRNFB4T3c'
		url = 'https://docs.google.com/uc?export=download&id=' + fileid
		data = pd.read_csv(url)

	return data
