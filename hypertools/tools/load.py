import requests
import pickle
import pandas as pd
import sys
from warnings import warn

from .reduce import reduce as reducer
from .align import align as aligner
from .._shared.helpers import format_data

def load(dataset, ndims=None, align=False):
    """
    Load example data

    Parameters
    ----------
    dataset : string
        The name of the example dataset.  `weights` is an fmri dataset comprised of
        36 subjects.  For each subject, the rows are fMRI measurements and the columns
        are parameters of a model fit to the fMRI data. `weights_sample` is a
        sample of 3 subjects from that dataset.  `weights_avg` is the dataset split
        in half and averaged into two groups. `spiral` is 3D spiral to
        highlight the `procrustes` function.  `mushrooms` is an example dataset
        comprised of features (columns) of a collection of mushroomm samples (rows).

    Returns
    ----------
    data : Numpy Array
        Example data

    ndims : int
        If not None, reduce data to ndims dimensions

    align : bool
        If True, run data through alignment algorithm in tools.alignment

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
    if dataset is 'weights_avg':
        fileid = '0B7Ycm4aSYdPPRmtPRnBJc3pieDg'
        url = 'https://docs.google.com/uc?export=download&id=' + fileid
        data = pickle.loads(requests.get(url, stream=True).content, **pickle_options)
    if dataset is 'weights_sample':
        fileid = '0B7Ycm4aSYdPPTl9IUUVlamJ2VjQ'
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

    if ndims is not None:
        data = reducer(data, ndims=ndims, internal=True)
    if align:
        data = aligner(data)

    return data
