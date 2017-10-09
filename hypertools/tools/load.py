import requests
import pickle
import pandas as pd
import deepdish as dd
import sys
from warnings import warn
from .analyze import analyze
from .._shared.helpers import format_data
from ..datageometry import DataGeometry

def load(dataset, reduce=None, ndims=None, align=None, normalize=None):
    """
    Load a .geo file or example data

    Parameters
    ----------
    dataset : string
        The name of the example dataset.  Can be a `.geo` file, or one of a
        number of example datasets listed below. `weights` is an fmri dataset
        comprised of 36 subjects.  For each subject, the rows are fMRI
        measurements and the columns are parameters of a model fit to the fMRI data. `weights_sample` is a sample of 3 subjects from that dataset.  `weights_avg` is the dataset split in half and averaged into two groups. `spiral` is 3D spiral to
        highlight the `procrustes` function.  `mushrooms` is an example dataset
        comprised of features (columns) of a collection of mushroomm samples (rows).

    normalize : str or None

    reduce : str or dict or None
        If not None, reduce data to ndims dimensions

    align : str or None
        If True, run data through alignment algorithm in tools.alignment

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

    if dataset[-4:] == '.geo':
        geo = dd.io.load(dataset)
        data = DataGeometry(fig=None, ax=None, data=geo['data'],
                            xform_data=geo['xform_data'], line_ani=None,
                            reduce=geo['reduce'], align=geo['align'],
                            normalize=geo['normalize'], kwargs=geo['kwargs'],
                            version=geo['version'])
    elif dataset is 'weights':
        fileid = '0B7Ycm4aSYdPPREJrZ2stdHBFdjg'
        url = 'https://docs.google.com/uc?export=download&id=' + fileid
        data = pickle.loads(requests.get(url, stream=True).content, **pickle_options)
    elif dataset is 'weights_avg':
        fileid = '0B7Ycm4aSYdPPRmtPRnBJc3pieDg'
        url = 'https://docs.google.com/uc?export=download&id=' + fileid
        data = pickle.loads(requests.get(url, stream=True).content, **pickle_options)
    elif dataset is 'weights_sample':
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

    return analyze(data, reduce=reduce, ndims=ndims, align=align, normalize=normalize)
