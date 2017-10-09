import requests
import pickle
import pandas as pd
<<<<<<< HEAD
import deepdish as dd
import sys
from warnings import warn
from .analyze import analyze
from .._shared.helpers import format_data
from ..datageometry import DataGeometry

def load(dataset, reduce=None, ndims=None, align=None, normalize=None):
    """
    Load a .geo file or example data
=======
import sys
from warnings import warn

from .reduce import reduce as reduceD
from .align import align as aligner

def load(dataset, ndims=None, align=False):
    """
    Load example data
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764

    Parameters
    ----------
    dataset : string
<<<<<<< HEAD
        The name of the example dataset.  Can be a `.geo` file, or one of a
        number of example datasets listed below. `weights` is an fmri dataset
        comprised of 36 subjects.  For each subject, the rows are fMRI
        measurements and the columns are parameters of a model fit to the fMRI data. `weights_sample` is a sample of 3 subjects from that dataset.  `weights_avg` is the dataset split in half and averaged into two groups. `spiral` is 3D spiral to
        highlight the `procrustes` function.  `mushrooms` is an example dataset
        comprised of features (columns) of a collection of mushroomm samples (rows).

    normalize : str or False or None
        If set to 'across', the columns of the input data will be z-scored
        across lists (default). That is, the z-scores will be computed with
        with repect to column n across all arrays passed in the list. If set
        to 'within', the columns will be z-scored within each list that is
        passed. If set to 'row', each row of the input data will be z-scored.
        If set to False, the input data will be returned with no z-scoring.

    reduce : str or dict
        Decomposition/manifold learning model to use.  Models supported: PCA,
        IncrementalPCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, FastICA,
        FactorAnalysis, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
        TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, and MDS. Can be
        passed as a string, but for finer control of the model parameters, pass
        as a dictionary, e.g. reduce={'model' : 'PCA', 'params' : {'whiten' : True}}.
        See scikit-learn specific model docs for details on parameters supported
        for each model.

    ndims : int
        Number of dimensions to reduce

    align : str or dict
        If str, either 'hyper' or 'SRM'.  If 'hyper', alignment algorithm will be
        hyperalignment. If 'SRM', alignment algorithm will be shared response
        model.  You can also pass a dictionary for finer control, where the 'model'
        key is a string that specifies the model and the params key is a dictionary
        of parameter values (default : 'hyper').

=======
        The name of the example dataset.  `weights` is an fmri dataset comprised of
        36 subjects.  For each subject, the rows are fMRI measurements and the columns
        are parameters of a model fit to the fMRI data. `weights_sample` is a
        sample of 3 subjects from that dataset.  `weights_avg` is the dataset split
        in half and averaged into two groups. `spiral` is 3D spiral to
        highlight the `procrustes` function.  `mushrooms` is an example dataset
        comprised of features (columns) of a collection of mushroomm samples (rows).

>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
    Returns
    ----------
    data : Numpy Array
        Example data

<<<<<<< HEAD
=======
    ndims : int
        If not None, reduce data to ndims dimensions

    align : bool
        If True, run data through alignment algorithm in tools.alignment

>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
    """
    if sys.version_info[0]==3:
        pickle_options = {
            'encoding' : 'latin1'
        }
    else:
        pickle_options = {}

<<<<<<< HEAD
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
=======
    if dataset is 'weights':
        fileid = '0B7Ycm4aSYdPPREJrZ2stdHBFdjg'
        url = 'https://docs.google.com/uc?export=download&id=' + fileid
        data = pickle.loads(requests.get(url, stream=True).content, **pickle_options)
    if dataset is 'weights_avg':
        fileid = '0B7Ycm4aSYdPPRmtPRnBJc3pieDg'
        url = 'https://docs.google.com/uc?export=download&id=' + fileid
        data = pickle.loads(requests.get(url, stream=True).content, **pickle_options)
    if dataset is 'weights_sample':
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
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

<<<<<<< HEAD
    return analyze(data, reduce=reduce, ndims=ndims, align=align, normalize=normalize)
=======
    if ndims is not None:
        data = reduceD(data, ndims, internal=True)
    if align:
        if len(data) == 1:
            warn('Data in list of length 1 can not be aligned. '
                 'Skipping the alignment.')
        else:
            data = aligner(data)

    return data
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
