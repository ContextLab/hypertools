# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import NotFittedError


def pad(x, c=None):
    """
    Horizontally pad one or more numpy arrays

    Parameters
    ----------
    x: a DataFrame or list of DataFrames
    c: (optional) specify the desired number of columns; default: None

    Returns
    -------
    :return: a padded DataFrame or list of padded DataFrames
    """
    if type(x) is list:
        if c is None:
            c = np.max([d.shape[1] for d in x])
        return [pad(d, c) for d in x]

    if c is None:
        return x

    y = np.zeros([x.shape[0], c])
    n = np.min([c, x.shape[1]])
    y[:, :n] = x.iloc[:, :n]
    return pd.DataFrame(data=y, index=x.index.copy())


def trim_and_pad(data):
    """
    Select out the common rows of a dataset and pad the columns as needed

    Parameters
    ----------
    :data: an DataFrame or list of DataFrames

    Returns
    -------
    :return: a DataFrame, or a list of DataFrames with compatible rows and columns
    """
    if len(data) == 0:
        return data

    if type(data) is not list:
        data = [data]

    # get common rows
    rows = set(data[0].index.values)
    for d in data[1:]:
        rows = rows.intersection(set(d.index.values))

    c = np.max([x.shape[1] for x in data])
    x = [pad(d.loc[rows], c) for d in data]
    return x


class Aligner(BaseEstimator):
    """
    Base class used to create Aligner objects.  Must supply the following keyword arguments to initialzed:
      - :param data: a dataset
      - :param fitter: a function for aligning the dataset (takes the dataset as its first argument, along with
          zero or more keyword arguments).  Returns a dictionary of fitted parameters.
      - :param transformer: a function for transforming the dataset (takes the data as an argument, and fitted
          parameters passed as keyword arguments

    :return: instances of the Aligner class are scikit-learn compatible model objects
    """
    def __init__(self, **kwargs):
        self.data = kwargs.pop('data', None)
        self.fitter = kwargs.pop('fitter', None)
        self.transformer = kwargs.pop('transformer', None)
        self.required = kwargs.pop('required', [])
        self.kwargs = kwargs

    def fit(self, data):
        assert data is not None, ValueError('cannot align empty dataset')
        self.data = data

        if self.fitter is None:
            NotFittedError('null fit function; returning without fitting alignment model')
            return

        data = trim_and_pad(dw.unstack(self.data))
        # noinspection DuplicatedCode
        params = self.fitter(data, **self.kwargs)
        assert type(params) is dict, ValueError('fit function must return a dictionary')
        assert all([r in params.keys() for r in self.required]), ValueError('one or more required fields not'
                                                                            'returned')
        for k, v in params.items():
            setattr(self, k, v)

    def transform(self, *_):
        assert self.data is not None, NotFittedError('must fit aligner before transforming data')
        for r in self.required:
            assert hasattr(self, r), NotFittedError(f'missing fitted attribute: {r}')

        if self.transformer is None:
            RuntimeWarning('null transform function; returning without fitting alignment model')
            return

        data = trim_and_pad(dw.unstack(self.data))
        required_params = {r: getattr(self, r) for r in self.required}
        return [self.transformer(d, **dw.core.update_dict(required_params, self.kwargs)) for d in data]

    def fit_transform(self, data):
        self.fit(data)
        return self.transform()
