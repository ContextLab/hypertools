import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from .._externals.ppca import PPCA

def format_data(x, ppca=False, text_args=None):
    """
    Formats data into a list of numpy arrays

    This function is useful to identify rows of your array that contain missing
    data or nans.  The returned indices can be used to remove the rows with
    missing data, or label the missing data points that are interpolated
    using PPCA.

    Parameters
    ----------

    x : numpy array, dataframe or (mixed) list
        The data to convert

    ppca : bool
        Performs PPCA to fill in missing values (default: False)

    Returns
    ----------
    data : list of numpy arrays
        A list of formatted arrays
    """

    from ..tools.text2mat import text2mat

    def get_type(data):
        """
        Checks what the data type is and returns it as a string label
        """
        if isinstance(data, list):
            if isinstance(data[0], str):
                return 'list_str'
            elif isinstance(data[0], (int, float)):
                return 'list_num'
        elif isinstance(data, np.ndarray):
            return 'array'
        elif isinstance(data, pd.DataFrame):
            return 'df'
        elif isinstance(data, str):
            return 'str'
        elif isinstance(data, (CountVectorizer, TfidfVectorizer)):
            return 'vecobj'
        else:
            raise TypeError('Unsupported data type passed. Supported types: '
                            'Numpy Array, Pandas DataFrame, String, List of strings'
                            ', List of numbers')

    def fill_missing(x):

        # ppca if missing data
        m = PPCA()
        m.fit(data=np.vstack(x))
        x_pca = m.transform()

        # if the whole row is missing, return nans
        all_missing = [idx for idx,a in enumerate(np.vstack(x)) if all([type(b)==np.nan for b in a])]
        if len(all_missing)>0:
            for i in all_missing:
                x_pca[i,:]=np.nan

        # get the original lists back
        if len(x)>1:
            x_split = np.cumsum([i.shape[0] for i in x][:-1])
            return list(np.split(x_pca, x_split, axis=0))
        else:
            return [x_pca]

    # not sure why i needed to import here, but its the only way I could get it to work
    from .df2mat import df2mat
    from .text2mat import text2mat

    # if x is not a list, make it one
    if type(x) is not list:
        x = [x]

    # check data type for each element in list
    dtypes = list(map(get_type, x))

    # handle text data:
    if any(map(lambda x: x in ['list_str', 'str', 'vecobj'], dtypes)):

        # default text args
        kwargs = {
            'vectorizer' : 'CountVectorizer',
            'vectorizer_params' : None,
            'text' : 'LatentDirichletAllocation',
            'text_params' : None,
            'n_components' : 20
        }

        # update with user specified args
        if text_args:
            kwargs.update(text_args)

        # filter text data
        text_data = []
        for i,j in zip(x, dtypes):
            if j in ['list_str', 'str']:
                text_data.append(np.array(i).reshape(-1, 1))
            elif j is 'vecobj':
                text_data.append(i)

        # convert text to numerical matrices
        text_data = text2mat(text_data, **kwargs)

    # replace the text data with transformed data
    for i, dtype in enumerate(dtypes):
        if dtype in ['list_str', 'str', 'vecobj']:
            x[i] = text_data.pop(0)
        elif dtype is 'df':
            x[i] = df2mat(x[i])

    # reshape anything that is 1d
    if any([i.ndim<=1 for i in x]):
        x = [np.reshape(i,(i.shape[0],1)) if i.ndim==1 else i for i in x]

    # if there are any nans in any of the lists, use ppca
    if ppca is True:
        if np.isnan(np.vstack(x)).any():
            warnings.warn('Missing data: Inexact solution computed with PPCA (see https://github.com/allentran/pca-magic for details)')
            x = fill_missing(x)

    return x
