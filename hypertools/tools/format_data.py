import warnings

import numpy as np

from .._externals.ppca import PPCA
from .._shared.helpers import get_type


def format_data(x, vectorizer='CountVectorizer',
                semantic='LatentDirichletAllocation', corpus='wiki', ppca=True, text_align='hyper'):
    """
    Formats data into a list of numpy arrays

    This function is useful to identify rows of your array that contain missing
    data or nans.  The returned indices can be used to remove the rows with
    missing data, or label the missing data points that are interpolated
    using PPCA.

    Parameters
    ----------

    x : numpy array, dataframe, string or (mixed) list
        The data to convert

    vectorizer : str, dict, class or class instance
        The vectorizer to use. Built-in options are 'CountVectorizer' or
        'TfidfVectorizer'. To change default parameters, set to a dictionary
        e.g. {'model' : 'CountVectorizer', 'params' : {'max_features' : 10}}. See
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
        for details. You can also specify your own vectorizer model as a class,
        or class instance.  With either option, the class must have a
        fit_transform method (see here: http://scikit-learn.org/stable/data_transforms.html).
        If a class, pass any parameters as a dictionary to vectorizer_params. If
        a class instance, no parameters can be passed.

    semantic : str, dict, class or class instance
        Text model to use to transform text data. Built-in options are
        'LatentDirichletAllocation' or 'NMF' (default: LDA). To change default
        parameters, set to a dictionary e.g. {'model' : 'NMF', 'params' :
        {'n_components' : 10}}. See
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
        for details on the two model options. You can also specify your own
        text model as a class, or class instance.  With either option, the class
        must have a fit_transform method (see here:
        http://scikit-learn.org/stable/data_transforms.html).
        If a class, pass any parameters as a dictionary to text_params. If
        a class instance, no parameters can be passed.

    corpus : list (or list of lists) of text samples or 'wiki', 'nips', 'sotus'.
         Text to use to fit the semantic model (optional). If set to 'wiki', 'nips'
         or 'sotus' and the default semantic and vectorizer models are used, a
         pretrained model will be loaded which can save a lot of time.

    ppca : bool
        Performs PPCA to fill in missing values (default: True)

    text_align : str
        Alignment algorithm to use when both text and numerical data are passed.
        If numerical arrays have the same shape, and the text data contains the
        same number of samples, the text and numerical data are automatically
        aligned to a common space. Example use case: an array of movie frames
        (frames by pixels) and text descriptions of the frame.  In this case,
        the movie and text will be automatically aligned to the same space
        (default: hyperalignment).

    Returns
    ----------
    data : list of numpy arrays
        A list of formatted arrays
    """

    # not sure why i needed to import here, but its the only way I could get it to work
    from .df2mat import df2mat
    from .text2mat import text2mat
    from ..datageometry import DataGeometry

    # if x is not a list, make it one
    if not isinstance(x, list):
        x = [x]

    if all([isinstance(xi, str) for xi in x]):
        x = [x]

    # check data type for each element in list
    dtypes = list(map(get_type, x))

    # handle text data:
    if any(map(lambda x: x in ['list_str', 'str', 'arr_str'], dtypes)):

        # default text args
        text_args = {
            'vectorizer' : vectorizer,
            'semantic' : semantic,
            'corpus' : corpus
        }

        # filter text data
        text_data = []
        for i,j in zip(x, dtypes):
            if j in ['list_str', 'str', 'arr_str']:
                text_data.append(np.asarray(i, dtype=object).reshape(-1, 1))
        # convert text to numerical matrices
        text_data = text2mat(text_data, **text_args)

    # replace the text data with transformed data
    processed_x = []
    textidx=0
    for i, dtype in enumerate(dtypes):
        if dtype in ['list_str', 'str', 'arr_str']:
            processed_x.append(text_data[textidx])
            textidx+=1
        elif dtype == 'df':
            processed_x.append(df2mat(x[i]))
        elif dtype == 'geo':
            text_args = {
                'vectorizer' : vectorizer,
                'semantic' : semantic,
                'corpus' : corpus
            }
            for j in format_data(x[i].get_data(), **text_args):
                processed_x.append(j)
        else:
            processed_x.append(x[i])

    # reshape anything that is 1d
    if any([i.ndim<=1 for i in processed_x]):
        processed_x = [np.reshape(i,(i.shape[0],1)) if i.ndim==1 else i for i in processed_x]

    contains_text = any([dtype in ['list_str', 'str', 'arr_str'] for dtype in dtypes])
    contains_num = any([dtype in ['list_num', 'array', 'df', 'arr_num'] for dtype in dtypes])

    # if there are any nans in any of the lists, use ppca
    if ppca is True:
        if contains_num:
            num_data = []
            for i,j in zip(processed_x, dtypes):
                if j in ['list_num', 'array', 'df', 'arr_num']:
                    num_data.append(i)
            if np.isnan(np.vstack(num_data)).any():
                warnings.warn('Missing data: Inexact solution computed with PPCA (see https://github.com/allentran/pca-magic for details)')
                num_data = fill_missing(num_data)
                x_temp = []
                for dtype in dtypes:
                    if dtype in ['list_str', 'str', 'arr_str']:
                        x_temp.append(text_data.pop(0))
                    elif dtype in ['list_num', 'array', 'df', 'arr_num']:
                        x_temp.append(num_data.pop(0))
                processed_x = x_temp

    # if input data contains both text and numerical data
    if contains_num and contains_text:

        # and if they have the same number of samples
        if np.unique(np.array([i.shape[0] for i, j in zip(processed_x, dtypes)])).shape[0] == 1:

            from .align import align as aligner

            # align the data
            warnings.warn('Numerical and text data with same number of '
                          'samples detected.  Aligning data to a common space.')
            processed_x = aligner(processed_x, align=text_align, format_data=False)

    return processed_x


def fill_missing(x):
    """Fill missing values using PPCA"""
    # ppca if missing data
    m = PPCA()
    x_stacked = np.vstack(x)
    m.fit(data=x_stacked)
    x_pca = m.transform()

    # if the whole row is missing, return nans
    all_missing = [idx for idx, a in enumerate(x_stacked) if np.all(np.isnan(a))]
    if len(all_missing)>0:
        for i in all_missing:
            x_pca[i, :] = np.nan

    # get the original lists back
    if len(x)>1:
        x_split = np.cumsum([i.shape[0] for i in x][:-1])
        return list(np.split(x_pca, x_split, axis=0))
    else:
        return [x_pca]
