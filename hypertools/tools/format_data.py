import numpy as np
import pandas as pd
import warnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from .._externals.ppca import PPCA
from .._shared.params import default_params
import six

def format_data(x, vectorizer='CountVectorizer', semantic='wiki', corpus=None,
                ppca=True, text_align='hyper'):
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

    vectorizer : str, dict, class or class instance
        The vectorizer to use for text data. Can be CountVectorizer or
        TfidfVectorizer.  See
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
        for details. You can also specify your own vectorizer model as a class,
        or class instance.  With either option, the class must have a
        fit_transform method (see here: http://scikit-learn.org/stable/data_transforms.html).
        If a class, pass any parameters as a dictionary to vectorizer_params. If
        a class instance, no parameters can be passed.

    semantic : str, dict, class or class instance
        Text model to use to transform text data. Can be
        LatentDirichletAllocation, NMF or None (default: LDA).
        If None, the text will be vectorized but not modeled. See http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
        for details on the two model options. You can also specify your own
        text model as a class, or class instance.  With either option, the class
        must have a fit_transform method (see here:
        http://scikit-learn.org/stable/data_transforms.html).
        If a class, pass any parameters as a dictionary to text_params. If
        a class instance, no parameters can be passed. By default, this is set to
        'wiki', which is a prefit model trained on sample of wikipedia articles.

    corpus : list (or list of lists) of text samples or 'wiki'
        Text to use to fit the semantic model (optional). Note: if you pass this
        parameter with an already-fit-model, corpus will be ignored. If 'wiki',
        corpus will be set to a list of sampled wikipedia articles (same
        articles used to fit the wiki model).

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

    from ..tools.text2mat import text2mat

    def get_type(data):
        """
        Checks what the data type is and returns it as a string label
        """
        if isinstance(data, list):
            if isinstance(data[0], (six.string_types, six.text_type, six.binary_type)):
                return 'list_str'
            elif isinstance(data[0], (int, float)):
                return 'list_num'
        elif isinstance(data, np.ndarray):
            return 'array'
        elif isinstance(data, pd.DataFrame):
            return 'df'
        elif isinstance(data, str):
            return 'str'
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
    if any(map(lambda x: x in ['list_str', 'str'], dtypes)):

        # default text args
        text_args = {
            'vectorizer' : vectorizer,
            'semantic' : semantic,
            'corpus' : corpus
        }

        # filter text data
        text_data = []
        for i,j in zip(x, dtypes):
            if j in ['list_str', 'str']:
                text_data.append(np.array(i).reshape(-1, 1))

        # convert text to numerical matrices
        text_data = text2mat(text_data, **text_args)

    # replace the text data with transformed data
    processed_x = []
    textidx=0
    for i, dtype in enumerate(dtypes):
        if dtype in ['list_str', 'str']:
            processed_x.append(text_data[textidx])
            textidx+=1
        elif dtype is 'df':
            processed_x.append(df2mat(x[i]))
        else:
            processed_x.append(x[i])

    # reshape anything that is 1d
    if any([i.ndim<=1 for i in processed_x]):
        processed_x = [np.reshape(i,(i.shape[0],1)) if i.ndim==1 else i for i in processed_x]

    contains_text = any([dtype in ['list_str', 'str'] for dtype in dtypes])
    contains_num = any([dtype in ['list_num', 'array', 'df'] for dtype in dtypes])

    # if there are any nans in any of the lists, use ppca
    if ppca is True:
        if contains_num:
            num_data = []
            for i,j in zip(processed_x, dtypes):
                if j in ['list_num', 'array', 'df']:
                    num_data.append(i)
            if np.isnan(np.vstack(num_data)).any():
                warnings.warn('Missing data: Inexact solution computed with PPCA (see https://github.com/allentran/pca-magic for details)')
                num_data = fill_missing(num_data)
                x_temp = []
                for dtype in dtypes:
                    if dtype in ['list_str', 'str']:
                        x_temp.append(text_data.pop(0))
                    elif dtype in ['list_num', 'array', 'df']:
                        x_temp.append(num_data.pop(0))
                processed_x = x_temp

    # if input data contains both text and numerical data
    if contains_num and contains_text:

        # and if they have the same number of samples
        if np.unique(np.array([i.shape[0] for i, j in zip(processed_x, dtypes)])).shape[0]==1:

            from .align import align as aligner

            # align the data
            warnings.warn('Numerical and text data with same number of '
                          'samples detected.  Aligning data to a common space.')
            processed_x = aligner(processed_x, align=text_align, format_data=False)

    return processed_x
