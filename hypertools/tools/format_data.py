import warnings

import numpy as np

from .._shared.helpers import get_type


def format_data(x, vectorizer='CountVectorizer',
                semantic='LatentDirichletAllocation', corpus='wiki', ppca=True,
                text_align='hyper', impute=None):
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

    impute : str, dict, class, class instance or None
        If missing (NaN) values are present and `ppca` is True, this
        overrides the default PPCA fill with a different `hypertools.impute`
        model (e.g. 'Kalman', 'KNNImputer'; see `hypertools.impute.impute`
        for accepted forms). If None (default), missing values are filled
        with PPCA, matching the pre-1.0 behavior byte-for-byte.

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

    # a pandas Series is a single 1-D dataset (QC 2026-07: was rejected as
    # "unsupported"); a tuple is treated like a list of datasets.
    import pandas as pd
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    elif isinstance(x, tuple):
        x = list(x)

    # if x is not a list, make it one
    if not isinstance(x, list):
        x = [x]

    if all([isinstance(xi, str) for xi in x]):
        x = [x]
    # a FLAT list of numbers is a SINGLE 1-D dataset, not a list of scalar
    # "datasets" -- wrap it so get_type sees one array (QC 2026-07: mapping
    # get_type over the individual numbers raised "Unsupported data type",
    # even though the message advertises "List of numbers" as supported).
    elif len(x) > 0 and all(
            isinstance(xi, (int, float, np.number)) and not isinstance(xi, bool)
            for xi in x):
        x = [np.asarray(x, dtype=float)]

    # check data type for each element in list
    dtypes = list(map(get_type, x))

    # GH #132: DataFrames are consumed positionally downstream, so datasets
    # with the SAME named columns in a DIFFERENT order would silently
    # misalign features (dataset 2's column 'b' lands in dataset 1's 'a'
    # slot). When multiple DataFrames with named (non-default, non-duplicate)
    # columns are passed: reorder later ones to match the first's column
    # order when the column sets agree, and raise a clear error when they
    # don't. DataFrames with default integer columns (e.g. wrapped arrays)
    # keep their positional behavior.
    import pandas as pd
    named_df_idx = [
        i for i, d in enumerate(dtypes)
        if d == 'df'
        and not isinstance(x[i].columns, pd.RangeIndex)
        and not x[i].columns.duplicated().any()
    ]
    if len(named_df_idx) > 1:
        x = list(x)  # don't rearrange the caller's list in place
        canonical = list(x[named_df_idx[0]].columns)
        for i in named_df_idx[1:]:
            cols = list(x[i].columns)
            if cols == canonical:
                continue
            if set(cols) == set(canonical):
                warnings.warn(
                    f'dataset {i} has the same columns as dataset '
                    f'{named_df_idx[0]} but in a different order; reordering '
                    f'{cols} to match {canonical} so features align by name '
                    '(GH #132).'
                )
                x[i] = x[i][canonical]
            else:
                missing = set(canonical) - set(cols)
                extra = set(cols) - set(canonical)
                raise ValueError(
                    f'DataFrame columns do not match across datasets: dataset '
                    f'{i} is missing {sorted(map(str, missing))} and has '
                    f'unexpected {sorted(map(str, extra))} relative to dataset '
                    f'{named_df_idx[0]} (columns {canonical}). Features are '
                    'matched by column name; rename or subset the columns so '
                    'all datasets share the same set (GH #132).'
                )

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
        elif dtype == 'list_num':
            # a numeric-list dataset -> array (QC 2026-07: a raw python list hit
            # "'list' object has no attribute 'ndim'" at the reshape below)
            processed_x.append(np.asarray(x[i], dtype=float))
        else:
            processed_x.append(x[i])

    # reshape anything that is 0d or 1d into a 2d (observations x features)
    # array. QC 2026-07: a 0-d array like np.array(5) has ndim 0, so the old
    # `if i.ndim==1` branch left it untouched and it later raised the opaque
    # "tuple index out of range" on i.shape[0]; a scalar is now one observation
    # with one feature (a (1, 1) array), consistent with [5] -> (1, 1).
    if any([i.ndim <= 1 for i in processed_x]):
        processed_x = [np.reshape(i, (i.shape[0] if i.ndim == 1 else 1, 1))
                       if i.ndim <= 1 else i for i in processed_x]

    contains_text = any([dtype in ['list_str', 'str', 'arr_str'] for dtype in dtypes])
    contains_num = any([dtype in ['list_num', 'array', 'df', 'arr_num'] for dtype in dtypes])

    # fail fast with a CLEAR message on non-finite data (QC 2026-07): otherwise
    # infinities surface as sklearn's opaque "Input X contains infinity" and an
    # entirely-missing dataset surfaces as "zero-size array to reduction" deep
    # inside PPCA. Missing values (NaN) are still fine -- PPCA imputes them.
    if contains_num:
        for _arr, _dtype in zip(processed_x, dtypes):
            if _dtype not in ['list_num', 'array', 'df', 'arr_num']:
                continue
            _num = np.asarray(_arr, dtype=float)
            if _num.shape[0] == 0:
                raise ValueError(
                    'input has no observations (0 rows); there is nothing to '
                    'plot or analyze.')
            if np.isinf(_num).any():
                raise ValueError(
                    'input contains infinite values; remove or replace them '
                    '(e.g. with np.nan for missing entries) before plotting or '
                    'analysis.')
            if _num.size and np.isnan(_num).all():
                raise ValueError(
                    'input is entirely missing (all values are NaN); there is '
                    'nothing to plot or analyze.')

    # if there are any nans in any of the lists, use ppca
    if ppca is True:
        if contains_num:
            num_data = []
            for i,j in zip(processed_x, dtypes):
                if j in ['list_num', 'array', 'df', 'arr_num']:
                    num_data.append(i)
            # check for NaNs PER dataset rather than by vstacking them all --
            # datasets can legitimately have DIFFERENT column counts (the
            # canonical hyperalignment case: each subject aligns padded to a
            # common width), and vstacking them raised "all the input array
            # dimensions ... must match exactly" (QC 2026-07).
            if any(np.isnan(np.asarray(a, dtype=float)).any() for a in num_data):
                if impute is not None:
                    num_data = fill_missing(num_data, model=impute)
                else:
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


def fill_missing(x, model='PPCA'):
    """Fill missing values using `hypertools.impute` (default model='PPCA').

    Wraps a fit/transform sequence (stack -> fit -> transform -> split back
    into a list) via the `hypertools.impute` dispatcher. With the default
    `model='PPCA'`, this is kept byte-compatible with the pre-1.0 behavior;
    passing a different `model` (str/dict/class/instance) routes through
    that imputer instead (see `format_data`'s `impute` argument).
    """
    from ..impute.impute import impute as imputer

    filled = imputer(x, model=model)
    if not isinstance(filled, list):
        filled = [filled]
    return [np.asarray(f) for f in filled]
