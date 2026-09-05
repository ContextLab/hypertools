import warnings

import numpy as np
import pandas as pd

from .._shared.helpers import get_type


def _contains_text(el):
    """True if `el` is (or recursively contains) a str/bytes."""
    if isinstance(el, (str, bytes)):
        return True
    if isinstance(el, (list, tuple)):
        return any(_contains_text(sub) for sub in el)
    return False


def _contains_dataset(el):
    """True if `el` is (or recursively contains) an array/DataFrame/Series."""
    if isinstance(el, (np.ndarray, pd.DataFrame, pd.Series)):
        return True
    if isinstance(el, (list, tuple)):
        return any(_contains_dataset(sub) for sub in el)
    return False


def _flatten_dataset_groups(x):
    """Flatten nested lists/tuples of array/DataFrame/Series datasets into a
    flat list of datasets (release-1.0 audit, F08-plot-inputs-010), matching
    `hyp.plot()`'s nested-group flattening. Lists containing strings are left
    intact (nested string lists denote text corpora), as are lists of plain
    numbers (a single 1-D dataset)."""
    flat = []
    for el in x:
        if (isinstance(el, (list, tuple)) and _contains_dataset(el)
                and not _contains_text(el)):
            flat.extend(_flatten_dataset_groups(list(el)))
        else:
            flat.append(el)
    return flat


def _prepare_df(df, warn=True):
    """Convert DataFrame column dtypes that `df2mat` cannot handle
    (release-1.0 audit, F08-plot-inputs-003/-006), returning a copy when a
    conversion applies (the caller's DataFrame is never mutated):

    - Categorical columns are converted to object dtype so `df2mat`
      dummy-codes them exactly like object-dtype string columns
      (F08-003: `df['species'].astype('category')` used to crash with
      "could not convert string to float").
    - datetime columns (tz-naive or tz-aware, any unit) are converted to
      float seconds since the Unix epoch (1970-01-01 00:00:00 UTC), with
      NaT becoming NaN (missing data), and a UserWarning names the
      converted columns (F08-006: Timestamps used to crash with a leaked
      "float() argument must be a ... not 'Timestamp'" error). Set
      `warn=False` to suppress the warning (used by internal re-derivations
      of the same DataFrame, e.g. axis-label inference in `hyp.plot`).
    """
    cat_idx = [j for j, dt in enumerate(df.dtypes)
               if isinstance(dt, pd.CategoricalDtype)]
    dt_idx = [j for j, dt in enumerate(df.dtypes)
              if pd.api.types.is_datetime64_any_dtype(dt)]
    if not cat_idx and not dt_idx:
        return df
    df = df.copy()
    for j in cat_idx:
        df.isetitem(j, df.iloc[:, j].astype(object))
    if dt_idx:
        if warn:
            _names = [str(df.columns[j]) for j in dt_idx]
            warnings.warn(
                f"DataFrame column(s) {_names} contain datetime values; "
                'converting to float seconds since the Unix epoch '
                '(1970-01-01 00:00:00 UTC) so they can be analyzed '
                'numerically (NaT becomes NaN, i.e. missing data). '
                "Consider normalizing (e.g. normalize='zscore') if the "
                'epoch scale would dominate other features, or drop the '
                'column(s) if absolute times are not meaningful.')
        for j in dt_idx:
            col = df.iloc[:, j]
            if getattr(col.dtype, 'tz', None) is not None:
                col = col.dt.tz_convert('UTC').dt.tz_localize(None)
            vals = (col.astype('datetime64[ns]').astype('int64')
                    .astype('float64') / 1e9)
            vals[col.isna()] = np.nan
            df.isetitem(j, vals)
    return df


def format_data(x, vectorizer='CountVectorizer',
                semantic='LatentDirichletAllocation', corpus='wiki', ppca=True,
                text_align='hyper', impute=None):
    """
    Formats data into a list of numpy arrays

    This function is the standard input pass shared by hypertools' analysis
    and plotting functions: it wraps the input into a list of 2-D
    (observations x features) float arrays, converting pandas DataFrames
    (binarizing text columns via `df2mat`) and embedding text (strings /
    lists of strings) into numeric matrices via `text2mat` along the way.
    Missing (NaN) values are filled via PPCA (or the `impute=` override),
    and when text and numeric datasets with matching sample counts are
    mixed, they are aligned into a common space.

    Input conversions (release-1.0 audit):

    - pandas Series (top-level or inside a list) become 1-D datasets;
      tuples are treated like lists.
    - Nested lists/tuples of arrays/DataFrames (e.g. ``[[arr1, arr2]]``)
      are flattened into a flat list of datasets, matching `hyp.plot()`.
    - Lists of bools are numeric 0/1 datasets, like ``np.array([True, ...])``.
    - numpy MaskedArray masked entries are treated as MISSING data:
      converted to NaN (with a UserWarning) so they flow into the standard
      PPCA/`impute=` fill, never analyzed as real values.
    - DataFrame Categorical columns are dummy-coded exactly like
      object-dtype string columns (via `df2mat`); datetime columns are
      converted to float seconds since the Unix epoch, with NaT becoming
      NaN and a UserWarning naming the converted columns.

    Parameters
    ----------

    x : numpy array, dataframe, series, string or (mixed, possibly nested) list
        The data to convert

    vectorizer : str, dict, class or class instance
        The vectorizer to use. Built-in options are 'CountVectorizer' or
        'TfidfVectorizer'. String names resolve in scikit-learn -> gensim ->
        Hugging Face order (GH #198): unrecognized names are treated as
        Hugging Face sentence-transformers model ids, and if loading one
        fails, a ``ValueError`` names the offending value and the built-in
        options (so typos don't surface as raw network errors). To change
        default parameters, set to a dictionary
        e.g. {'model' : 'CountVectorizer', 'kwargs' : {'max_features' : 10}}
        (the legacy {'model', 'params'} form is also still accepted). See
        https://scikit-learn.org/stable/api/sklearn.feature_extraction.html
        for details. You can also specify your own vectorizer model as a class,
        or class instance.  With either option, the class must have a
        fit_transform method (see https://scikit-learn.org/stable/data_transforms.html).
        To set parameters, use the dict form (or a configured class
        instance); a bare class is instantiated with its defaults.

    semantic : str, dict, class or class instance
        Text model to use to transform text data. Built-in options are
        'LatentDirichletAllocation' or 'NMF' (default: LDA for count
        vectorizers; for embedding vectorizers -- gensim or Hugging Face
        model ids -- the semantic stage defaults to none and `corpus` is
        unused). To change default
        parameters, set to a dictionary e.g. {'model' : 'NMF', 'kwargs' :
        {'n_components' : 10}} (the legacy {'model', 'params'} form is also
        still accepted). See
        https://scikit-learn.org/stable/api/sklearn.decomposition.html
        for details on the two model options. You can also specify your own
        text model as a class, or class instance.  With either option, the class
        must have a fit_transform method (see
        https://scikit-learn.org/stable/data_transforms.html).
        To set parameters, use the dict form (or a configured class
        instance); a bare class is instantiated with its defaults.

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
    -------
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

    # an empty list holds NO datasets: fail fast with the same no-data error
    # the numeric path raises, BEFORE the vacuous all(...) check below could
    # route [] toward the text/LDA corpus pipeline (QC 2026-07 /
    # X2-error-quality-005).
    if len(x) == 0:
        raise ValueError(
            'input has no observations (0 rows); there is nothing to '
            'plot or analyze.')

    if all([isinstance(xi, str) for xi in x]):
        x = [x]
    # a FLAT list of numbers is a SINGLE 1-D dataset, not a list of scalar
    # "datasets" -- wrap it so get_type sees one array (QC 2026-07: mapping
    # get_type over the individual numbers raised "Unsupported data type",
    # even though the message advertises "List of numbers" as supported).
    # bools count as numbers (release-1.0 audit, F08-plot-inputs-013):
    # [True, False, True] is the same data as np.array([True, False, True]),
    # which has always been accepted; np.bool_ is listed explicitly because
    # it is neither an np.number subclass nor (numpy >= 2) a python bool.
    elif len(x) > 0 and all(
            isinstance(xi, (bool, int, float, np.number, np.bool_))
            for xi in x):
        x = [np.asarray(x, dtype=float)]

    # nested lists of datasets, e.g. [[arr1, arr2]], are flattened into a
    # flat list of datasets (release-1.0 audit, F08-plot-inputs-010) --
    # matching hyp.plot()'s nested-group flattening, minus the styling
    # semantics (format_data returns data, not a figure). Lists containing
    # strings are left intact (nested string lists denote text corpora),
    # as are lists of numbers (a single 1-D dataset, handled above).
    x = _flatten_dataset_groups(x)

    # per-dataset conversions (release-1.0 audit):
    # - a pandas Series inside a list is a 1-D dataset, like a top-level
    #   Series (converted above)
    # - a numpy MaskedArray's masked entries are MISSING data
    #   (F08-plot-inputs-009): np.asarray() silently drops the mask, so the
    #   invalid underlying values used to be analyzed/plotted as real data.
    #   Convert them to NaN (flowing into the standard missing-data path:
    #   PPCA imputation by default, impute= to override, ppca=False to keep
    #   the NaNs) and warn.
    x_converted = []
    for _i, _el in enumerate(x):
        if isinstance(_el, pd.Series):
            _el = _el.to_numpy()
        if isinstance(_el, np.ma.MaskedArray) and _el.dtype.kind in 'biufc':
            _n_masked = int(np.ma.count_masked(_el))
            if _n_masked:
                warnings.warn(
                    f'dataset {_i} is a numpy masked array with {_n_masked} '
                    'masked (invalid) entries; treating them as missing '
                    'data (converted to NaN and, by default, filled via '
                    'PPCA -- pass impute= to choose a different imputation '
                    'model, or ppca=False to leave them as NaN).')
            _el = np.ma.filled(_el.astype(float), np.nan)
        x_converted.append(_el)
    x = x_converted

    # check data type for each element in list; on failure, name the
    # offending dataset by its list index (release-1.0 audit,
    # F08-plot-inputs-008) -- e.g. [good_array, None] now points at
    # dataset 1 instead of raising the generic message alone.
    dtypes = []
    for _i, _el in enumerate(x):
        try:
            dtypes.append(get_type(_el))
        except TypeError as e:
            if len(x) > 1:
                raise TypeError(f'dataset {_i} of the input list: '
                                f'{e}') from None
            raise
        if dtypes[-1] == 'list_arr':
            # a list of arrays that survived the flattening above mixes
            # arrays with text (or other non-dataset elements) -- there is
            # no unambiguous way to interpret it as a single dataset
            # (F08-plot-inputs-010: it used to crash later with
            # "'list' object has no attribute 'ndim'").
            raise ValueError(
                f'dataset {_i} is a list that mixes numpy arrays with '
                'text or other non-array elements, which is ambiguous. '
                'Pass arrays and text corpora as separate top-level '
                'datasets (nested lists containing ONLY arrays/DataFrames '
                'are flattened into separate datasets automatically).')

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

        # convert text to numerical matrices. A typo'd vectorizer=/semantic=
        # string is indistinguishable up front from a Hugging Face
        # sentence-transformers model id (the third tier of the GH #198
        # sklearn -> gensim -> HuggingFace name-resolution order), so an
        # unknown name only fails once the HuggingFace download is
        # attempted -- translate that network-layer error (a raw 401
        # RepositoryNotFoundError with request IDs) into one that names the
        # offending kwarg and the built-in options (release-1.0 audit,
        # F08-plot-inputs-011).
        from .text2mat import (_spec_model_name,
                               _SKLEARN_VECTORIZER_NAMES,
                               _SKLEARN_SEMANTIC_NAMES,
                               _GENSIM_VECTORIZER_NAMES,
                               _GENSIM_SEMANTIC_NAMES)
        # membership is tested against the FROZEN built-in name sets, not
        # the live vectorizer_models/texts registries: text2mat's name
        # resolution inserts every unrecognized name into those registries
        # on first use, so a live-registry test stopped flagging a typo'd
        # name from its second use onward (and the raw HF error escaped
        # unwrapped).
        _unrecognized = []
        _vname = _spec_model_name(vectorizer)
        if (_vname is not None and _vname not in _SKLEARN_VECTORIZER_NAMES
                and _vname not in _GENSIM_VECTORIZER_NAMES):
            _unrecognized.append(
                ('vectorizer', _vname,
                 "'CountVectorizer', 'TfidfVectorizer' (scikit-learn); "
                 "'Word2Vec', 'Doc2Vec', 'FastText' (gensim)"))
        _sname = _spec_model_name(semantic)
        if (_sname is not None and _sname not in _SKLEARN_SEMANTIC_NAMES
                and _sname not in _GENSIM_SEMANTIC_NAMES):
            _unrecognized.append(
                ('semantic', _sname,
                 "'LatentDirichletAllocation', 'NMF' (scikit-learn); "
                 "'LdaModel', 'LsiModel', 'HdpModel' (gensim)"))
        try:
            text_data = text2mat(text_data, **text_args)
        except (OSError, ImportError) as e:
            # RepositoryNotFoundError (unknown HF id) and offline
            # connection errors are both OSError subclasses; when the HF
            # tier itself is not installed (no pydata-wrangler[hf]),
            # datawrangler instead raises ModuleNotFoundError (an
            # ImportError) before any network call -- every one of these
            # means "the unrecognized name fell through to the Hugging
            # Face tier and failed there", so all are rewrapped into the
            # same clear ValueError. Only rewrap when a non-built-in
            # string name was in play (a genuine sklearn/gensim failure
            # re-raises untouched).
            _blamed = ([u for u in _unrecognized if u[1] in str(e)]
                       or _unrecognized)
            if _blamed:
                _msgs = '; '.join(
                    f"{kw}='{name}' is not a built-in model name "
                    f'(built-ins: {builtins})'
                    for kw, name, builtins in _blamed)
                raise ValueError(
                    f'{_msgs}. Unrecognized names are treated as Hugging '
                    'Face sentence-transformers model ids (GH #198), and '
                    'loading this one failed (see the chained error for '
                    'details). Check the spelling if you meant a built-in '
                    'model, or the model id and your network connection '
                    'if you meant a Hugging Face model.') from e
            raise

    # replace the text data with transformed data
    processed_x = []
    textidx=0
    for i, dtype in enumerate(dtypes):
        if dtype in ['list_str', 'str', 'arr_str']:
            processed_x.append(text_data[textidx])
            textidx+=1
        elif dtype == 'df':
            # convert Categorical/datetime columns first (release-1.0
            # audit, F08-plot-inputs-003/-006) -- see _prepare_df.
            processed_x.append(df2mat(_prepare_df(x[i])))
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

    # reject >2-D arrays with a clear message (QC 2026-07 / F15-analyze-012):
    # they previously slipped through and were either silently axis-mangled
    # (constant data came back transposed) or crashed deep inside normalize
    # with an opaque "truth value of an array" error. fMRI users routinely
    # hold 3-D (or 4-D) arrays; tell them exactly what shape is expected.
    for _i, _arr in enumerate(processed_x):
        if getattr(_arr, 'ndim', 2) > 2:
            raise ValueError(
                f'each dataset must be a 2-D (observations x features) '
                f'array; dataset {_i} has {_arr.ndim} dimensions (shape '
                f'{tuple(_arr.shape)}). To analyze or plot multiple '
                'datasets, pass them as a list of 2-D arrays.')

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
                    warnings.warn('Missing data: filling missing values '
                                  'with PPCA (observed values are '
                                  'preserved exactly; only the NaN '
                                  'entries are reconstructed). Pass '
                                  'impute= to choose a different '
                                  'imputation model -- see '
                                  'hypertools.impute.')
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
        elif len(set(i.shape[1] for i in processed_x)) > 1:
            # mixed text + numeric datasets whose sample counts differ
            # CANNOT be auto-aligned to a common space, and (text having
            # been embedded to its own topic-vector width) their mismatched
            # column counts used to surface later as a bare numpy/hypertools
            # column-count error that never mentioned text or alignment
            # (release-1.0 audit, D08-tutorials-analysis-012 /
            # D05-gallery-data-text-013). format_data itself still returns
            # the per-dataset matrices (formatting mixed-width lists is a
            # documented standalone use), but WARN with the real reason so
            # any downstream shared-feature-space error is explained.
            _counts = [
                f"dataset {i}: {'text' if j in ('list_str', 'str', 'arr_str') else 'numeric'}, "
                f'{arr.shape[0]} sample(s)'
                for i, (arr, j) in enumerate(zip(processed_x, dtypes))]
            warnings.warn(
                'mixed text and numeric datasets were passed with '
                f"DIFFERENT sample counts ({'; '.join(_counts)}), so they "
                'cannot be auto-aligned to a common space (alignment '
                'requires one text sample per numeric observation). The '
                'datasets keep their own feature dimensionalities (text '
                'embeds to its own topic-vector width), which most '
                'hypertools analyses require to match.', UserWarning)

    return processed_x


def fill_missing(x, model='PPCA'):
    """Fill missing values using `hypertools.impute` (default model='PPCA').

    Wraps a fit/transform sequence (stack -> fit -> transform -> split back
    into a list) via the `hypertools.impute` dispatcher. With the default
    `model='PPCA'`, missing entries are filled with the PPCA
    reconstruction while every observed (non-NaN) value is preserved
    exactly (see `hypertools.impute.ppca`); passing a different `model`
    (str/dict/class/instance) routes through that imputer instead (see
    `format_data`'s `impute` argument).
    """
    from ..impute.impute import impute as imputer

    filled = imputer(x, model=model)
    if not isinstance(filled, list):
        filled = [filled]
    return [np.asarray(f) for f in filled]
