import numpy as np
import inspect
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from .._shared.helpers import memoize
from .format_data import format_data

# vector models
vectorizer_models = {
    'CountVectorizer' : CountVectorizer,
    'TfidfVectorizer' : TfidfVectorizer
}

# text models
texts = {
    'LatentDirichletAllocation' : LatentDirichletAllocation,
    'NMF' : NMF
}

@memoize
def text2mat(data, vectorizer='CountVectorizer', vectorizer_params=None,
             text='LatentDirichletAllocation', text_params=None,
             n_components=20, fit_model=False):
    """
    Turns a list of text samples into a matrix using a vectorizer and a text model

    Parameters
    ----------

    data : list (or list of lists) of text samples
        The text data to transform

    vectorizer : str, class or class instance
        The vectorizer to use. Can be CountVectorizer or TfidfVectorizer.  See
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
        for details. You can also specify your own vectorizer model as a class,
        or class instance.  With either option, the class must have a
        fit_transform method (see here: http://scikit-learn.org/stable/data_transforms.html).
        If a class, pass any parameters as a dictionary to vectorizer_params. If
        a class instance, no parameters can be passed.

    vectorizer_params : dict
        Parameters for vectorizer model. See link above for details

    text : str, class or class instance
        Text model to use to transform the data. Can be
        LatentDirichletAllocation, NMF or None (default: LDA).
        If None, the text will be vectorized but not modeled. See http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
        for details on the two model options. You can also specify your own
        text model as a class, or class instance.  With either option, the class
        must have a fit_transform method (see here:
        http://scikit-learn.org/stable/data_transforms.html).
        If a class, pass any parameters as a dictionary to text_params. If
        a class instance, no parameters can be passed.

    text_params : dict
        Parameters for text model. See link above for details

    n_components : int
        The number of components to estimate in the text model

    Returns
    ----------

    transformed data : list of numpy arrays
        The transformed text data
    """

    # subfunction to loop over arrays
    def transform_list(x, model, fit_model):
        split = np.cumsum([len(xi) for xi in x])[:-1]
        if text is None:
            if fit_model:
                x_r = np.vsplit(model.transform(np.vstack(x).ravel()).toarray(), split)
            else:
                x_r = np.vsplit(model.fit_transform(np.vstack(x).ravel()).toarray(), split)
        else:
            if fit_model:
                x_r = np.vsplit(model.transform(np.vstack(x).ravel()), split)
            else:
                x_r = np.vsplit(model.fit_transform(np.vstack(x).ravel()), split)

        return [xi for xi in x_r]

    # check the type of the param
    def check_mtype(x):
        if type(x) is str:
            return 'str'
        elif type(x) is dict:
            return 'dict'
        elif inspect.isclass(x):
            return 'class'
        elif x is None:
            return 'None'
        else:
            try:
                if inspect.isclass(type(x)):
                    return 'class_instance'
            except:
                raise TypeError('Parameter must of type string, dict, class, or'
                                ' class instance.')

    # check the type of the vectorizer model
    vtype = check_mtype(vectorizer)

    # support user defined vectorizer model
    if vtype in ('class', 'class_instance'):
        if hasattr(vectorizer, 'fit_transform'):
            vectorizer_models.update({'user_model' : vectorizer})
            vectorizer = 'user_model'
        else:
            raise RuntimeError('Error: Vectorizer model must have fit_transform '
                               'method following the scikit-learn API. See here '
                               'for more details: '
                               'http://scikit-learn.org/stable/data_transforms.html')

    # check the type of the text model
    ttype = check_mtype(text)

    # support user defined text model
    if ttype in ('class', 'class_instance'):
        if hasattr(text, 'fit_transform'):
            texts.update({'user_model' : text})
            text = 'user_model'
        else:
            raise RuntimeError('Error: Text model must have fit_transform '
                               'method following the scikit-learn API. See here '
                               'for more details: '
                               'http://scikit-learn.org/stable/data_transforms.html')

    # initialize params
    if vectorizer_params is None:
        vectorizer_params = {}
    if text_params is None:
        text_params = {}
    text_params.update({'n_components' : n_components})

    if vectorizer:
        # intialize vectorizer model
        if vtype in ('str', 'dict', 'class'):
            vmodel = vectorizer_models[vectorizer](**vectorizer_params)
        elif vtype is 'class_instance':
            # otherwise, its a class instance so don't iniatilize it
            vmodel = vectorizer_models[vectorizer]

    if text:
        # initialize text model
        if ttype in ('str', 'dict', 'class'):
            tmodel = texts[text](**text_params)
        elif ttype is 'class_instance':
            # otherwise, its a class instance so don't iniatilize it
            tmodel = texts[text]

    # if both vectorizer and text model, put them in a pipeline
    if vectorizer and text:
        model = Pipeline([(vectorizer, vmodel),
                          (text, tmodel)])
    elif vectorizer:
        model = vmodel
    else:
        model = tmodel

    if type(data) is not list:
        data = [data]

    return transform_list(data, model, fit_model)
