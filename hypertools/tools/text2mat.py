import numpy as np
import inspect
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from .._shared.helpers import memoize
from .._shared.params import default_params
from .load import load

# vector models
vectorizer_models = {
    'CountVectorizer' : CountVectorizer,
    'TfidfVectorizer' : TfidfVectorizer
}

# text models
texts = {
    'LatentDirichletAllocation' : LatentDirichletAllocation,
    'NMF' : NMF,
}


@memoize
def text2mat(data, vectorizer='CountVectorizer',
             semantic='LatentDirichletAllocation', corpus='wiki'):
    """
    Turns a list of text samples into a matrix using a vectorizer and a text model

    Parameters
    ----------

    data : list (or list of lists) of text samples
        The text data to transform

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

    Returns
    ----------

    transformed data : list of numpy arrays
        The transformed text data
    """
    if semantic is None:
        semantic = 'LatentDirichletAllocation'
    if vectorizer is None:
        vectorizer = 'CountVectorizer'
    model_is_fit=False
    if corpus is not None:
        if corpus in ('wiki', 'nips', 'sotus',):
            if semantic == 'LatentDirichletAllocation' and vectorizer == 'CountVectorizer':
                semantic = load(corpus + '_model')
                vectorizer = None
                model_is_fit = True
            else:
                corpus = np.array(load(corpus).get_data())
        else:
            corpus = np.array([corpus])

    vtype = _check_mtype(vectorizer)
    if vtype == 'str':
        vectorizer_params = default_params(vectorizer)
    elif vtype == 'dict':
        vectorizer_params = default_params(vectorizer['model'], vectorizer['params'])
        vectorizer = vectorizer['model']
    elif vtype in ('class', 'class_instance'):
        if hasattr(vectorizer, 'fit_transform'):
            vectorizer_models.update({'user_model' : vectorizer})
            vectorizer = 'user_model'
        else:
            raise RuntimeError('Error: Vectorizer model must have fit_transform '
                               'method following the scikit-learn API. See here '
                               'for more details: '
                               'http://scikit-learn.org/stable/data_transforms.html')
    ttype = _check_mtype(semantic)
    if ttype == 'str':
        text_params = default_params(semantic)
    elif ttype == 'dict':
        text_params = default_params(semantic['model'], semantic['params'])
        semantic = semantic['model']
    elif ttype in ('class', 'class_instance'):
        if hasattr(semantic, 'fit_transform'):
            texts.update({'user_model' : semantic})
            semantic = 'user_model'
        else:
            raise RuntimeError('Text model must have fit_transform '
                               'method following the scikit-learn API. See here '
                               'for more details: '
                               'http://scikit-learn.org/stable/data_transforms.html')
    if vectorizer:
        if vtype in ('str', 'dict'):
            vmodel = vectorizer_models[vectorizer](**vectorizer_params)
        elif vtype == 'class':
            vmodel = vectorizer_models[vectorizer]()
        elif vtype == 'class_instance':
            vmodel = vectorizer_models[vectorizer]
    else:
        vmodel = None

    if semantic:
        if ttype in ('str', 'dict'):
            tmodel = texts[semantic](**text_params)
        elif ttype == 'class':
            tmodel = texts[semantic]()
        elif ttype == 'class_instance':
            tmodel = texts[semantic]
    else:
        tmodel = None

    if not isinstance(data, list):
        data = [data]

    if corpus is None:
        _fit_models(vmodel, tmodel, data, model_is_fit)
    else:
        _fit_models(vmodel, tmodel, corpus, model_is_fit)

    return _transform(vmodel, tmodel, data)


def _transform(vmodel, tmodel, x):
    split = np.cumsum([len(xi) for xi in x])[:-1]
    if vmodel is not None:
        x = np.vsplit(vmodel.transform(np.vstack(x).ravel()).toarray(), split)
    if tmodel is not None:
        if isinstance(tmodel, Pipeline):
            x = np.vsplit(tmodel.transform(np.vstack(x).ravel()), split)
        else:
            x = np.vsplit(tmodel.transform(np.vstack(x)), split)
    return [xi for xi in x]


def _fit_models(vmodel, tmodel, x, model_is_fit):
    if model_is_fit==True:
        return
    if vmodel is not None:
        try:
            check_is_fitted(vmodel, ['vocabulary_'])
        except NotFittedError:
            vmodel.fit(np.vstack(x).ravel())
    if tmodel is not None:
        try:
            check_is_fitted(tmodel, ['components_'])
        except NotFittedError:
            if isinstance(tmodel, Pipeline):
                tmodel.fit(np.vstack(x).ravel())
            else:
                tmodel.fit(vmodel.transform(np.vstack(x).ravel()))


def _check_mtype(x):
    if isinstance(x, str):
        return 'str'
    elif isinstance(x, dict):
        return 'dict'
    elif inspect.isclass(x):
        return 'class'
    elif isinstance(x, type(None)):
        return 'None'
    else:
        try:
            if inspect.isclass(type(x)):
                return 'class_instance'
        except:
            raise TypeError('Parameter must of type string, dict, class, or'
                            ' class instance.')
