import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.pipeline import Pipeline
from .._shared.helpers import format_data

def text2mat(data, vectorizer='count', vectorizer_params=None, text_model='LDA',
             text_params=None, n_components=20, fit_model=False):
    """
    Turns a list of text samples into a matrix using a vectorizer and a text model

    Parameters
    ----------

    data : list (or list of lists) of text samples
        The text data to transform

    vectorizer : str, class or class instance
        The vectorizer to use. Can be count or tfidf.  See
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
        for details. You can also specify your own vectorizer model as a class,
        or class instance.  With either option, the class must have a
        fit_transform method (see here: http://scikit-learn.org/stable/data_transforms.html).
        If a class, pass any parameters as a dictionary to vectorizer_params. If
        a class instance, no parameters can be passed.

    vectorizer_params : dict
        Parameters for vectorizer model. See link above for details

    text_model : str, class or class instance
        Text model to use to transform the data. Can be LDA, NMF or None
        (default: LDA). If None, the text will be vectorized but not modeled. See http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
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
    def transform_list(x, model, text_model, fit_model):
        split = np.cumsum([len(xi) for xi in x])[:-1]
        if text_model is None:
            if fit_model:
                x_r = np.vsplit(model.transform(np.vstack(x).ravel()).toarray(), split)
            else:
                x_r = np.vsplit(model.fit_transform(np.vstack(x).ravel()).toarray(), split)
        else:
            if fit_model:
                x_r = np.vsplit(model.transform(np.vstack(x).ravel()), split)
            else:
                x_r = np.vsplit(model.fit_transform(np.vstack(x).ravel()), split)

        if len(x)>1:
            return [xi for xi in x_r]
        else:
            return [x_r[0]]

    # vector models
    vectorizer_models = {
        'count' : CountVectorizer,
        'tfidf' : TfidfVectorizer
    }

    # support user defined vectorizers
    if hasattr(vectorizer, '__class__'):
        if hasattr(vectorizer, 'fit_transform'):
            vectorizer_models.update({'user_model' : vectorizer})
            vectorizer = 'user_model'
        else:
            raise RuntimeError('Error: Vectorizer model must have fit_transform '
                               'method following the scikit-learn API. See here '
                               'for more details: '
                               'http://scikit-learn.org/stable/data_transforms.html')

    # text models
    text_models = {
        'LDA' : LatentDirichletAllocation,
        'NMF' : NMF
    }

    # support user defined text model
    if hasattr(text_model, '__class__'):
        if hasattr(text_model, 'fit_transform'):
            text_models.update({'user_model' : text_model})
            text = 'user_model'
        else:
            raise RuntimeError('Error: Text model must have fit_transform '
                               'method following the scikit-learn API. See here '
                               'for more details: '
                               'http://scikit-learn.org/stable/data_transforms.html')


    # vector params
    if vectorizer_params is None:
        vectorizer_params = {}

    # text params
    if text_params is None:
        text_params = {}

    # update text dict with n_topics
    text_params.update({'n_components' : n_topics})

    # if it has a __name__, it is a class (not a class instance)
    if hasattr(vectorizer_models[vectorizer], '__name__'):
        model = vectorizer_models[vectorizer](**vectorizer_params)
    else:
        # otherwise, its a class instance so don't iniatilize it
        model = vectorizer_models[vectorizer]

    # if no text model, just return vectorized text, else, return text fit to model
    if text_model is not None:

        # if it has a __name__, it is a class (not a class instance)
        if hasattr(text_models[text_model], '__name__'):
            tmodel = text_models[text_model](**text_params)
        else:
            # otherwise, its a class instance so don't iniatilize it
            tmodel = text_models[text_model]

        # create a vector->text pipeline
        model = Pipeline([(vectorizer, model),
                          (text_model, tmodel)])

    # format data into list of arrays
    x = format_data(data)

    return transform_list(x, model, text_model, fit_model)
