import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.pipeline import Pipeline
from .._shared.helpers import format_data

def text2mat(data, vectorizer='count', vectorizer_params=None, text_model='LDA',
             text_params=None, ndims=20):
    """
    Turns a list of text samples into a matrix using a vectorizer and a text model

    Parameters
    ----------

    data : list (or list of lists) of text samples
        The text data to transform

    vectorizer : str
        The vectorizer to use. Can be count or tfidf.  See
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
        for details

    vectorizer_params : dict
        Parameters for vectorizer model. See link above for details

    text_model : str
        Text model to use to transform the data. Can be LDA, NMF or None
        (default: LDA). If None, the text will be vectorized but not modeled. See http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
        for details on the two model options

    text_params : dict
        Parameters for text model. See link above for details

    ndims : int
        The number of components to estimate in the text model

    Returns
    ----------

    transformed data : list of numpy arrays
        The transformed text data
    """

    # subfunction to loop over arrays
    def transform_list(x, model, text_model):
        split = np.cumsum([len(xi) for xi in x])[:-1]
        if text_model is None:
            x_r = np.vsplit(model.fit_transform(np.vstack(x).ravel()).toarray(), split)
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

    # text models
    text_models = {
        'LDA' : LatentDirichletAllocation,
        'NMF' : NMF
    }

    # vector params
    if vectorizer_params is None:
        vectorizer_params = {}

    # text params
    if text_params is None:
        text_params = {}

    # update text dict with ndims
    text_params.update({'n_components' : ndims})

    # if no text model, just return vectorized text, else, return text fit to model
    if text_model is None:
        model = vectorizer_models[vectorizer](**vectorizer_params)
    else:
        vmodel = vectorizer_models[vectorizer](**vectorizer_params)
        tmodel = text_models[text_model](**text_params)
        model = Pipeline([(vectorizer, vmodel),
                          (text_model, tmodel)])

    # format data into list of arrays
    x = format_data(data)

    return transform_list(x, model, text_model)
