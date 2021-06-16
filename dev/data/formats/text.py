import six
import numpy as np
import os
import warnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from array import is_array, wrangle_array
from dataframe import is_dataframe
from null import is_empty
from ...core.configurator import get_default_options
from ...data.io import load


defaults = get_default_options()
decomposition_models = ['LatentDirichletAllocation', 'NMF']
text_vectorizers = ['CountVectorizer', 'TfidfVectorizer']
text_models = ['USE', 'LatentDirichletAllocation', 'NMF']
corpora = ['minipedia', #curated wikipedia dataset
           'wikipedia', #full wikipedia dataset
           'neurips', #corpus of NeurIPS articles
           'sotus', #corpus of State of the Union presidential addresses
           'khan', #TODO: add khan academy dataset from Tehut's thesis project
           'imdb', #movie reviews corpus
           ] #also see: https://github.com/huggingface/datasets/tree/master/datasets
use_corpora = [str(k) for k in defaults['corpora'].keys()]


def is_text(x):
    if type(x) == list:
        return np.all([is_text(t) for t in x])
    return (type(x) in six.string_types) or (type(x) == np.str_)


def to_str_list(x, encoding='utf-8'):
    def to_string(s):
        if type(s) == str:
            return s
        elif is_empty(s) or (s is None):
            return ''
        elif type(s) in [bytes, np.bytes_]:
            return s.decode(encoding)
        elif is_array(s) or is_dataframe(s) or (type(s) == list):
            if len(s) == 1:
                return to_string(s[0])
            else:
                return to_str_list(s, encoding=encoding)
        else:
            return str(s)

    if is_array(x) or (type(x) == list):
        return [to_string(s) for s in x]
    elif is_text(x):
        return [x]
    else:
        raise Exception('Unsupported data type: {type(x)}')


def get_corpus(c, encoding='utf-8'):
    if c in corpora:
        fname = os.path.join(eval(defaults['data']['datadir']), 'corpora', f'{c}.npy')
        if not os.path.exists(fname):
            if not os.path.exists(os.path.abspath(os.path.join(fname, os.pardir))):
                os.makedirs(os.path.abspath(os.path.join(fname, os.pardir)))
            corpus_words = to_str_list(load(c).data[0])

            np.save(fname, corpus_words)
            return corpus_words
        else:
            corpus_words = np.load(fname, allow_pickle=True)
            return to_str_list(corpus_words)
    else:
        if is_text(c):
            if type(c) == list:
                return c
            else:
                return [c]
        elif os.path.exists(c):
            return to_str_list([x[0] for x in np.load(c, allow_pickle=True).tolist()])
        else:
            raise Exception(f'Unknown corpus: {c}')


def vectorize_text(text, vectorizer='CountVectorizer', vocabulary=None, return_vocab=False):
    if not (type(text) == list):
        text = [text]
    assert is_text(text), f'Must vectorize a string or list of strings (given: {type(text)})'

    if type(vectorizer) in six.string_types:
        assert vectorizer in text_vectorizers, f'Text vectorizer must be a function or a member of {text_vectorizers}'
        vectorizer = eval(vectorizer)
    assert callable(vectorizer), f'Text vectorizer must be a function or a member of {text_vectorizers}'

    text2vec = vectorizer(max_df=eval(defaults['text']['max_df']),
                          min_df=eval(defaults['text']['min_df']),
                          stop_words=defaults['text']['stop_words'],
                          strip_accents=defaults['text']['strip_accents'],
                          lowercase=eval(defaults['text']['lowercase']),
                          vocabulary=vocabulary)
    vectorized_text = text2vec.fit_transform(text)

    if not return_vocab:
        return vectorized_text
    else:
        vocab = text2vec.get_feature_names()
        return vectorized_text, vocab


# %%

def get_text_model(corpus, model, vectorizer, n_components=50):
    if type(model) in six.string_types:
        assert model in text_models, f'Text model must be a function or a member of {text_models}'
        model = eval(model)
    assert callable(model), f'Text model must be a function or a member of {text_models}'

    if type(vectorizer) in six.string_types:
        assert vectorizer in text_vectorizers, f'Text vectorizer must be a function or a member of {text_vectorizers}'
        vectorizer = eval(vectorizer)
    assert callable(vectorizer), f'Text vectorizer must be a function or a member of {text_vectorizers}'

    if corpus in corpora:
        saveable = True
    else:
        if not os.path.exists(corpus):
            assert is_text(corpus), f'Corpus must be a list of strings, or one of {corpora}'
        saveable = False

    if saveable:
        fname = os.path.join(eval(defaults['data']['datadir']), 'text-models', model.__name__,
                             f'{corpus}-{vectorizer.__name__}-{n_components}.npz')
        if not os.path.exists(os.path.abspath(os.path.join(fname, os.pardir))):
            os.makedirs(os.path.abspath(os.path.join(fname, os.pardir)))

    if saveable and os.path.exists(fname):
        with np.load(fname, allow_pickle=True) as x:
            return {'vocab': x['vocab'].tolist(), 'model': x['model'].tolist()}
    else:
        corpus = get_corpus(corpus)
        vectorized_corpus, vocab = vectorize_text(corpus, vectorizer=vectorizer, return_vocab=True)

        if n_components == None:
            n_components = eval(defaults['text']['topics'])
        args = {'n_components': n_components,
                'max_iter': eval(defaults['text']['max_iter'])}

        if model.__name__ == 'NMF' and (args['n_components'] > len(corpus)):
            args['n_components'] = len(corpus)

        if model.__name__ == 'LatentDirichletAllocation':
            args['learning_method'] = defaults['text']['learning_method']
            args['learning_offset'] = eval(defaults['text']['learning_offset'])

        # return args, vectorized_corpus, vocab

        embeddings = model(**args).fit(vectorized_corpus)

        if saveable:
            np.savez(fname, vocab=vocab, model=embeddings)

        return {'vocab': vocab, 'model': embeddings}

    # %%


def text_vectorizer(text, model='USE', **kwargs):
    warnings.simplefilter('ignore')

    def USE(text, **kwargs):
        if 'USE_corpus' in kwargs.keys():
            corpus = kwargs['USE_corpus']
        else:
            corpus = defaults['corpora'][defaults['text']['USE_corpus']]

        model = hub.load(corpus)
        return np.array(model(text))

    def sklearn_vectorizer(text, model, **kwargs):
        if 'corpus' in kwargs.keys():
            corpus = kwargs['corpus']
        else:
            corpus = defaults['text']['corpus']

        assert (corpus in corpora) or is_text(corpus) or os.path.exists(corpus), f'Cannot use corpus: {corpus}'

        if 'vectorizer' in kwargs.keys():
            vecterizer = kwargs['vectorizer']
            kwargs.pop('vectorizer', None)
        else:
            vectorizer = defaults['text']['vectorizer']

        model = get_text_model(corpus, model, vectorizer)
        return model['model'].transform(vectorize_text(text, vectorizer=vectorizer, vocabulary=model['vocab']))

    assert (model in text_models) or (callable(model)), f'Unsupported model: {model}'
    if not (type(text) == list):
        text = [text]

    if callable(model):
        return model(text, **kwargs)
    elif model == 'USE':
        return USE(text, **kwargs)
    else:
        return sklearn_vectorizer(text, model, **kwargs)


def wrangle_text(data, **kwargs):
    return wrangle_array(text_vectorizer(data, **kwargs))



