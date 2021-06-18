import six
import numpy as np
import os
import warnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

from array import is_array, wrangle_array
from dataframe import is_dataframe
from null import is_empty
from ...core.configurator import get_default_options
from ...data.io import load


defaults = get_default_options()
sklearn_text_vectorizers = ['CountVectorizer', 'TfidfVectorizer']
sklearn_text_embeddings = ['LatentDirichletAllocation', 'NMF']
flair_text_embeddings = ['BytePairEmbeddings', 'CharacterEmbeddings', 'ELMoEmbeddings', 'FastTextEmbeddings',
                         'FlairEmbeddings', 'OneHotEmbeddings', 'PooledFlairEmbeddings', 'TransformerWordEmbeddings',
                         'WordEmbeddings', 'DocumentPoolEmbeddings', 'DocumentRNNEmbeddings',
                         'TransformerDocumentEmbeddings', 'SentenceTransformerDocumentEmbeddings', 'StackedEmbeddings']
corpora = ['minipedia',  # curated wikipedia dataset
           'wikipedia',  # full wikipedia dataset
           'neurips',    # corpus of NeurIPS articles
           'sotus',      # corpus of State of the Union presidential addresses
           'khan',       # TODO: add khan academy dataset from Tehut's thesis project
           'imdb',       # movie reviews corpus
           ]             # also see: https://github.com/huggingface/datasets/tree/master/datasets



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


def get_corpus(c):  # FIXME: needs debugging
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
        if (type(c) == str) and os.path.exists(c):
            # noinspection PyTypeChecker
            return to_str_list([x[0] for x in np.load(c, allow_pickle=True).tolist()])
        elif is_text(c):
            if type(c) == list:
                return c
            else:
                return [c]
        else:
            raise Exception(f'Unknown corpus: {c}')


def vectorize_text(text, vectorizer='CountVectorizer', vocabulary=None, return_model=False, **kwargs):
    if not (type(text) == list):
        text = [text]
    assert is_text(text), f'Must vectorize a string or list of strings (given: {type(text)})'

    if type(vectorizer) in six.string_types:
        assert vectorizer in sklearn_text_vectorizers, f'Text vectorizer must be a function or a member of {sklearn_text_vectorizers}'
        vectorizer = eval(vectorizer)
    assert callable(vectorizer), f'Text vectorizer must be a function or a member of {sklearn_text_vectorizers}'

    # noinspection PyCallingNonCallable
    text2vec = vectorizer(max_df=eval(defaults['text']['max_df']),
                          min_df=eval(defaults['text']['min_df']),
                          stop_words=defaults['text']['stop_words'],
                          strip_accents=defaults['text']['strip_accents'],
                          lowercase=eval(defaults['text']['lowercase']),
                          vocabulary=vocabulary)
    vectorized_text = text2vec.fit_transform(text)

    if return_model:
        return vectorized_text, {'model': text2vec, 'args': [], 'kwargs': {**{'vocabulary': vocabulary}, **kwargs}}
    else:
        return vectorized_text


def get_text_model(corpus, model, vectorizer, n_components=50):
    if type(model) in six.string_types:
        assert model in sklearn_text_embeddings, f'Text model must be a function or a member of {sklearn_text_embeddings}'
        model = eval(model)
    assert callable(model), f'Text model must be a function or a member of {sklearn_text_embeddings}'

    if type(vectorizer) in six.string_types:
        assert vectorizer in sklearn_text_vectorizers, f'Text vectorizer must be a function or a member of {sklearn_text_vectorizers}'
        vectorizer = eval(vectorizer)
    assert callable(vectorizer), f'Text vectorizer must be a function or a member of {sklearn_text_vectorizers}'

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

    # noinspection PyUnboundLocalVariable
    if saveable and os.path.exists(fname):
        with np.load(fname, allow_pickle=True) as x:
            return {'vocab': x['vocab'].tolist(), 'model': x['model'].tolist()}
    else:
        corpus = get_corpus(corpus)
        # noinspection PyTypeChecker
        vectorized_corpus, vocab = vectorize_text(corpus, vectorizer=vectorizer, return_vocab=True)

        if n_components is None:
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


def text_vectorizer(text, model='UniversalSentenceEncoder', return_model=False, **kwargs):
    warnings.simplefilter('ignore')

    # noinspection PyShadowingNames, PyPep8Naming, PyUnusedLocal
    def UniversalSentenceEncoder(text, **kwargs):
        if 'corpus' in kwargs.keys():
            corpus = kwargs.pop('corpus', None)
        else:
            corpus = defaults['corpora'][defaults['text']['universal_sentence_encoder_corpus']]

        use_model = hub.load(corpus)
        if return_model:
            return np.array(use_model(text)), {'model': use_model, 'args': [], 'kwargs': {'corpus': corpus, **kwargs}
        else:
            return np.array(use_model(text))

    # noinspection PyShadowingNames
    def sklearn_vectorizer(text, model, **kwargs):
        if 'corpus' in kwargs.keys():
            corpus = kwargs['corpus']
        else:
            corpus = defaults['text']['corpus']

        assert (corpus in corpora) or is_text(corpus) or os.path.exists(corpus), f'Cannot use corpus: {corpus}'

        if 'vectorizer' in kwargs.keys():
            vectorizer = kwargs['vectorizer']
            kwargs.pop('vectorizer', None)
        else:
            vectorizer = defaults['text']['vectorizer']

        # noinspection PyUnboundLocalVariable
        model = get_text_model(corpus, model, vectorizer)

        if
        return model['model'].transform(vectorize_text(text, vectorizer=vectorizer, vocabulary=model['vocab']))

    assert (model in sklearn_text_embeddings) or (callable(model)), f'Unsupported model: {model}'
    if not (type(text) == list):
        text = [text]

    if callable(model):
        # noinspection PyCallingNonCallable
        return model(text, **kwargs)
    elif model == 'USE':
        return USE(text, **kwargs)
    else:
        return sklearn_vectorizer(text, model, **kwargs)


def wrangle_text(data, return_model=False, **kwargs):
    if 'vectorize_kwargs' in kwargs.keys():
        vectorize_kwargs = kwargs.pop('vectorize_kwargs', None)
    else:
        vectorize_kwargs = {}

    if 'text_embedding_kwargs' in kwargs.keys():
        text_embedding_kwargs = kwargs.pop('text_embedding_kwargs', None)
    else:
        text_embedding_kwargs = {}

    text_vecs, vec_model = text_vectorizer(data, return_model=True, **vectorize_kwargs)
    text_embeddings, embedding_model = text_embedder(text_vecs, return_model=True, **text_embedding_kwargs)
    df, df_model = wrangle_array(text_embeddings, return_model=True, **kwargs)

    if return_model:
        steps = ['vec', 'embedding', 'df']
        model = {'model': [], 'args': [], 'kwargs': []}
        for s in steps:
            model['model'].append(eval(f'{s}_model'))
            model['args'].append(eval(f'{s}_model["args"]'))
            model['kwargs'].append(eval(f'{s}_model["kwargs"]'))
        return df, model
    else:
        return df
