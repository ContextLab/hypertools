import six
import numpy as np
import os
import warnings
from sklearn.feature_extraction import text
from sklearn import decomposition
from flair import embeddings
from flair.data import Sentence
from datasets import load_dataset, get_dataset_config_names, list_datasets
import requests
import io

from array import is_array, wrangle_array
from dataframe import is_dataframe
from null import is_empty
from ...core.configurator import get_default_options
from ...data.io import load

defaults = get_default_options()

# TODO: if a corpus is specified, the given model should be *trained* on the given corpus.
#   instructions 1: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md
#   instructions 2: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md
#   other instructions: https://huggingface.co/transformers/training.html
#   suggested heuristic:
#     - by default, only "update" the already trained model (much faster)
#     - if from_scratch=True, train the full model from scratch (print a warning that it'll take a REALLY long time)

# TODO: support text generation
#   link: https://huggingface.co/transformers/model_doc/gpt_neo.html

# TODO: in core.configurator.py, add a function for loading in default arguments to a given function and then replacing
#   any specified arguments that are passed in.  need one section in config.ini per function whose defaults are
#   specified.  this could also be a decorator-- e.g. pass in default args based on the function name and then update
#   with any given keyword args.  (if implemented as a decorator, it might fit better in decorate.py.)


def get_text_model(x):
    # noinspection PyShadowingNames
    def model_lookup(model_name, parent):
        try:
            return eval(f'{parent}.{model_name}')
        except AttributeError:
            return None

    for p in ['text', 'decomposition', 'embeddings']:
        module = model_lookup(x, p)
        if module is not None:
            return module, eval(p)
    return None, None


def get_corpus(dataset_name='wikipedia', config_name='20200501.en'):
    def get_formatter(s):
        return s[s.find('_'):(s.rfind('_') + 1)]

    # hypertools corpora
    hypertools_corpora = {
        'minipedia': '1mRNAZlTbZzSvV3tAQfSjNm587xdYKVkX',
        'neurips': '1Qo61vh2P3Rpb9PM1lyXb5M2iw7uB03uY',
        'sotus': '1uKJtxs-C0KDM2my0K6W2p0jCF6howg1y',
        'khan': '1KPhKxQlQrZHSPlCgky7K2bsfHlvJK039'}

    if dataset_name in hypertools_corpora.keys():
        return load(hypertools_corpora[dataset_name])['corpus']

    # Hugging-Face Corpus
    try:
        data = load_dataset(dataset_name, config_name)
    except FileNotFoundError:
        raise RuntimeError(f'Corpus not found: {dataset_name}.  Available corpora: {", ".join(list_datasets())}')
    except ValueError:
        raise RuntimeError(f'Configuration for {dataset_name} corpus not found: {config_name}. '
                           f'Available configurations: {", ".join(get_dataset_config_names(data_name))}')

    corpus = []
    for k in data.keys():
        for document in data[k].data['text']:
            corpus.append(' '.join([w if '_' not in w else w.replace(get_formatter(w), ' ')
                                    for w in str(document).split()]))
    return corpus


# noinspection PyShadowingNames
def apply_text_model(x, text, *args, return_model=False, **kwargs):
    if callable(x):
        return x(text)

    model, parent = get_text_model(x)
    if (model is None) or (parent is None):
        raise RuntimeError(f'unknown text processing module: {x}')

    if hasattr(parent, 'fit_transform'):  # scikit-learn model
        model = model(*args, **kwargs)
        transformed_text = model.fit_transform(text)
        if return_model:
            return transformed_text, {'model': model, 'args': args, 'kwargs': kwargs}
        return transformed_text
    elif hasattr(parent, 'embed'):        # flair model
        if 'embedding_args' in kwargs.keys():
            embedding_args = kwargs.pop('embedding_args', None)
        else:
            embedding_args = []

        if 'embedding_kwargs' in kwargs.keys():
            embedding_kwargs = kwargs.pop('embedding_kwargs', None)
        else:
            embedding_kwargs = {}

        model = model(*embedding_args, **embedding_kwargs)
        wrapped_text = Sentence(text, **kwargs)
        model.embed(wrapped_text)

        embeddings = np.empty(len(wrapped_text), len(wrapped_text[0].embedding))
        embeddings[:] = np.nan

        for i, token in enumerate(wrapped_text):
            if len(token.embedding) > 0:
                embeddings[i, :] = token.embedding

        if return_model:
            return embeddings, {'model': model, 'args': args,
                                'kwargs': {'embedding_args': embedding_args,
                                           'embedding_kwargs': embedding_kwargs,
                                           **kwargs}}
        else:
            return embeddings
    else:                                 # unknown model
        raise RuntimeError('Cannot apply text model: {model}')


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


# noinspection PyShadowingNames
def text_vectorizer(text, return_model=False, **kwargs):
    if 'model' not in kwargs.keys():
        if return_model:
            return text, {'model': [], 'args': [], 'kwargs': kwargs}
        else:
            return text
    else:
        model = kwargs.pop('model', None)
        if type(model) is str:
            model_name = model
        elif callable(model):
            model_name = model.__name__
        else:
            model_name = str(model)

        if model_name in defaults.keys():
            model_kwargs = defaults[model_name]
        else:
            model_kwargs = {}

        for k, v in kwargs.items():
            model_kwargs[k] = v

        return apply_text_model(x, text, *args, return_model=return_model, **model_kwargs)


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
        model = {'model': [], 'args': [], 'kwargs': {}}
        for s in steps:
            model['model'].append(eval(f'{s}_model'))
            model['args'].append(eval(f'{s}_model["args"]'))
            model['kwargs'].append(eval(f'{s}_model["kwargs"]'))
        return df, model
    else:
        return df
