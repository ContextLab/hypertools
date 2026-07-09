import numpy as np
import inspect
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from .._shared.params import default_params
from ..io.load import load

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

# GH #198 name-resolution order for vectorizer=/semantic= string specs:
# scikit-learn (above) -> gensim (below, lazily imported) -> Hugging Face
# (data-wrangler fallback, see `_hf_fallback_model`). Keys match gensim's
# own class names, per Jeremy's directive on GH #198; values are the
# corresponding wrapper class names in `hypertools.tools.gensim_models`.
_GENSIM_VECTORIZER_NAMES = {
    'Word2Vec': 'Word2VecVectorizer',
    'Doc2Vec': 'Doc2VecVectorizer',
    'FastText': 'FastTextVectorizer',
}
_GENSIM_SEMANTIC_NAMES = {
    'LdaModel': 'LdaVectorizer',
    'LsiModel': 'LsiVectorizer',
    'HdpModel': 'HdpVectorizer',
}


def _gensim_model_for(name, kind):
    """Look up a gensim wrapper class by its gensim-familiar name.

    Parameters
    ----------

    name : str
        A model name, e.g. 'Word2Vec' or 'LdaModel'.

    kind : 'vectorizer' or 'semantic'
        Which family of gensim wrappers is eligible for `name`.

    Returns
    ----------

    cls : class or None
        The matching wrapper class from `hypertools.tools.gensim_models`
        (e.g. `Word2VecVectorizer` for `name='Word2Vec'`), or None if
        `name` is not a recognized gensim model name for `kind`.

    Raises
    ----------

    ImportError
        If `name` is a recognized gensim model name but the optional
        `gensim` dependency is not installed. Names the pip extra needed
        to fix it (`pip install "hypertools[gensim]"`).
    """
    names = (_GENSIM_VECTORIZER_NAMES if kind == 'vectorizer'
             else _GENSIM_SEMANTIC_NAMES)
    if name not in names:
        return None
    from . import gensim_models  # lazy: raises a friendly ImportError
    return getattr(gensim_models, names[name])


def _hf_fallback_model(name):
    """Wrap an unresolved vectorizer=/semantic= string as a pretrained
    Hugging Face sentence-transformers embedding model -- the third and
    final tier of the sklearn -> gensim -> HuggingFace name-resolution
    order (GH #198), built on data-wrangler's existing HF text-embedding
    support (`datawrangler.zoo.text.apply_text_model`).

    Parameters
    ----------

    name : str
        A Hugging Face sentence-transformers model name/id, e.g.
        'all-MiniLM-L6-v2'.

    Returns
    ----------

    cls : class
        A fit/transform/fit_transform class (fit is a no-op -- Hugging
        Face sentence-transformers models are pretrained) that embeds
        documents with the named model when `.transform()` is called.
    """
    class _HFTextModel(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            """No-op fit (the pretrained HuggingFace model needs no fitting); marks `self` as fitted."""
            self.fitted_ = True
            return self

        def transform(self, X):
            """Embed each document in `X` with the pretrained HuggingFace model `name`.

            Returns
            -------
            scipy.sparse.csr_matrix
                The embedded documents as a sparse matrix.
            """
            from datawrangler.zoo.text import apply_text_model
            from scipy import sparse
            embedded = apply_text_model(name, list(X), mode='transform')
            return sparse.csr_matrix(np.asarray(embedded))

    return _HFTextModel


def _spec_model_name(x):
    """Extract the string model name from a vectorizer=/semantic= spec,
    if there is one.

    Parameters
    ----------

    x : str, dict, class or class instance
        A vectorizer=/semantic= spec.

    Returns
    ----------

    name : str or None
        `x` itself if it is a string; `x['model']` if `x` is a dict with a
        string 'model' key (covers both the canonical dict spec and the
        legacy `{'model', 'params'}` spec); otherwise None.
    """
    if isinstance(x, str):
        return x
    if isinstance(x, dict) and isinstance(x.get('model'), str):
        return x['model']
    return None


def _resolve_registry_name(name, registry, kind):
    """Resolve `name` into `registry` (mutating it in place) following the
    sklearn -> gensim -> HuggingFace order, so the rest of `text2mat`'s
    dispatch logic (which only knows how to look names up in `registry`)
    needs no gensim- or HuggingFace-specific special-casing.

    Parameters
    ----------

    name : str
        A vectorizer=/semantic= model name.

    registry : dict
        `vectorizer_models` or `texts` -- mutated in place by inserting
        the resolved class under `name` if it is not already a native
        scikit-learn entry.

    kind : 'vectorizer' or 'semantic'
        Which family of gensim wrappers is eligible for `name`.
    """
    if name in registry:
        return  # tier 1: already a native scikit-learn registry entry
    gensim_cls = _gensim_model_for(name, kind)  # tier 2: gensim
    if gensim_cls is not None:
        registry[name] = gensim_cls
        return
    registry[name] = _hf_fallback_model(name)  # tier 3: HuggingFace


def text2mat(data, vectorizer='CountVectorizer',
             semantic='LatentDirichletAllocation', corpus='wiki'):
    """
    Turns a list of text samples into a matrix using a vectorizer and a text model

    Parameters
    ----------

    data : list (or list of lists) of text samples
        The text data to transform

    vectorizer : str, dict, class or class instance
        The vectorizer to use. A string name is resolved in three tiers
        (GH #198): (1) the built-in scikit-learn registry --
        'CountVectorizer' or 'TfidfVectorizer'; (2) if not found there, the
        gensim wrapper registry (`hypertools.tools.gensim_models`, requires
        `pip install "hypertools[gensim]"`) -- 'Word2Vec', 'Doc2Vec', or
        'FastText'; (3) if still not found, the name is treated as a
        Hugging Face sentence-transformers model name/id (e.g.
        'all-MiniLM-L6-v2'), via data-wrangler's HF embedding support. To
        change default parameters, set to a dictionary e.g.
        {'model' : 'CountVectorizer', 'kwargs' : {'max_features' : 10}}
        (the legacy {'model', 'params'} form is also still accepted). See
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
        for scikit-learn details. You can also specify your own vectorizer
        model as a class, or class instance.  With either option, the class
        must have a fit_transform method (see here: http://scikit-learn.org/stable/data_transforms.html).
        If a class, pass any parameters as a dictionary to vectorizer_params. If
        a class instance, no parameters can be passed.

    semantic : str, dict, class or class instance
        Text model to use to transform text data. Built-in options are
        'LatentDirichletAllocation' or 'NMF' (default: LDA). String names
        are resolved using the same three-tier order as `vectorizer`
        (sklearn -> gensim -> HuggingFace, GH #198); the gensim tier adds
        'LdaModel', 'LsiModel', and 'HdpModel'. To change default
        parameters, set to a dictionary e.g. {'model' : 'NMF', 'kwargs' :
        {'n_components' : 10}} (the legacy {'model', 'params'} form is also
        still accepted). See
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
        for details on the two scikit-learn model options. You can also
        specify your own text model as a class, or class instance.  With
        either option, the class must have a fit_transform method (see
        here: http://scikit-learn.org/stable/data_transforms.html).
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
    # semantic=None means "no semantic-stage model" (skip straight from the
    # vectorizer's output to the returned matrix) -- semantic's function
    # default is the string 'LatentDirichletAllocation', so an explicit
    # None here is always a deliberate user override, never an "unset"
    # sentinel (GH #198; also fixes GH #244's test_transform_no_text_model,
    # whose name always documented this intent even though the previous
    # `semantic is None -> 'LatentDirichletAllocation'` fallback silently
    # ran LDA instead).
    if vectorizer is None:
        vectorizer = 'CountVectorizer'
    model_is_fit=False
    if corpus is not None:
        if corpus in ('wiki', 'nips', 'sotus',):
            if semantic == 'LatentDirichletAllocation' and vectorizer == 'CountVectorizer':
                # The hosted pretrained topic model was pickled with an older
                # scikit-learn; unpickling it under a newer sklearn emits an
                # InconsistentVersionWarning. Its persisted state is just the
                # fitted vocabulary_ / components_ arrays, which load and
                # transform correctly across versions (verified numerically,
                # QC 2026-07), so the warning is noise for this known-safe
                # model -- silence it narrowly so every default text plot
                # doesn't print a scary (and here spurious) "invalid results"
                # warning.
                import warnings as _warnings
                try:
                    from sklearn.exceptions import InconsistentVersionWarning
                    _ver_warn = InconsistentVersionWarning
                except Exception:
                    _ver_warn = UserWarning
                with _warnings.catch_warnings():
                    _warnings.simplefilter('ignore', _ver_warn)
                    semantic = load(corpus + '_model')
                vectorizer = None
                model_is_fit = True
            else:
                corpus = np.array(load(corpus))
        else:
            corpus = np.array([corpus])

    # GH #198: resolve vectorizer=/semantic= string specs (including the
    # 'model' name inside a dict spec) through the sklearn -> gensim ->
    # HuggingFace order before the existing str/dict/class dispatch below
    # ever runs -- once resolved, the name is a plain `vectorizer_models`/
    # `texts` registry key like any built-in scikit-learn one.
    _vname = _spec_model_name(vectorizer)
    if _vname is not None:
        _resolve_registry_name(_vname, vectorizer_models, 'vectorizer')
    _sname = _spec_model_name(semantic)
    if _sname is not None:
        _resolve_registry_name(_sname, texts, 'semantic')

    vtype = _check_mtype(vectorizer)
    if vtype == 'str':
        vectorizer_params = default_params(vectorizer) or {}
    elif vtype == 'dict':
        vectorizer_params = default_params(
            vectorizer['model'],
            vectorizer.get('kwargs', vectorizer.get('params', {}))) or {}
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
    # GH #198 (QC P1-2): embedding vectorizers (gensim Word2Vec/Doc2Vec/
    # FastText) emit continuous, often-negative document vectors, which the
    # default topic-model semantic stage (LatentDirichletAllocation / NMF)
    # cannot consume ("Negative values in data passed to ..."). Running a
    # topic model on embeddings is not meaningful anyway, so skip the
    # semantic stage in that combination with a clear warning rather than
    # crashing. Pass semantic=None explicitly to silence this.
    if (isinstance(vectorizer, str) and vectorizer in _GENSIM_VECTORIZER_NAMES
            and isinstance(semantic, str)
            and semantic in ('LatentDirichletAllocation', 'NMF')):
        warnings.warn(
            f"vectorizer={vectorizer!r} produces continuous embeddings that "
            f"the {semantic} semantic model cannot consume; skipping the "
            f"semantic stage and returning the embeddings directly. Pass "
            f"semantic=None to silence this warning.",
            UserWarning, stacklevel=2)
        semantic = None

    ttype = _check_mtype(semantic)
    if ttype == 'str':
        text_params = default_params(semantic) or {}
    elif ttype == 'dict':
        text_params = default_params(
            semantic['model'],
            semantic.get('kwargs', semantic.get('params', {}))) or {}
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
