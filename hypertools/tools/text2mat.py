"""Text-to-matrix conversion: `text2mat()` and the sklearn -> gensim ->
HuggingFace vectorizer=/semantic= name-resolution order (GH #198).

Hugging Face model path (`_hf_fallback_model`/`_HFTextModel.transform`):
before the lazy `sentence_transformers` import, this module sets
`HF_HUB_DISABLE_PROGRESS_BARS=1`, `HF_HUB_VERBOSITY=error`, and
`TOKENIZERS_PARALLELISM=false` via `os.environ.setdefault(...)` (GH #285).
`huggingface_hub`/`transformers`/`sentence_transformers` read these
variables at import time, so callers no longer need to set them by hand
before importing hypertools (as the tutorial notebooks used to) -- and a
caller's own explicit setting always wins, since `setdefault` never
overwrites an already-set value.
"""

import numpy as np
import inspect
import os
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

# The built-in scikit-learn names, frozen at import time -- BEFORE
# `_resolve_registry_name` ever mutates the live registries above (it
# inserts every gensim/HF-resolved name it sees, so live-registry
# membership is NOT "is this a built-in name": an unknown/typo'd name
# joined `vectorizer_models`/`texts` on its first use and then dodged
# `format_data`'s clear-ValueError rewrap on every later use in the same
# process, re-raising the raw Hugging Face error instead (release-1.0 CI
# fix, run 29582796739 follow-up).
_SKLEARN_VECTORIZER_NAMES = frozenset(vectorizer_models)
_SKLEARN_SEMANTIC_NAMES = frozenset(texts)

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
    -------

    cls : class or None
        The matching wrapper class from `hypertools.tools.gensim_models`
        (e.g. `Word2VecVectorizer` for `name='Word2Vec'`), or None if
        `name` is not a recognized gensim model name for `kind`.

    Raises
    ------

    ImportError
        If `name` is a recognized gensim model name but the `[gensim]`
        extra could not be installed on demand (the message names the
        manual `pip install "hypertools[gensim]"` command).
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
    -------

    cls : class
        A fit/transform/fit_transform class (fit is a no-op -- Hugging
        Face sentence-transformers models are pretrained) that embeds
        documents with the named model when `.transform()` is called.
    """
    class _HFTextModel(BaseEstimator, TransformerMixin):
        # A pretrained embedding model: `fit` is a no-op and `transform`
        # returns continuous (often negative) sentence embeddings rather
        # than word counts, so `text2mat` skips its default topic-model
        # semantic stage for this vectorizer and never loads or embeds a
        # corpus for it (see `_is_pretrained_vectorizer`).
        _hypertools_pretrained = True

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
            # `huggingface_hub`/`transformers`/`sentence_transformers` read
            # these three environment variables AT IMPORT TIME, so they must
            # be set immediately before the lazy import below -- setting
            # them any later (e.g. inside `apply_text_model`, after the
            # import) is a no-op (GH #285). `setdefault` leaves a user's own
            # explicit setting untouched. Values match the ones every
            # tutorial notebook used to set by hand before this fix.
            os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
            os.environ.setdefault('HF_HUB_VERBOSITY', 'error')
            os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
            from .._shared.lazy_import import lazy_import
            lazy_import('sentence_transformers', purpose=f'the {name!r} text model')   # [text]
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
    -------

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


def _is_pretrained_vectorizer(vectorizer, name):
    """Whether a vectorizer= spec resolves to a pretrained embedding model.

    Parameters
    ----------

    vectorizer : str, dict, class or class instance
        The vectorizer= spec, AFTER `_resolve_registry_name` has run on
        its name (so a Hugging Face name is already in `vectorizer_models`).

    name : str or None
        `_spec_model_name(vectorizer)`.

    Returns
    -------

    pretrained : bool
        True if the class (or instance) the spec resolves to carries a
        truthy `_hypertools_pretrained` attribute -- the Hugging Face
        sentence-transformers tier (`_hf_fallback_model`), or a user
        class/instance that marks itself the same way. Such a vectorizer
        needs no fitting and emits continuous embeddings, so the default
        topic-model semantic stage does not apply to it and `corpus=` is
        unused.
    """
    cls = vectorizer_models.get(name) if name is not None else vectorizer
    return bool(getattr(cls, '_hypertools_pretrained', False))


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
        gensim wrapper registry (`hypertools.tools.gensim_models`; the
        `[gensim]` extra installs itself on first use) -- 'Word2Vec', 'Doc2Vec', or
        'FastText'; (3) if still not found, the name is treated as a
        Hugging Face sentence-transformers model name/id (e.g.
        'all-MiniLM-L6-v2'), via data-wrangler's HF embedding support. To
        change default parameters, set to a dictionary e.g.
        {'model' : 'CountVectorizer', 'kwargs' : {'max_features' : 10}}
        (the legacy {'model', 'params'} form is also still accepted). See
        https://scikit-learn.org/stable/api/sklearn.feature_extraction.html
        for scikit-learn details. You can also specify your own vectorizer
        model as a class, or class instance.  With either option, the class
        must have a fit_transform method (see
        https://scikit-learn.org/stable/data_transforms.html).
        To set parameters, use the dict form (or a configured class
        instance); a bare class is instantiated with its defaults.

    semantic : str, dict, class or class instance
        Text model to use to transform text data. Built-in options are
        'LatentDirichletAllocation' or 'NMF' (default: LDA for count
        vectorizers; for embedding vectorizers -- gensim 'Word2Vec'/
        'Doc2Vec'/'FastText', or a Hugging Face sentence-transformers model
        -- the semantic stage defaults to none and `corpus` is unused, since
        a topic model cannot consume continuous embeddings; a gensim
        vectorizer warns when it skips the stage, a Hugging Face one skips
        it silently, and an explicit 'NMF' with a Hugging Face vectorizer
        raises `ValueError`). Pass None to skip the semantic stage
        explicitly. String names are resolved using the same three-tier
        order as `vectorizer` (sklearn -> gensim -> HuggingFace, GH #198);
        the gensim tier adds 'LdaModel', 'LsiModel', and 'HdpModel'. To
        change default
        parameters, set to a dictionary e.g. {'model' : 'NMF', 'kwargs' :
        {'n_components' : 10}} (the legacy {'model', 'params'} form is also
        still accepted). See
        https://scikit-learn.org/stable/api/sklearn.decomposition.html
        for details on the two scikit-learn model options. You can also
        specify your own text model as a class, or class instance.  With
        either option, the class must have a fit_transform method (see
        https://scikit-learn.org/stable/data_transforms.html).
        To set parameters, use the dict form (or a configured class
        instance); a bare class is instantiated with its defaults.

    corpus : list (or list of lists) of text samples or 'wiki', 'nips', 'sotus'.
         Text to use to fit the semantic model (optional). If set to 'wiki', 'nips'
         or 'sotus' and the default semantic and vectorizer models are used, a
         pretrained model will be loaded which can save a lot of time. Unused
         (never loaded or embedded) when the vectorizer is a pretrained
         Hugging Face embedding model and there is no semantic stage.

    Returns
    -------

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
    if corpus is not None and not isinstance(corpus, str):
        # corpus= must be TEXT: a list (or list of lists) of strings, or a
        # hosted corpus name. A non-string scalar (e.g. corpus=12345) used
        # to leak "'numpy.int64' object has no attribute 'lower'" from deep
        # inside the vectorizer, naming neither the argument nor the fix
        # (release-1.0 audit, D05-gallery-data-text-013).
        def _all_text(c):
            if isinstance(c, str):
                return True
            if isinstance(c, (list, tuple, np.ndarray)):
                return len(c) > 0 and all(_all_text(ci) for ci in c)
            return False
        if not _all_text(corpus):
            raise ValueError(
                f'corpus= must be a (non-empty) list of text samples (or '
                f'list of lists of them), or one of the hosted corpus '
                f"names 'wiki'/'nips'/'sotus'; got "
                f'{type(corpus).__name__}: {corpus!r}.')
    # GH #198: resolve vectorizer=/semantic= string specs (including the
    # 'model' name inside a dict spec) through the sklearn -> gensim ->
    # HuggingFace order before the existing str/dict/class dispatch below
    # ever runs -- once resolved, the name is a plain `vectorizer_models`/
    # `texts` registry key like any built-in scikit-learn one. This runs
    # BEFORE the corpus block so the embedding-vectorizer decision that
    # follows is made before any corpus is loaded or embedded.
    _vname = _spec_model_name(vectorizer)
    if _vname is not None:
        _resolve_registry_name(_vname, vectorizer_models, 'vectorizer')
    _sname = _spec_model_name(semantic)
    if _sname is not None:
        _resolve_registry_name(_sname, texts, 'semantic')

    # GH #198 (QC P1-2): embedding vectorizers emit continuous, often-
    # negative document vectors, which the default topic-model semantic
    # stage (LatentDirichletAllocation / NMF) cannot consume ("Negative
    # values in data passed to ..."). Running a topic model on embeddings
    # is not meaningful anyway, so skip the semantic stage in that
    # combination rather than crashing:
    #   - gensim vectorizers (Word2Vec/Doc2Vec/FastText) skip it with a
    #     clear warning; pass semantic=None explicitly to silence it.
    #   - pretrained embedding vectorizers (the Hugging Face tier, marked
    #     `_hypertools_pretrained`) skip it SILENTLY when `semantic` is the
    #     function default (LDA): that is the resolved default for an
    #     embedding vectorizer, not a user mistake. With no semantic stage
    #     there is nothing left to fit, so `corpus` is dropped here and a
    #     hosted corpus name is never downloaded or embedded. (Before this
    #     guard, `hyp.plot(docs, vectorizer='all-MiniLM-L6-v2')` embedded
    #     the entire hosted 'wiki' corpus -- 3,136 documents, ~13 s -- and
    #     then crashed inside `LatentDirichletAllocation.fit`.) An explicit
    #     non-default topic model ('NMF', or a dict spec) raises a clear
    #     ValueError before any corpus work instead of sklearn's internal
    #     "Negative values" error after it.
    if (_vname in _GENSIM_VECTORIZER_NAMES
            and isinstance(semantic, str)
            and semantic in ('LatentDirichletAllocation', 'NMF')):
        warnings.warn(
            f"vectorizer={_vname!r} produces continuous embeddings that "
            f"the {semantic} semantic model cannot consume; skipping the "
            f"semantic stage and returning the embeddings directly. Pass "
            f"semantic=None to silence this warning.",
            UserWarning, stacklevel=2)
        semantic = None
    elif _is_pretrained_vectorizer(vectorizer, _vname):
        if semantic == 'LatentDirichletAllocation':
            semantic = None
        elif _sname in ('LatentDirichletAllocation', 'NMF'):
            raise ValueError(
                f"semantic={_sname!r} cannot be used with "
                f"vectorizer={_vname!r}: a pretrained embedding vectorizer "
                f"produces continuous (often negative) embeddings, not the "
                f"word counts the {_sname} topic model requires. Pass "
                f"semantic=None (the default for embedding vectorizers) to "
                f"return the embeddings directly, or pair {_sname!r} with a "
                f"count vectorizer ('CountVectorizer' or 'TfidfVectorizer').")
        if semantic is None:
            corpus = None  # nothing to fit: the corpus would go unused

    if corpus is not None:
        if isinstance(corpus, str) and \
                corpus not in ('wiki', 'nips', 'sotus'):
            # a plain string is almost never a meaningful training corpus:
            # it is fit as ONE literal document, so a typo'd hosted-corpus
            # name silently produced confidently-wrong topic vectors
            # (release-1.0 audit, D08-tutorials-analysis-011).
            warnings.warn(
                f"corpus={corpus!r} is not one of the hosted corpora "
                "('wiki', 'nips', 'sotus'), so it is being treated as a "
                "LITERAL one-document training corpus -- the resulting "
                "embeddings will reflect that single string, not a real "
                "corpus. If you meant a hosted corpus, check the spelling; "
                "to train on your own corpus, pass a list of text "
                "samples.", UserWarning)
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
                               'https://scikit-learn.org/stable/data_transforms.html')
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
                               'https://scikit-learn.org/stable/data_transforms.html')
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
    if model_is_fit:
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
        except Exception:  # genuinely broad: any failure here means x's
            # type doesn't fit inspect.isclass's expectations, and every
            # such failure maps to the same TypeError
            raise TypeError('Parameter must of type string, dict, class, or'
                            ' class instance.')
