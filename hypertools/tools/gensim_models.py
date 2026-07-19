"""sklearn-API wrappers around gensim topic/embedding models (GH #198).

`hypertools.tools.text2mat` resolves `vectorizer=`/`semantic=` string specs
in three tiers: scikit-learn registry first, then the gensim wrappers
defined here, then a Hugging Face (data-wrangler) fallback -- exactly the
order Jeremy specified on GH #198 ("first look for the scikit-learn model;
then, if that doesn't exist, look for the gensim model; then, if that
doesn't exist, look for the huggingface model"). Every class below exposes
the same ``fit``/``transform``/``fit_transform`` surface as a scikit-learn
estimator (via ``sklearn.base.BaseEstimator``/``TransformerMixin``), so
`text2mat`'s existing dispatch machinery (which already knows how to drive
any scikit-learn-shaped model) needs no gensim-specific special-casing.

This module is imported lazily -- only when a gensim model name is actually
requested -- and requires the optional ``gensim`` dependency
(``pip install "hypertools[gensim]"``).

Vectorizer-stage models (``Word2VecVectorizer``, ``Doc2VecVectorizer``,
``FastTextVectorizer``) turn a list of raw text documents into one
embedding vector per document (mean of the trained per-word vectors, or a
Doc2Vec-inferred vector). Semantic-stage models (``LdaVectorizer``,
``LsiVectorizer``, ``HdpVectorizer``) turn a document-term matrix (produced
by any vectorizer, gensim or scikit-learn) into a dense
(n_docs, n_topics) topic-proportion matrix, mirroring
``sklearn.decomposition.LatentDirichletAllocation``/``NMF``.

Determinism note: gensim's multi-threaded training (``workers`` > 1) uses
one RNG stream per worker thread, so results are **not** reproducible
across runs even with a fixed ``seed``/``random_state``. Every vectorizer
-stage wrapper below therefore forces ``workers=1`` whenever ``seed`` is
not None (the default is ``seed=0``, so training is single-threaded and
deterministic by default); pass ``seed=None`` (and optionally your own
``workers=``) to opt back into faster, non-deterministic multi-threaded
training.
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import gensim  # noqa: F401
except ImportError as exc:
    raise ImportError(
        'hypertools.tools.gensim_models requires gensim. Install it with '
        'pip install "hypertools[gensim]"'
    ) from exc

from gensim.utils import simple_preprocess


def _tokenize(doc):
    """Tokenize a single document with gensim's `simple_preprocess`.

    Parameters
    ----------

    doc : str
        A single text document.

    Returns
    -------

    tokens : list of str
        Lowercased, punctuation-stripped tokens.
    """
    return simple_preprocess(str(doc))


def _effective_workers(seed, workers):
    """Resolve the number of gensim training threads to use.

    Parameters
    ----------

    seed : int or None
        The requested random seed. When not None, training must be
        single-threaded (``workers=1``) for gensim's results to be
        reproducible -- see the module docstring.

    workers : int or None
        The user-requested worker count, or None to pick a default.

    Returns
    -------

    workers : int
        1 if `seed` is not None and `workers` was not explicitly set;
        otherwise `workers` (defaulting to 4 when both are None).
    """
    if workers is not None:
        return workers
    return 1 if seed is not None else 4


def _dense_to_bow_corpus(X):
    """Convert a dense document-term matrix into a gensim bag-of-words
    corpus (a list of per-document ``[(term_id, count), ...]`` lists),
    built internally so every semantic-stage wrapper below can be dropped
    in wherever a document-term matrix is available -- no separate
    tokenization/dictionary-building step required by the caller.

    Parameters
    ----------

    X : array-like, shape (n_docs, n_features) or (n_features,)
        A dense document-term (or document-embedding) matrix, e.g. the
        output of `CountVectorizer.transform` or `TfidfVectorizer.transform`.
        A 1-D input is treated as a single document.

    Returns
    -------

    corpus : list of list of (int, float)
        One bag-of-words representation per document (zero entries
        omitted).

    n_features : int
        The number of columns (vocabulary size) of `X`.
    """
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    corpus = [[(int(j), float(v)) for j, v in enumerate(row) if v != 0]
              for row in X]
    return corpus, X.shape[1]


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Vectorizer-stage model: a document's vector is the mean of its
    tokens' trained gensim Word2Vec word vectors (out-of-vocabulary tokens
    at transform time are skipped; a document with no known tokens gets
    the zero vector).

    Parameters
    ----------

    vector_size : int
        Dimensionality of the word (and, by averaging, document) vectors.
        Default: 100.

    window : int
        Maximum distance between the current and predicted word within a
        sentence. Default: 5.

    min_count : int
        Ignores all words with total frequency lower than this. Default: 1
        (so short synthetic corpora keep every word).

    epochs : int
        Number of training epochs. Default: 10.

    seed : int or None
        Random seed for reproducible training. Default: 0. When not None,
        training is forced single-threaded (see `_effective_workers`) --
        multi-threaded gensim training is not reproducible even with a
        fixed seed.

    workers : int or None
        Number of worker threads. Default: None, which resolves to 1 when
        `seed` is not None (deterministic) or 4 otherwise.

    Attributes
    ----------

    model_ : gensim.models.Word2Vec
        The trained Word2Vec model (set by `fit`).
    """

    def __init__(self, vector_size=100, window=5, min_count=1, epochs=10,
                 seed=0, workers=None):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.workers = workers

    def fit(self, X, y=None):
        """Train a Word2Vec model on the given documents.

        Parameters
        ----------

        X : list of str
            Raw text documents.

        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------

        self : Word2VecVectorizer
        """
        from gensim.models import Word2Vec
        tokenized = [_tokenize(doc) for doc in X]
        self.model_ = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            seed=self.seed if self.seed is not None else 1,
            workers=_effective_workers(self.seed, self.workers),
        )
        return self

    def transform(self, X):
        """Embed documents as the mean of their trained word vectors.

        Parameters
        ----------

        X : list of str
            Raw text documents.

        Returns
        -------

        embeddings : scipy.sparse.csr_matrix, shape (n_docs, vector_size)
        """
        rows = []
        for doc in X:
            tokens = [t for t in _tokenize(doc) if t in self.model_.wv]
            if tokens:
                rows.append(np.mean(self.model_.wv[tokens], axis=0))
            else:
                rows.append(np.zeros(self.vector_size, dtype=np.float64))
        return sparse.csr_matrix(np.vstack(rows).astype(np.float64))


class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    """Vectorizer-stage model: a document's vector is either its trained
    gensim Doc2Vec tag vector (for documents seen during `fit`) or an
    inferred vector (`Doc2Vec.infer_vector`) for new documents at
    transform time. To keep `transform` consistent regardless of whether
    a document was part of the training corpus, this wrapper always uses
    `infer_vector`.

    Note that while two identically-seeded *fits* produce identical
    `transform` outputs, repeated `transform()` calls on the SAME fitted
    instance are not bitwise-identical: gensim's `infer_vector` advances
    the model's internal random-number state on every call.

    Parameters
    ----------

    vector_size : int
        Dimensionality of the document vectors. Default: 100.

    window : int
        Maximum distance between the current and predicted word within a
        sentence. Default: 5.

    min_count : int
        Ignores all words with total frequency lower than this. Default: 1.

    epochs : int
        Number of training epochs (also used as the number of inference
        steps in `infer_vector`). Default: 10.

    seed : int or None
        Random seed for reproducible training/inference. Default: 0. See
        `Word2VecVectorizer` for the single-worker determinism note.

    workers : int or None
        Number of worker threads used during training. Default: None
        (resolves per `_effective_workers`).

    Attributes
    ----------

    model_ : gensim.models.doc2vec.Doc2Vec
        The trained Doc2Vec model (set by `fit`).
    """

    def __init__(self, vector_size=100, window=5, min_count=1, epochs=10,
                 seed=0, workers=None):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.workers = workers

    def fit(self, X, y=None):
        """Train a Doc2Vec model on the given documents.

        Parameters
        ----------

        X : list of str
            Raw text documents.

        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------

        self : Doc2VecVectorizer
        """
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        tagged = [TaggedDocument(_tokenize(doc), [i])
                  for i, doc in enumerate(X)]
        self.model_ = Doc2Vec(
            documents=tagged,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            seed=self.seed if self.seed is not None else 1,
            workers=_effective_workers(self.seed, self.workers),
        )
        return self

    def transform(self, X):
        """Infer document vectors for the given documents.

        Parameters
        ----------

        X : list of str
            Raw text documents.

        Returns
        -------

        embeddings : scipy.sparse.csr_matrix, shape (n_docs, vector_size)
        """
        rows = [self.model_.infer_vector(_tokenize(doc), epochs=self.epochs)
                for doc in X]
        return sparse.csr_matrix(np.vstack(rows).astype(np.float64))


class FastTextVectorizer(BaseEstimator, TransformerMixin):
    """Vectorizer-stage model: a document's vector is the mean of its
    tokens' trained gensim FastText word vectors. Unlike
    `Word2VecVectorizer`, out-of-vocabulary tokens are subword-safe:
    FastText represents every word (seen or unseen) as the sum of its
    character n-gram vectors, so `transform` can produce meaningful
    vectors for words never seen during `fit`.

    Parameters
    ----------

    vector_size : int
        Dimensionality of the word (and document) vectors. Default: 100.

    window : int
        Maximum distance between the current and predicted word within a
        sentence. Default: 5.

    min_count : int
        Ignores all words with total frequency lower than this. Default: 1.

    epochs : int
        Number of training epochs. Default: 10.

    seed : int or None
        Random seed for reproducible training. Default: 0. See
        `Word2VecVectorizer` for the single-worker determinism note.

    workers : int or None
        Number of worker threads. Default: None (resolves per
        `_effective_workers`).

    Attributes
    ----------

    model_ : gensim.models.FastText
        The trained FastText model (set by `fit`).
    """

    def __init__(self, vector_size=100, window=5, min_count=1, epochs=10,
                 seed=0, workers=None):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.workers = workers

    def fit(self, X, y=None):
        """Train a FastText model on the given documents.

        Parameters
        ----------

        X : list of str
            Raw text documents.

        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------

        self : FastTextVectorizer
        """
        from gensim.models import FastText
        tokenized = [_tokenize(doc) for doc in X]
        self.model_ = FastText(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            seed=self.seed if self.seed is not None else 1,
            workers=_effective_workers(self.seed, self.workers),
        )
        return self

    def transform(self, X):
        """Embed documents as the mean of their (subword-safe) word
        vectors.

        Parameters
        ----------

        X : list of str
            Raw text documents.

        Returns
        -------

        embeddings : scipy.sparse.csr_matrix, shape (n_docs, vector_size)
        """
        rows = []
        for doc in X:
            tokens = _tokenize(doc)
            vecs = []
            for t in tokens:
                try:
                    vecs.append(self.model_.wv[t])
                except KeyError:
                    continue
            if vecs:
                rows.append(np.mean(vecs, axis=0))
            else:
                rows.append(np.zeros(self.vector_size, dtype=np.float64))
        return sparse.csr_matrix(np.vstack(rows).astype(np.float64))


class LdaVectorizer(BaseEstimator, TransformerMixin):
    """Semantic-stage model: gensim's `LdaModel` (Latent Dirichlet
    Allocation) trained over a bag-of-words corpus built internally from a
    dense document-term matrix (e.g. the output of a vectorizer-stage
    model). `transform` returns dense (n_docs, num_topics) topic
    proportions -- each row sums to (approximately) 1, mirroring
    `sklearn.decomposition.LatentDirichletAllocation`.

    Parameters
    ----------

    num_topics : int
        Number of latent topics. Default: 20 (matches
        `hypertools`' scikit-learn LDA/NMF default).

    passes : int
        Number of passes through the corpus during training. Default: 1.

    seed : int or None
        Random state for reproducible training (gensim's `random_state`).
        Default: 0.

    Attributes
    ----------

    model_ : gensim.models.LdaModel
        The trained LDA model (set by `fit`).

    n_features_ : int
        Vocabulary size (number of document-term matrix columns) seen
        during `fit`; `transform` requires the same width.
    """

    def __init__(self, num_topics=20, passes=1, seed=0):
        self.num_topics = num_topics
        self.passes = passes
        self.seed = seed

    def fit(self, X, y=None):
        """Fit an LDA model to a dense document-term matrix.

        Parameters
        ----------

        X : array-like, shape (n_docs, n_features)
            A dense document-term matrix (e.g. from `CountVectorizer`).

        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------

        self : LdaVectorizer
        """
        from gensim.models import LdaModel
        corpus, n_features = _dense_to_bow_corpus(X)
        self.n_features_ = n_features
        id2word = {i: str(i) for i in range(n_features)}
        self.model_ = LdaModel(corpus=corpus, id2word=id2word,
                               num_topics=self.num_topics,
                               passes=self.passes, random_state=self.seed)
        return self

    def transform(self, X):
        """Compute topic proportions for a dense document-term matrix.

        Parameters
        ----------

        X : array-like, shape (n_docs, n_features)
            Must have the same number of columns seen during `fit`.

        Returns
        -------

        proportions : numpy.ndarray, shape (n_docs, num_topics)
            Each row sums to approximately 1.
        """
        corpus, n_features = _dense_to_bow_corpus(X)
        if n_features != self.n_features_:
            raise ValueError(
                f'LdaVectorizer: expected {self.n_features_} features, '
                f'got {n_features}'
            )
        rows = []
        for bow in corpus:
            vec = np.zeros(self.num_topics, dtype=np.float64)
            for topic_id, weight in self.model_.get_document_topics(
                    bow, minimum_probability=0.0):
                vec[topic_id] = weight
            rows.append(vec)
        return np.vstack(rows)


class LsiVectorizer(BaseEstimator, TransformerMixin):
    """Semantic-stage model: gensim's `LsiModel` (Latent Semantic
    Indexing) trained over a bag-of-words corpus built internally from a
    dense document-term matrix. `transform` returns dense
    (n_docs, num_topics) projections (unlike LDA these are signed real
    values, not probabilities -- they do not sum to 1).

    Parameters
    ----------

    num_topics : int
        Number of latent dimensions. Default: 20.

    seed : int or None
        Random seed for reproducible training (gensim's `random_seed`).
        Default: 0.

    Attributes
    ----------

    model_ : gensim.models.LsiModel
        The trained LSI model (set by `fit`).

    n_features_ : int
        Vocabulary size seen during `fit`; `transform` requires the same
        width.
    """

    def __init__(self, num_topics=20, seed=0):
        self.num_topics = num_topics
        self.seed = seed

    def fit(self, X, y=None):
        """Fit an LSI model to a dense document-term matrix.

        Parameters
        ----------

        X : array-like, shape (n_docs, n_features)
            A dense document-term matrix.

        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------

        self : LsiVectorizer
        """
        from gensim.models import LsiModel
        corpus, n_features = _dense_to_bow_corpus(X)
        self.n_features_ = n_features
        id2word = {i: str(i) for i in range(n_features)}
        self.model_ = LsiModel(corpus=corpus, id2word=id2word,
                               num_topics=self.num_topics,
                               random_seed=self.seed)
        return self

    def transform(self, X):
        """Compute LSI projections for a dense document-term matrix.

        Parameters
        ----------

        X : array-like, shape (n_docs, n_features)
            Must have the same number of columns seen during `fit`.

        Returns
        -------

        projections : numpy.ndarray, shape (n_docs, num_topics)
        """
        corpus, n_features = _dense_to_bow_corpus(X)
        if n_features != self.n_features_:
            raise ValueError(
                f'LsiVectorizer: expected {self.n_features_} features, '
                f'got {n_features}'
            )
        rows = []
        for bow in corpus:
            vec = np.zeros(self.num_topics, dtype=np.float64)
            for topic_id, weight in self.model_[bow]:
                vec[topic_id] = weight
            rows.append(vec)
        return np.vstack(rows)


class HdpVectorizer(BaseEstimator, TransformerMixin):
    """Semantic-stage model: gensim's `HdpModel` (Hierarchical Dirichlet
    Process), which discovers its own number of topics from the data
    rather than taking a fixed `num_topics` (unlike `LdaVectorizer`/
    `LsiVectorizer`). Because the discovered topic count varies by corpus,
    `transform` produces a fixed-width dense matrix by truncating to the
    first `max_topics` HDP topic indices (dropping -- not renormalizing --
    any probability mass assigned to topics beyond `max_topics`); increase
    `max_topics` to capture more of the tail.

    Parameters
    ----------

    max_topics : int
        Fixed output width -- the number of (leading) HDP topic indices
        to keep. Default: 20.

    seed : int or None
        Random state for reproducible training (gensim's `random_state`).
        Default: 0.

    Attributes
    ----------

    model_ : gensim.models.HdpModel
        The trained HDP model (set by `fit`).

    n_features_ : int
        Vocabulary size seen during `fit`; `transform` requires the same
        width.
    """

    def __init__(self, max_topics=20, seed=0):
        self.max_topics = max_topics
        self.seed = seed

    def fit(self, X, y=None):
        """Fit an HDP model to a dense document-term matrix.

        Parameters
        ----------

        X : array-like, shape (n_docs, n_features)
            A dense document-term matrix.

        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------

        self : HdpVectorizer
        """
        from gensim.models import HdpModel
        corpus, n_features = _dense_to_bow_corpus(X)
        self.n_features_ = n_features
        id2word = {i: str(i) for i in range(n_features)}
        self.model_ = HdpModel(corpus, id2word, random_state=self.seed)
        return self

    def transform(self, X):
        """Compute (truncated) topic proportions for a dense
        document-term matrix.

        Parameters
        ----------

        X : array-like, shape (n_docs, n_features)
            Must have the same number of columns seen during `fit`.

        Returns
        -------

        proportions : numpy.ndarray, shape (n_docs, max_topics)
        """
        corpus, n_features = _dense_to_bow_corpus(X)
        if n_features != self.n_features_:
            raise ValueError(
                f'HdpVectorizer: expected {self.n_features_} features, '
                f'got {n_features}'
            )
        rows = []
        for bow in corpus:
            vec = np.zeros(self.max_topics, dtype=np.float64)
            for topic_id, weight in self.model_[bow]:
                if topic_id < self.max_topics:
                    vec[topic_id] = weight
            rows.append(vec)
        return np.vstack(rows)
