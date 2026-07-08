# -*- coding: utf-8 -*-
"""Tests for gensim vectorizer=/semantic= wrappers (GH #198):

Jeremy's directive on GH #198: "we're already supporting sentence
transformers via HF embeddings (through the shift to wrapping the
data-wrangler package). adding wrappers for gensim seems like a good
direction. for the API, let's parse it as: first look for the scikit-learn
model; then, if that doesn't exist, look for the gensim model; then, if
that doesn't exist, look for the huggingface model."

`hypertools.tools.text2mat.text2mat`'s `vectorizer=`/`semantic=` string
specs now resolve through that exact three-tier order (scikit-learn ->
gensim -> HuggingFace); the gensim wrappers themselves live in
`hypertools.tools.gensim_models` and expose the same
fit/transform/fit_transform surface as any scikit-learn estimator, so no
gensim-specific special-casing was needed in `text2mat`'s dispatch logic.

Every test below trains real (tiny) gensim models on a real ~50-sentence
synthetic corpus -- no mocks anywhere. `gensim` is an optional dependency
(`pip install "hypertools[gensim]"`); the `requires_gensim` skip mirrors
the `requires_pylsl` (tests/test_lsl_streaming.py, GH #130) /
`requires_covering_font` (tests/test_multibyte.py, GH #205) pattern: a
missing gensim on a local machine SKIPS the gensim-dependent tests, but on
CI (GITHUB_ACTIONS=true) `test_ci_has_gensim` FAILS hard instead -- gensim
is listed in the `dev` extra specifically so CI always has it.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools.tools.text2mat import (
    text2mat, texts, vectorizer_models, _resolve_registry_name,
)

try:
    import gensim
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

requires_gensim = pytest.mark.skipif(
    not GENSIM_AVAILABLE,
    reason="gensim is not installed -- install it with `pip install "
           "hypertools[gensim]` to exercise the gensim vectorizer=/"
           "semantic= wrappers",
)


# --------------------------------------------------------------- guard


def test_ci_has_gensim():
    # GH #198 CI guard (mirrors the GH #130 pylsl / GH #205 fonts pattern):
    # on CI, gensim must be importable -- it is listed in the `dev` extra
    # specifically to exercise this module. A failure here means every
    # requires_gensim-gated test below just silently skipped on this PR.
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        pytest.skip("only meaningful on CI (GITHUB_ACTIONS=true); a "
                    "missing gensim on a local machine is expected and "
                    "handled by requires_gensim's skip")
    assert GENSIM_AVAILABLE, (
        'gensim failed to import on CI -- check that `pip install -e '
        '".[dev]"` actually installed gensim (pyproject.toml dev extra).'
    )


# --------------------------------------------------------- synthetic data

# 25 cooking-themed + 25 astronomy-themed sentences: two obvious, mostly
# disjoint vocabularies, so a real Word2Vec/Doc2Vec/FastText/LDA/LSI/HDP
# model trained on them should recover a genuine two-topic structure
# (verified below via cosine similarity and topic-proportion assertions --
# not tautological checks).
COOKING = [
    "chop the onions and garlic finely before adding them to the hot pan",
    "simmer the tomato sauce with basil oregano and a pinch of sugar",
    "bake the sourdough bread until the crust turns golden and crusty",
    "whisk the eggs and sugar together for the sponge cake batter",
    "season the vegetable soup generously with salt pepper and thyme",
    "grill the marinated chicken breast with olive oil and lemon zest",
    "boil the pasta in salted water until it is al dente",
    "roast the root vegetables in the oven with rosemary and garlic",
    "knead the pizza dough on a floured countertop for ten minutes",
    "saute the mushrooms and shallots in butter until they caramelize",
    "fold the whipped cream gently into the chocolate mousse mixture",
    "marinate the beef skewers overnight in soy sauce and ginger",
    "preheat the oven to bake the buttery flaky croissants evenly",
    "stir the risotto slowly while adding warm chicken stock gradually",
    "garnish the salad with toasted almonds and a citrus vinaigrette",
    "reduce the red wine sauce until it coats the back of a spoon",
    "blanch the green beans quickly before plunging them into ice water",
    "caramelize the onions slowly over low heat for a rich flavor",
    "dice the bell peppers and celery for the hearty vegetable stew",
    "brine the turkey overnight to keep the roasted meat moist",
    "toast the spices in a dry skillet before grinding them fresh",
    "layer the lasagna with ricotta cheese marinara sauce and noodles",
    "whip the butter and sugar until light and fluffy for frosting",
    "poach the salmon fillet gently in a court bouillon with herbs",
    "drizzle the warm caramel sauce over the vanilla ice cream sundae",
]

ASTRONOMY = [
    "the powerful telescope observed a distant spiral galaxy at dawn",
    "astronomers carefully measured the orbit of the newly found exoplanet",
    "the glowing nebula illuminates the night sky with vivid colors",
    "scientists tracked the icy comet as it approached the sun",
    "the dense star cluster contains thousands of ancient burning stars",
    "gravity shapes the elliptical orbit of planets around distant stars",
    "the robotic space probe reached the edge of the solar system",
    "astronomers study the faint cosmic microwave background radiation",
    "a massive black hole warps spacetime near the galactic center",
    "the space agency launched a new satellite to study solar flares",
    "researchers detected gravitational waves from two merging neutron stars",
    "the red giant star will eventually collapse into a white dwarf",
    "astronauts aboard the space station observed a spectacular meteor shower",
    "the rover collected soil samples from the dusty martian surface",
    "light from the ancient quasar traveled billions of years to reach us",
    "the observatory recorded a rare transit of a planet across its star",
    "cosmic rays from deep space constantly bombard the upper atmosphere",
    "the spiral arms of the galaxy are rich with young blue stars",
    "scientists modeled the formation of planets from a swirling dust disk",
    "the interstellar probe sent back images of the outer planets",
    "a supernova explosion briefly outshines an entire galaxy of stars",
    "the telescope array mapped radio waves emitted by a pulsar",
    "astronomers calculated the mass of the orbiting binary star system",
    "the lunar mission studied craters on the far side of the moon",
    "solar wind particles stream outward from the surface of the sun",
]

DOCS = COOKING + ASTRONOMY  # 50 documents, indices [0:25) cooking, [25:50) astronomy
N_TOPIC = len(COOKING)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ------------------------------------------------- vectorizer-stage shapes


@requires_gensim
def test_word2vec_vectorizer_shape():
    out = text2mat([DOCS], vectorizer='Word2Vec', semantic=None, corpus=DOCS)
    assert len(out) == 1
    assert out[0].shape == (len(DOCS), 100)  # default vector_size=100


@requires_gensim
def test_doc2vec_vectorizer_shape():
    out = text2mat([DOCS], vectorizer='Doc2Vec', semantic=None, corpus=DOCS)
    assert out[0].shape == (len(DOCS), 100)


@requires_gensim
def test_fasttext_vectorizer_shape():
    out = text2mat([DOCS], vectorizer='FastText', semantic=None, corpus=DOCS)
    assert out[0].shape == (len(DOCS), 100)


@requires_gensim
def test_word2vec_same_topic_more_similar_than_cross_topic():
    # real, non-tautological invariant: a trained Word2Vec model should
    # place same-topic document vectors (mean of shared-vocabulary word
    # vectors) closer together, on average, than cross-topic pairs.
    vecs = text2mat([DOCS], vectorizer='Word2Vec', semantic=None,
                    corpus=DOCS)[0]
    same = ([_cos(vecs[i], vecs[j])
             for i in range(N_TOPIC) for j in range(i + 1, N_TOPIC)]
            + [_cos(vecs[i], vecs[j])
               for i in range(N_TOPIC, len(DOCS))
               for j in range(i + 1, len(DOCS))])
    cross = [_cos(vecs[i], vecs[j])
             for i in range(N_TOPIC)
             for j in range(N_TOPIC, len(DOCS))]
    assert np.mean(same) > np.mean(cross), (
        f'mean same-topic cosine similarity ({np.mean(same):.4f}) should '
        f'exceed mean cross-topic cosine similarity ({np.mean(cross):.4f})'
    )


# --------------------------------------------------- semantic-stage shapes


@requires_gensim
def test_lda_semantic_topic_proportions():
    out = text2mat([DOCS], vectorizer='CountVectorizer', semantic='LdaModel',
                   corpus=DOCS)
    assert out[0].shape == (len(DOCS), 20)  # default num_topics=20
    assert np.allclose(out[0].sum(axis=1), 1.0, atol=1e-4), (
        'LdaModel topic proportions must sum to ~1 per document'
    )
    assert (out[0] >= 0).all()


@requires_gensim
def test_lsi_semantic_shape():
    out = text2mat([DOCS], vectorizer='CountVectorizer', semantic='LsiModel',
                   corpus=DOCS)
    assert out[0].shape == (len(DOCS), 20)


@requires_gensim
def test_hdp_semantic_shape():
    out = text2mat([DOCS], vectorizer='CountVectorizer', semantic='HdpModel',
                   corpus=DOCS)
    assert out[0].shape == (len(DOCS), 20)  # default max_topics=20


# ------------------------------------------------------------- end-to-end


@requires_gensim
def test_plot_word2vec_vectorizer_end_to_end():
    fig = hyp.plot(DOCS, vectorizer='Word2Vec', semantic=None, corpus=DOCS,
                   show=False)
    assert fig is not None


@requires_gensim
def test_plot_lda_semantic_end_to_end():
    fig = hyp.plot(DOCS, semantic='LdaModel', corpus=DOCS, show=False)
    assert fig is not None


# -------------------------------------------------------------- determinism


@requires_gensim
def test_word2vec_determinism_with_seed():
    a = text2mat([DOCS], vectorizer='Word2Vec', semantic=None, corpus=DOCS)[0]
    b = text2mat([DOCS], vectorizer='Word2Vec', semantic=None, corpus=DOCS)[0]
    assert np.allclose(a, b), (
        'Word2VecVectorizer with the default seed=0 (forcing workers=1) '
        'must produce identical output across repeated fits'
    )


@requires_gensim
def test_doc2vec_determinism_with_seed():
    a = text2mat([DOCS], vectorizer='Doc2Vec', semantic=None, corpus=DOCS)[0]
    b = text2mat([DOCS], vectorizer='Doc2Vec', semantic=None, corpus=DOCS)[0]
    assert np.allclose(a, b)


# ------------------------------------------------------------- dict specs


@requires_gensim
def test_dict_spec_vectorizer_word2vec():
    out = text2mat([DOCS],
                   vectorizer={'model': 'Word2Vec',
                               'kwargs': {'vector_size': 25}},
                   semantic=None, corpus=DOCS)
    assert out[0].shape == (len(DOCS), 25)


# --------------------------------------------------------------- parse order


def test_parse_order_nmf_resolves_sklearn_not_gensim():
    # 'NMF' exists in the scikit-learn registry (sklearn.decomposition.NMF)
    # -- per Jeremy's directive, sklearn is tried FIRST, so it must resolve
    # there even though gensim also ships an `Nmf` model. This test needs
    # no gensim install: sklearn's entry short-circuits before gensim is
    # ever imported.
    _resolve_registry_name('NMF', texts, 'semantic')
    assert texts['NMF'].__module__.startswith('sklearn'), (
        f"'NMF' resolved to {texts['NMF'].__module__}, expected a "
        "scikit-learn module (tier 1 must win over gensim, tier 2)"
    )


def test_parse_order_word2vec_not_in_sklearn_vectorizer_registry():
    # sanity check on the other side of the same invariant: 'Word2Vec' is
    # not a scikit-learn vectorizer name, so it must NOT already be present
    # in the registry before gensim resolution runs.
    assert 'Word2Vec' not in vectorizer_models or not str(
        vectorizer_models.get('Word2Vec', object).__module__
    ).startswith('sklearn')


# --------------------------------------------------- ImportError message


def test_import_error_without_gensim_names_the_extra():
    # a REAL import-blocking sys.meta_path finder (not a mock of
    # hypertools) run in a subprocess so this process's already-imported
    # gensim (if installed) is untouched -- mirrors
    # tests/test_lsl_streaming.py::test_import_error_without_pylsl_names_the_extra.
    script = textwrap.dedent("""
        import sys
        import importlib.abc, importlib.machinery

        class BlockLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                raise ImportError("gensim blocked for test")

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == 'gensim' or name.startswith('gensim.'):
                    return importlib.machinery.ModuleSpec(name, BlockLoader())
                return None

        sys.meta_path.insert(0, Blocker())

        import matplotlib
        matplotlib.use('Agg')
        from hypertools.tools.text2mat import text2mat

        try:
            text2mat([['cat dog pet', 'star planet orbit']],
                     vectorizer='Word2Vec', semantic=None, corpus=None)
        except ImportError as exc:
            assert 'hypertools[gensim]' in str(exc), str(exc)
            print("SUBPROCESS_OK")
        else:
            raise AssertionError('expected ImportError, none raised')
    """)
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert 'SUBPROCESS_OK' in result.stdout


@requires_gensim
def test_word2vec_all_oov_doc_gets_zero_vector():
    """A transform-time document whose tokens were ALL unseen at fit time
    must produce a zero vector (the documented OOV fallback), not crash."""
    from hypertools.tools.gensim_models import Word2VecVectorizer

    vec = Word2VecVectorizer(vector_size=25, seed=0)
    vec.fit(DOCS)
    def _dense(m):
        return np.asarray(m.todense() if hasattr(m, 'todense') else m)

    out = _dense(vec.transform(['zzzunknown qqqmystery wwwnonsense']))
    assert out.shape == (1, 25)
    assert np.allclose(out, 0.0)
    # a mixed doc (one known token) must NOT be all-zero
    known_word = DOCS[0].split()[0]
    mixed = _dense(vec.transform([f'{known_word} zzzunknown']))
    assert not np.allclose(mixed, 0.0)


# --- QC P1-2: embedding vectorizer + default topic-model semantic ---
@requires_gensim
def test_word2vec_default_semantic_skips_topic_model_with_warning():
    # Word2Vec emits negative embeddings; the default LDA semantic can't
    # consume them. text2mat should WARN and skip the semantic stage rather
    # than raising "Negative values in data passed to LatentDirichletAllocation".
    with pytest.warns(UserWarning, match="skipping the semantic stage"):
        out = text2mat([DOCS], vectorizer='Word2Vec', corpus=None)  # semantic left at default
    assert len(out) == 1 and out[0].shape[0] == len(DOCS)


@requires_gensim
def test_word2vec_explicit_semantic_none_no_skip_warning():
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)  # any skip-warning would fail here
        # (other benign warnings are filtered to error too, so scope tightly)
        try:
            out = text2mat([DOCS], vectorizer='Word2Vec', semantic=None, corpus=None)
        except UserWarning as e:
            assert "skipping the semantic stage" not in str(e), \
                "explicit semantic=None must not emit the skip warning"
            out = text2mat([DOCS], vectorizer='Word2Vec', semantic=None, corpus=None)
    assert out[0].shape[0] == len(DOCS)
