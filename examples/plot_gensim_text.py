"""
=============================
Gensim text models
=============================

`hyp.tools.text2mat` resolves ``vectorizer=``/``semantic=`` string specs in
three tiers (GH #198): scikit-learn's built-ins, then
`gensim <https://radimrehurek.com/gensim/>`_'s models -- `'Word2Vec'`,
`'Doc2Vec'`, `'FastText'` (vectorizer tier), and `'LdaModel'`, `'LsiModel'`,
`'HdpModel'` (semantic tier) -- then HuggingFace sentence-transformers.
gensim is an optional extra that hypertools installs on demand the first
time a gensim model is requested (pre-install it with
``pip install "hypertools[gensim]"``).
This example embeds a small multi-topic corpus with gensim's Word2Vec
(averaged word vectors, no semantic-stage model) and separately with
CountVectorizer + gensim's LDA, then plots both.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import matplotlib.pyplot as plt
import hypertools as hyp
from hypertools.tools.text2mat import text2mat

docs = [
    'the cat sat on the mat and the dog barked at the cat',
    'dogs and cats are popular household pets around the world',
    'kittens and puppies play together in the yard every morning',
    'the stock market rallied after the central bank cut interest rates',
    'investors watched bond yields fall as inflation data cooled',
    'quarterly earnings beat expectations across most sectors',
    'the galaxy contains billions of stars orbiting a supermassive black hole',
    'astronomers detected a new exoplanet in the habitable zone',
    'the telescope captured images of a distant nebula forming new stars',
]
topics = (['pets'] * 3) + (['finance'] * 3) + (['astronomy'] * 3)

# gensim Word2Vec: average trained word vectors per document (no
# semantic-stage model -- semantic=None)
w2v_vecs = text2mat([docs], vectorizer='Word2Vec', semantic=None,
                     corpus=docs)[0]

# CountVectorizer -> gensim LdaModel: bag-of-words counts, then topic
# proportions from a Latent Dirichlet Allocation model
lda_vecs = text2mat([docs], vectorizer='CountVectorizer',
                     semantic={'model': 'LdaModel',
                               'kwargs': {'num_topics': 3}},
                     corpus=docs)[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                          subplot_kw={'projection': '3d'})
hyp.plot(w2v_vecs, '.', hue=topics, ax=axes[0],
         title='gensim Word2Vec (averaged word vectors)')
hyp.plot(lda_vecs, '.', hue=topics, ax=axes[1],
         title='CountVectorizer + gensim LdaModel')
plt.tight_layout()
plt.show()
