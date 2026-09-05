"""
==================
Gensim text models
==================

`hyp.plot` accepts text directly. Its ``vectorizer=``/``semantic=`` string
specs resolve in three tiers: scikit-learn's built-ins, then
`gensim <https://radimrehurek.com/gensim/>`_'s models -- `'Word2Vec'`,
`'Doc2Vec'`, `'FastText'` (vectorizer tier) and `'LdaModel'`,
`'LsiModel'`, `'HdpModel'` (semantic tier) -- then HuggingFace
sentence-transformers. gensim is an optional extra that hypertools installs
on demand the first time a gensim model is requested (pre-install it with
``pip install "hypertools[gensim]"``). The two panels embed the same small
three-topic corpus in two ways -- gensim's Word2Vec (averaged word vectors,
no semantic-stage model) on the left, and CountVectorizer counts fed to
gensim's LDA on the right -- and color each document by its topic. For
documents this short, LDA's topic proportions are nearly one-hot, so
documents that LDA assigns to the same topic land on (almost) the same
point in the right-hand panel.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import matplotlib.pyplot as plt
import hypertools as hyp

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

fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                         subplot_kw={'projection': '3d'})

# gensim Word2Vec: average trained word vectors per document (no
# semantic-stage model -- semantic=None). corpus=docs trains the model on
# these documents themselves.
hyp.plot(docs, 'o', vectorizer='Word2Vec', semantic=None, corpus=docs,
         hue=topics, ax=axes[0],
         title='gensim Word2Vec (averaged word vectors)')

# CountVectorizer -> gensim LdaModel: bag-of-words counts, then topic
# proportions from a Latent Dirichlet Allocation model
hyp.plot(docs, 'o', vectorizer='CountVectorizer',
         semantic={'model': 'LdaModel', 'kwargs': {'num_topics': 3}},
         corpus=docs, hue=topics, ax=axes[1],
         title='CountVectorizer + gensim LdaModel')
plt.tight_layout()
