# -*- coding: utf-8 -*-
"""
=============================
Plotting text
=============================

To plot text, simply pass the text data to the plot function.  By default, the
text samples will be transformed into a vector of word counts and then modeled
using Latent Dirichlet Allocation (# of topics = 50) using a model fit to a
large sample of wikipedia pages.  If you specify semantic=None, the word
count vectors will be plotted. To convert the text to a matrix (or list of
matrices), we also expose the format_data function. Note: the wikipedia topic
model works best on sentence- or paragraph-length documents with common
dictionary words; very short or slang-heavy snippets can land on nearly
identical topic vectors.
"""

# Code source: Andrew Heusser
# License: MIT

# load hypertools
import hypertools as hyp

# load the data: two groups of documents (about cats and about dogs), plus
# one document about something else entirely
data = [["The cat curled up on the windowsill and purred in the afternoon "
         "sun. Cats are independent animals that groom themselves and hunt "
         "mice, and a house cat sleeps for most of the day.",
         "Kittens play with yarn and chase laser pointers across the floor. "
         "A cat communicates by purring and meowing, and most cats prefer "
         "to nap somewhere warm and quiet.",
         "Many people keep cats as pets because they are quiet and clean "
         "animals. A pet cat uses a litter box and eats fish and small "
         "birds."],
        ["The dog barked at the mail carrier and wagged its tail with joy. "
         "Dogs are loyal animals that live in packs and love to play fetch "
         "with their owners in the park.",
         "Puppies chew on bones and dig holes in the garden. A dog learns "
         "commands like sit and stay, and working dogs herd sheep and "
         "guard the farm.",
         "Many people keep dogs as pets because they are friendly and "
         "protective companions. A pet dog needs daily walks and loves to "
         "run and swim outdoors."],
        "The stock market rallied after the central bank cut interest "
        "rates, and investors bought shares in technology companies as "
        "bond yields fell and the economy grew."]

# plot it
hyp.plot(data, 'o')

# convert text to matrix without plotting
# mtx = hyp.tools.format_data(data, vectorizer='TfidfVectorizer', semantic='NMF')
