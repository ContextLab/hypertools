# -*- coding: utf-8 -*-
"""
=============================================================
Five paintings, described in words, drawn in their own colors
=============================================================

Text becomes geometry, tinted by the art itself. Short descriptions of five
famous paintings are embedded with a sentence-transformer and reduced *together*
into one shared 3-D space with ``hyp.reduce`` (UMAP). Each painting is one
cloud, plotted by ``hyp.plot`` in a color taken from the actual canvas; the box,
the spin animation (``animate='spin'``) and the markers are all the library's.
A stack of side panels lists each description, colored to match its dots.

**Data & graceful degradation.** The painting descriptions are bundled inline
(so the example is self-contained). The per-canvas color is extracted by
k-means over the pixels of the real image, downloaded from
`Wikimedia Commons <https://commons.wikimedia.org>`_ and cached; if the image
cannot be fetched, a hand-picked representative color is used instead. Text is
embedded with ``all-MiniLM-L6-v2`` when sentence-transformers is installed,
falling back to a character n-gram TF-IDF vector otherwise -- so the example
renders without a large model download. The pipeline (embed -> reduce together
-> one cloud/color per painting -> spin) is identical either way.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import os
import tempfile
import textwrap
import urllib.request

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

import hypertools as hyp

CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
os.makedirs(CACHE, exist_ok=True)
FILEPATH = 'https://commons.wikimedia.org/wiki/Special:FilePath/'

PAINTINGS = {
    'Starry Night': {
        'file': 'Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
        'fallback': '#2a3f6b',
        'blurb': 'Swirling cobalt sky, blazing yellow stars, a flame-like cypress over a sleeping village.',
        'text': ('A turbulent night sky churns above a quiet village, painted in thick swirling '
                 'brushstrokes of deep cobalt and midnight blue. Great spirals of wind coil across '
                 'the heavens, and enormous yellow stars blaze like halos of fire. A luminous crescent '
                 'moon burns in the corner. A dark flame-shaped cypress tree rises toward the sky over '
                 'a small town beneath a slender church spire, the whole canvas rolling with restless '
                 'rhythmic musical waves of paint.'),
    },
    'Mona Lisa': {
        'file': 'Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
        'fallback': '#6b5533',
        'blurb': 'A serene, enigmatic smile; soft sfumato haze; a dreamlike landscape of misty blue mountains.',
        'text': ('A serene woman sits in three-quarter view, hands softly folded, gazing out with a '
                 'faint and famously enigmatic smile. Leonardos delicate sfumato dissolves every edge '
                 'into soft brown and golden shadow. Warm earth tones glow across her face and dark '
                 'garments. Behind her a dreamlike landscape of winding rivers, misty blue mountains '
                 'and hazy valleys recedes into golden light, calm and intimate and mysterious.'),
    },
    'Water Lilies': {
        'file': 'Claude_Monet_-_Water_Lilies_-_1906,_Ryerson.jpg',
        'fallback': '#3f7d6e',
        'blurb': 'A still pond of floating lilies, reflected clouds, shimmering greens and blues.',
        'text': ('The surface of a still pond fills the entire canvas, no horizon and no sky but the '
                 'skys soft reflection. Pale pink and white water lilies float in drifting clusters '
                 'across cool greens, blues and lavender. Monet dabs shimmering broken color over the '
                 'water, capturing reflected clouds and the trembling light of a summer afternoon in '
                 'his Giverny garden, everything atmosphere and reflection dissolving into gentle haze.'),
    },
    'The Scream': {
        'file': ('Edvard_Munch,_1893,_The_Scream,_oil,_tempera_and_pastel_on_'
                 'cardboard,_91_x_73_cm,_National_Gallery_of_Norway.jpg'),
        'fallback': '#b5502e',
        'blurb': 'A figure clutching its face on a bridge; a sky of blood-orange dread over a swirling fjord.',
        'text': ('A gaunt hairless figure stands on a bridge, hands clasped to its hollow face, mouth '
                 'open in a silent endless scream. The sky burns in violent streaks of blood orange and '
                 'fiery red above a swirling blue-black fjord. Everything undulates in wavy distorted '
                 'lines as if the whole world writhes with the figures anguish. Munchs expressionist '
                 'vision pulses with raw anxiety and dread, trembling with existential terror.'),
    },
    'The Great Wave': {
        'file': 'Tsunami_by_hokusai_19th_century.jpg',
        'fallback': '#1f4e79',
        'blurb': 'A towering wave clawing at tiny boats, deep Prussian blue, tiny Mount Fuji far beyond.',
        'text': ('An enormous curling wave rears up over the sea, its crest breaking into countless '
                 'clawing fingers of white foam reaching like talons toward tiny fishing boats below. '
                 'Painted in deep Prussian blue and pale cream, the ukiyo-e woodblock print frames the '
                 'small snow-capped cone of Mount Fuji far in the distance, dwarfed by the towering '
                 'water. Hokusais bold flat shapes and rhythmic curves capture a breathless instant.'),
    },
}

WINDOW, STEP = 10, 1


def embed(texts):
    """Sentence-transformer embedding, or a char n-gram TF-IDF fallback."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return np.asarray(model.encode(texts, show_progress_bar=False),
                          dtype=float)
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
        return vec.fit_transform(texts).toarray().astype(float)


def windows(text, size, step):
    w = text.split()
    return [' '.join(w[i:i + size])
            for i in range(0, max(1, len(w) - size + 1), step)]


def canvas_color(spec):
    """Dominant color from the real canvas (k-means over pixels), or the
    hand-picked fallback if the image cannot be fetched."""
    try:
        from PIL import Image
        from sklearn.cluster import KMeans
        dest = os.path.join(CACHE, 'paint_' + spec['file'][:20] + '.jpg')
        if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
            req = urllib.request.Request(
                FILEPATH + spec['file'] + '?width=400',
                headers={'User-Agent': 'hypertools-gallery/1.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            with open(dest, 'wb') as f:
                f.write(data)
        im = Image.open(dest).convert('RGB')
        im.thumbnail((200, 200))
        px = np.asarray(im).reshape(-1, 3).astype(float)
        km = KMeans(n_clusters=6, n_init=4, random_state=0).fit(px)
        counts = np.bincount(km.labels_, minlength=6)
        rgb = km.cluster_centers_[np.argmax(counts)] / 255.0
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        if lum > 0.5:                     # keep swatches legible on white
            rgb = rgb * (0.5 / max(lum, 1e-6))
        return tuple(rgb)
    except Exception:
        return to_rgb(spec['fallback'])


# collect every painting's windows + one canvas-derived color each, then embed
# ALL windows in a SINGLE call so every vector shares the same dimensionality
# (the TF-IDF fallback fits its vocabulary once -- embedding per painting would
# give each a different feature dimension)
all_windows, owners, colors_by_name = [], [], {}
for name, spec in PAINTINGS.items():
    wins = windows(spec['text'], WINDOW, STEP)
    all_windows += wins
    owners += [name] * len(wins)
    colors_by_name[name] = canvas_color(spec)
all_vecs = embed(all_windows)
owners = np.array(owners)

# reduce all descriptions together into a shared 3-D space, then trim each
# cloud's few farthest outliers so the clouds read cleanly. The UMAP kwargs
# trade global structure for tighter per-painting clumps: n_neighbors=12 keeps
# the graph local so one description's windows stay together, min_dist=0.25
# lets a clump pack closely, and random_state=42 fixes the stochastic layout.
red = np.asarray(hyp.reduce(
    all_vecs, reduce={'model': 'UMAP',
                      'kwargs': {'n_neighbors': 12, 'min_dist': 0.25,
                                 'random_state': 42}}, ndims=3))
# no rescaling here: hyp.plot already mean-centers every dataset and rescales
# them into [-1, 1] with ONE shared affine before drawing (and the outlier trim
# below compares distances to a percentile, so it is scale-free)
keep = np.ones(len(red), bool)
for name in PAINTINGS:
    idx = np.where(owners == name)[0]
    dist = np.linalg.norm(red[idx] - np.median(red[idx], 0), axis=1)
    keep[idx[dist > np.percentile(dist, 85)]] = False
red, owners = red[keep], owners[keep]

clouds = [red[owners == name] for name in PAINTINGS]
colors = [colors_by_name[name] for name in PAINTINGS]

# THE hypertools call: 5 clouds, one canvas color each, box + spin.
# reduce=None says the clouds are already 3-D, so plot skips its default
# IncrementalPCA; color=colors gives one canvas color per cloud (the color list
# lines up element-for-element with the cloud list); animate='spin' orbits the
# camera rotations=2 full turns over duration=12 s at frame_rate=20 fps (240
# frames) while the points stay fixed, so the parallax reveals the 3-D layout.
duration, fps = 12, 20
fig, ani = hyp.plot(clouds, '.', color=colors, reduce=None, markersize=5,
                    animate='spin', rotations=2, duration=duration,
                    frame_rate=fps, size=(11, 6.6), show=False)

# main plot on the left; description side-panels on the right
ax = fig.axes[0]
ax.set_position([0.0, 0.0, 0.6, 1.0])
fig.text(0.30, 0.965,
         'Five paintings, described in words, drawn in their own colors',
         ha='center', va='top', fontsize=13.5, fontweight='bold',
         color='#1a1a1a')
for i, name in enumerate(PAINTINGS):
    y = 0.90 - i * 0.178
    c = colors_by_name[name]
    fig.text(0.635, y, name, ha='left', va='top', fontsize=13,
             fontweight='bold', color=c)
    body = '\n'.join(textwrap.wrap(PAINTINGS[name]['blurb'], 42)[:3])
    fig.text(0.635, y - 0.032, body, ha='left', va='top', fontsize=8.8,
             color=c)
    fig.add_artist(Rectangle((0.635, y - 0.11), 0.12, 0.014,
                             transform=fig.transFigure, facecolor=c,
                             edgecolor='#dddddd', lw=0.4))
