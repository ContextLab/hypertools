# -*- coding: utf-8 -*-
"""
=============================================================
Five paintings, described in words, drawn in their own colors
=============================================================

Text becomes geometry, tinted by the art itself. A full paragraph
describing each of five famous paintings is cut into overlapping word
windows and handed to ``hyp.plot`` **as text** -- a list of five lists of
strings. One call embeds every window with a sentence-transformer
(``vectorizer='all-MiniLM-L6-v2'``), reduces all of them together into one
shared 3-D space with UMAP, keeps the five clouds separate (the nesting of
the input is the grouping), spins the camera, and annotates each cloud with
its painting's name.

Each cloud is drawn in a colour taken from the **actual canvas**:
``hypertools.plot.colors.image_palette`` clusters the downloaded image's
pixels and orders the result by ``pixel_fraction * chroma``, so the vivid
subject wins rather than the muted background -- Starry Night comes out
cobalt, not canvas-beige. The side panels show the complete description
that was embedded, in that painting's colour, so nothing about the geometry
is hidden.

**Data & graceful degradation.** The descriptions are bundled inline (so the
text side is fully offline and deterministic). Each canvas is downloaded
once from Wikimedia Commons and cached; if an image cannot be fetched, a
hand-picked representative colour is used instead, and the run says so.
Text embedding needs the ``[text]`` extra (``pip install
"hypertools[text]"``); without it, ``vectorizer='TfidfVectorizer'`` is used,
and the pipeline (embed -> reduce together -> one cloud/colour per painting
-> spin) is identical either way.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import os
import tempfile
import textwrap
import urllib.request
from typing import NamedTuple

import numpy as np
from matplotlib.colors import to_rgb

import hypertools as hyp
from hypertools.plot.colors import image_palette

CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
FILEPATH = 'https://commons.wikimedia.org/wiki/Special:FilePath/'
# the ONLY committed fixture bytes in the gallery: a 64-px thumbnail the
# test-suite extracts a palette from, so no canvas is ever fetched by a test
PALETTE_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'data', 'painting_palette_fixture.png')
try:
    import sentence_transformers  # noqa: F401 -- only asks whether [text] is installed
    VECTORIZER = 'all-MiniLM-L6-v2'
except ImportError:
    VECTORIZER = 'TfidfVectorizer'

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
LUMA = np.array([0.2126, 0.7152, 0.0722])    # sRGB luminance weights
MAX_LUMINANCE = 0.6                          # legible on a white page


class Paintings(NamedTuple):
    names: list                 # one per painting, in PAINTINGS order
    descriptions: list          # per painting, its list of word windows
    colors: list                # one RGB tuple per painting
    vectorizer: str             # how the windows are embedded
    source: str                 # which path produced the colours


def windows(text, size=WINDOW, step=STEP):
    """Overlapping word windows: one observation per window."""
    words = text.split()
    return [' '.join(words[i:i + size])
            for i in range(0, max(1, len(words) - size + 1), step)]


# --- the data half: the ONLY code here that reaches the network -------------
def canvas_color(spec):
    """The painting's most salient colour, from the real canvas.

    The download and the cache are this example's job (hypertools never
    fetches an image); choosing the colour is the library's:
    ``image_palette`` orders clusters by ``pixel_fraction * chroma``, so a
    small vivid region beats a large muted one. Ordering by cluster SIZE --
    which is what this example used to do -- returns the background.

    One legibility floor on top of that ordering: the first colour whose
    luminance is at most MAX_LUMINANCE. Measured on the real canvases, The
    Great Wave's two most salient clusters are its cream sky and foam
    (luminance 0.88 and 0.94), which vanish on a white page; its Prussian
    blue is third.
    """
    if os.environ.get('HYPERTOOLS_OFFLINE'):
        raise RuntimeError('HYPERTOOLS_OFFLINE is set: refusing to fetch')
    os.makedirs(CACHE, exist_ok=True)
    try:
        dest = os.path.join(CACHE, 'paint_' + spec['file'][:20] + '.jpg')
        if not os.path.exists(dest):
            req = urllib.request.Request(
                FILEPATH + spec['file'] + '?width=400',
                headers={'User-Agent': 'hypertools-gallery/1.1'})
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = response.read()
            with open(dest + '.part', 'wb') as handle:
                handle.write(payload)
            os.replace(dest + '.part', dest)   # never a truncated cache
        return next(tuple(c) for c in image_palette(dest)
                    if float(c @ LUMA) <= MAX_LUMINANCE)
    except Exception as error:
        print(f'canvas {spec["file"][:20]}... unavailable ({error!r}); '
              'using its hand-picked colour')
        return to_rgb(spec['fallback'])


def load_paintings(paintings=PAINTINGS):
    """The ONLY function here that may touch the network (the canvases)."""
    names = list(paintings)
    return Paintings(names, [windows(paintings[n]['text']) for n in names],
                     [canvas_color(paintings[n]) for n in names],
                     VECTORIZER, 'Wikimedia Commons canvases')


def fixture_data(paintings=PAINTINGS):
    """The same payload with every colour drawn from the one committed
    thumbnail and a deterministic TF-IDF embedding. No network, no model."""
    names = list(paintings)
    return Paintings(names, [windows(paintings[n]['text']) for n in names],
                     [tuple(c) for c in image_palette(PALETTE_FIXTURE,
                                                      n_colors=6)[:len(names)]],
                     'TfidfVectorizer', 'palette fixture')


# --- the figure half: no network, deterministic given its input -------------
def construct_artifact(data):
    """`data.descriptions` / `data.colors` in, the animation out. Returns
    the HyperAnimation wrapper, never the unpacked pair."""
    # labels= annotates per OBSERVATION, not per dataset: one sub-list per
    # cloud, carrying the painting's name on its MIDDLE window (roughly the
    # centre of a text trajectory) and None everywhere else.
    labels = [[name if i == len(cloud) // 2 else None
               for i in range(len(cloud))]
              for name, cloud in zip(data.names, data.descriptions)]
    # THE hypertools call: raw TEXT in, five clouds out. The nesting of the
    # input is the grouping, the vectorizer/semantic/corpus trio selects
    # the embedding instead of the default bag-of-words + LDA, and reduce=
    # puts every window into one shared UMAP space so the clouds are
    # directly comparable. n_neighbors=12 keeps one description's windows
    # together, min_dist=0.25 lets a clump pack closely, random_state=42
    # fixes the stochastic layout. 15 fps: the side panels' antialiased
    # text compresses badly in a GIF, and 240 frames at 20 fps was 7 MB.
    anim = hyp.plot(
        data.descriptions, '.',
        vectorizer=data.vectorizer, semantic=None, corpus=None,
        reduce={'model': 'UMAP', 'kwargs': {'n_neighbors': 12, 'min_dist': 0.25,
                                            'random_state': 42}},
        ndims=3, color=data.colors, markersize=5, labels=labels,
        animate='spin', rotations=2,
        title='five paintings, described in words, drawn in their own colors',
        duration=12, frame_rate=15, size=(13, 9), show=False)
    # the descriptions that were actually embedded, each in its cloud's colour
    fig = anim.figure
    fig.axes[0].set_position([0.0, 0.0, 0.52, 1.0])
    for i, name in enumerate(data.names):
        y = 0.94 - i * 0.19
        fig.text(0.55, y, name, ha='left', va='top', fontsize=12,
                 fontweight='bold', color=data.colors[i])
        body = '\n'.join(textwrap.wrap(PAINTINGS[name]['text'], 62))
        fig.text(0.55, y - 0.028, body, ha='left', va='top', fontsize=7,
                 color=data.colors[i])
    return anim


if __name__ == '__main__':
    paintings = load_paintings()
    print(f'paintings: {len(paintings.names)}, '
          f'{sum(len(c) for c in paintings.descriptions)} description windows, '
          f'embedded with {paintings.vectorizer} ({paintings.source})')
    anim = construct_artifact(paintings)
    fig = anim.figure
