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
cobalt, not canvas-beige. The right-hand column shows, for each painting,
its name with the artist and year, the complete description that was
embedded (in that painting's colour, so nothing about the geometry is
hidden), and a thumbnail of the canvas itself, vertically centred on its
description.

**Data & graceful degradation.** The descriptions are bundled inline (so the
text side is fully offline and deterministic). Each canvas is downloaded
once from Wikimedia Commons and cached; if an image cannot be fetched, a
hand-picked representative colour is used instead, the thumbnail slot shows
a flat swatch of that colour, and the run says so. Text embedding needs the
``[text]`` extra (``pip install "hypertools[text]"``); without it,
``vectorizer='TfidfVectorizer'`` is used, and the pipeline (embed -> reduce
together -> one cloud/colour per painting -> spin) is identical either way.
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
from matplotlib.image import imread

import hypertools as hyp
from hypertools.plot.colors import image_palette

CACHE = os.path.join(tempfile.gettempdir(), 'hypertools_gallery_cache')
FILEPATH = 'https://commons.wikimedia.org/wiki/Special:FilePath/'
# the ONLY committed fixture bytes in the gallery: a 64-px thumbnail the
# test-suite extracts a palette from, so no canvas is ever fetched by a test
# (`__file__` is absent when sphinx-gallery executes this script; it runs
# from the examples directory, so the relative path resolves there)
PALETTE_FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals()
    else os.getcwd(), 'data', 'painting_palette_fixture.png')
try:
    import sentence_transformers  # noqa: F401 -- only asks whether [text] is installed
    VECTORIZER = 'all-MiniLM-L6-v2'
except ImportError:
    VECTORIZER = 'TfidfVectorizer'

PAINTINGS = {
    'Starry Night': {
        'artist': 'Vincent van Gogh', 'year': '1889',
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
        'artist': 'Leonardo da Vinci', 'year': 'c. 1503-1506',
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
        'artist': 'Claude Monet', 'year': '1906',
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
        'artist': 'Edvard Munch', 'year': '1893',
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
        'artist': 'Katsushika Hokusai', 'year': 'c. 1831',
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
# layout, in figure fractions unless noted. The 3-D box is zoomed (zoom= is
# hypertools' camera zoom; the library re-applies it every frame) until its
# projected height roughly matches the description column, and everything
# else is placed from MEASURED extents: the text column starts one GAP right
# of the box's widest projected edge over the whole orbit, the thumbnails one
# GAP right of the widest text block, so the two gaps are equal by
# construction. Thumbnails share one left edge and one width, chosen so the
# tallest (Mona Lisa, portrait) is at most THUMB_MAX_H tall.
SIZE, BOX_ZOOM, MARGIN, GAP = (17.5, 9), 2.3, 0.01, 0.026
WRAP, THUMB_MAX_H, TITLE_PAD = 50, 0.165, 0.012


class Paintings(NamedTuple):
    names: list                 # one per painting, in PAINTINGS order
    descriptions: list          # per painting, its list of word windows
    colors: list                # one RGB tuple per painting
    images: list                # per painting, the cached canvas path or None
    vectorizer: str             # how the windows are embedded
    source: str                 # which path produced the colours


def windows(text, size=WINDOW, step=STEP):
    """Overlapping word windows: one observation per window."""
    words = text.split()
    return [' '.join(words[i:i + size])
            for i in range(0, max(1, len(words) - size + 1), step)]


# --- the data half: the ONLY code here that reaches the network -------------
def fetch_canvas(spec):
    """The painting's canvas (400 px wide), downloaded once into the cache.

    The download and the cache are this example's job (hypertools never
    fetches an image). Returns the local path, or None -- and says so -- if
    the canvas cannot be fetched.
    """
    if os.environ.get('HYPERTOOLS_OFFLINE'):
        raise RuntimeError('HYPERTOOLS_OFFLINE is set: refusing to fetch')
    os.makedirs(CACHE, exist_ok=True)
    dest = os.path.join(CACHE, 'paint_' + spec['file'][:20] + '.jpg')
    try:
        if not os.path.exists(dest):
            req = urllib.request.Request(
                FILEPATH + spec['file'] + '?width=400',
                headers={'User-Agent': 'hypertools-gallery/1.1'})
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = response.read()
            with open(dest + '.part', 'wb') as handle:
                handle.write(payload)
            os.replace(dest + '.part', dest)   # never a truncated cache
        return dest
    except Exception as error:
        print(f'canvas {spec["file"][:20]}... unavailable ({error!r}); '
              'using its hand-picked colour and a flat swatch')
        return None


def canvas_color(spec, path):
    """The painting's most salient colour, from the real canvas.

    Choosing the colour is the library's job: ``image_palette`` orders
    clusters by ``pixel_fraction * chroma``, so a small vivid region beats a
    large muted one. Ordering by cluster SIZE -- which is what this example
    used to do -- returns the background.

    One legibility floor on top of that ordering: the first colour whose
    luminance is at most MAX_LUMINANCE. Measured on the real canvases, The
    Great Wave's two most salient clusters are its cream sky and foam
    (luminance 0.88 and 0.94), which vanish on a white page; its Prussian
    blue is third. Without a canvas, the hand-picked colour stands in.
    """
    if path is None:
        return to_rgb(spec['fallback'])
    return next(tuple(c) for c in image_palette(path)
                if float(c @ LUMA) <= MAX_LUMINANCE)


def load_paintings(paintings=PAINTINGS):
    """The ONLY function here that may touch the network (the canvases)."""
    names = list(paintings)
    images = [fetch_canvas(paintings[n]) for n in names]
    return Paintings(names, [windows(paintings[n]['text']) for n in names],
                     [canvas_color(paintings[n], p) for n, p in zip(names, images)],
                     images, VECTORIZER, 'Wikimedia Commons canvases')


def fixture_data(paintings=PAINTINGS):
    """The same payload with every colour drawn from the one committed
    thumbnail, no canvas images (flat swatches stand in), and a
    deterministic TF-IDF embedding. No network, no model."""
    names = list(paintings)
    return Paintings(names, [windows(paintings[n]['text']) for n in names],
                     [tuple(c) for c in image_palette(PALETTE_FIXTURE,
                                                      n_colors=6)[:len(names)]],
                     [None] * len(names), 'TfidfVectorizer', 'palette fixture')


# --- the figure half: no network, deterministic given its input -------------
def figure_box(text):
    """A drawn text's bounding box in figure fractions (needs a draw first)."""
    fig = text.get_figure()
    return text.get_window_extent().transformed(fig.transFigure.inverted())


def drawn_extent(anim, frames):
    """The union bounding box (figure fractions, (x, y) lower and upper) of
    everything drawn over `frames`, read from the rendered pixels.

    The wireframe box's projected size changes with the camera angle, so
    the figure is measured over the orbit -- the pixels are the one record
    of it that needs no private state. Every fifth frame of a 2-rotation
    spin samples the cube's 90-degree symmetry period ~9 times."""
    fig = anim.figure
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for i in frames:
        anim.draw_frame(i)
        fig.canvas.draw()
        dark = np.asarray(fig.canvas.buffer_rgba())[..., :3].min(-1) < 250
        rows, cols = np.nonzero(dark)
        n_rows, n_cols = dark.shape
        lo = np.minimum(lo, [cols.min() / n_cols, 1 - (rows.max() + 1) / n_rows])
        hi = np.maximum(hi, [(cols.max() + 1) / n_cols, 1 - rows.min() / n_rows])
    return lo, hi


def construct_artifact(data):
    """`data.descriptions` / `data.colors` / `data.images` in, the animation
    out. Returns the HyperAnimation wrapper, never the unpacked pair."""
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
        title='Descriptions of five famous paintings',
        duration=12, frame_rate=15, size=SIZE, zoom=BOX_ZOOM, show=False)
    fig, ax = anim.figure, anim.figure.axes[0]
    ax.set_position([0.15, 0.0, 0.6, 1.0])  # roomy: nothing clipped while measuring
    ax.title.set_visible(False)
    box_lo, box_hi = drawn_extent(anim, range(0, anim.n_frames, 5))
    ax.title.set_visible(True)
    # one block per painting: bold name, then artist and year in italics
    # (placed after measuring the name), then the description that was
    # actually embedded, all in that cloud's colour
    text_x = MARGIN + (box_hi[0] - box_lo[0]) + GAP
    blocks, texts = [], []
    for i, name in enumerate(data.names):
        y, color = 0.945 - i * 0.187, data.colors[i]
        head = fig.text(text_x, y, name, ha='left', va='baseline', fontsize=12,
                        fontweight='bold', color=color)
        body = fig.text(text_x, y - 0.018,
                        '\n'.join(textwrap.wrap(PAINTINGS[name]['text'], WRAP)),
                        ha='left', va='top', fontsize=7, color=color)
        blocks.append((head, body))
        texts += [head, body]
    fig.canvas.draw()                      # so the rendered extents exist
    for (head, _body), name, color in zip(blocks, data.names, data.colors):
        texts.append(fig.text(
            figure_box(head).x1 + 0.006, head.get_position()[1],
            f"{PAINTINGS[name]['artist']}, {PAINTINGS[name]['year']}",
            ha='left', va='baseline', fontsize=9, fontstyle='italic', color=color))
    fig.canvas.draw()
    boxes = [figure_box(t) for t in texts]
    text_x1 = max(b.x1 for b in boxes)
    text_mid = (max(b.y1 for b in boxes) + min(b.y0 for b in boxes)) / 2
    # slide the axes so the box sits MARGIN from the left edge, centred on the
    # text column (a translation: the projection's scale only depends on the
    # axes' size), then hang the title just above its highest projected point
    shift = np.array([MARGIN - box_lo[0], text_mid - (box_lo[1] + box_hi[1]) / 2])
    ax.set_position([0.15 + shift[0], shift[1], 0.6, 1.0])
    # Axes3D rescales a title's y= (matplotlib draws 3-D titles lower than
    # asked), so place it, measure where it landed, and correct in proportion
    top, y0 = box_hi[1] + shift[1] + TITLE_PAD, shift[1]
    ax.set_title(ax.get_title(), y=top - y0,
                 fontproperties=ax.title.get_fontproperties())
    fig.canvas.draw()
    ax.set_title(ax.get_title(), fontproperties=ax.title.get_fontproperties(),
                 y=(top - y0) ** 2 / (figure_box(ax.title).y0 - y0))
    # thumbnails: a missing canvas becomes a flat 4:3 swatch of the painting's
    # colour, so the layout is identical offline
    thumbs = [np.full((3, 4, 3), color) if path is None else imread(path)
              for path, color in zip(data.images, data.colors)]
    aspects = [im.shape[0] / im.shape[1] for im in thumbs]
    fig_w, fig_h = fig.get_size_inches()
    width = min(1 - MARGIN - (text_x1 + GAP),
                THUMB_MAX_H * fig_h / (max(aspects) * fig_w))
    for (head, body), im, aspect in zip(blocks, thumbs, aspects):
        # true aspect, centred on the block's vertical midpoint
        head_box, body_box = figure_box(head), figure_box(body)
        height = width * aspect * fig_w / fig_h
        middle = (max(head_box.y1, body_box.y1) + min(head_box.y0, body_box.y0)) / 2
        thumb = fig.add_axes([text_x1 + GAP, middle - height / 2, width, height])
        thumb.imshow(im)
        thumb.axis('off')
    return anim

if __name__ == '__main__':
    paintings = load_paintings()
    print(f'paintings: {len(paintings.names)}, '
          f'{sum(len(c) for c in paintings.descriptions)} description windows, '
          f'embedded with {paintings.vectorizer} ({paintings.source}); '
          f'{sum(p is not None for p in paintings.images)} canvases fetched')
    anim = construct_artifact(paintings)
    fig = anim.figure
