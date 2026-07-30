# -*- coding: utf-8 -*-
"""
===========================================================================
The shape of a conversation: per-speaker paths, revealed one turn at a time
===========================================================================

A conversation as geometry. Each **turn** (a contiguous run of speech by one
speaker) is embedded as a little sliding-window trajectory; all the windows
share one reduced 3-D space (``hyp.reduce`` with UMAP). Every turn is its own
*disjoint* trajectory (a separate array in the list handed to ``hyp.plot``),
colored by **speaker** (hue). ``animate='serial'`` then reveals the turns one
at a time, accumulating the whole conversation as it plays. A label under the
title names the speaker of the moment in that speaker's colour, and a caption
shows the current line with the words of the window being drawn right now in
bold.

Here the conversation is Lewis Carroll's *Mad Tea-Party* (Alice in
Wonderland). The turns are bundled inline -- quoted verbatim from the
`Project Gutenberg <https://www.gutenberg.org>`_ text -- so the example is
fully offline and deterministic.

**What gets embedded is SPOKEN TEXT ONLY.** Every narrative attribution
("said the Hatter", "Alice replied") and all surrounding narration has been
stripped, so the geometry reflects what the characters *say*, not how the
narrator introduces them -- otherwise every one of Alice's turns would be
pulled together by the repeated words "said Alice" rather than by their
content. For the same reason the caption shows only the quoted line: who is
speaking is carried by the colour, the legend and the speaker label.

**Embeddings & graceful degradation.** Turns are embedded with a
sentence-transformer (``all-MiniLM-L6-v2``) when it is installed; otherwise the
example falls back to a character n-gram TF-IDF vector, so it renders without a
~90 MB model download. Either way the pipeline (embed -> reduce -> disjoint
per-turn paths -> ``animate='serial'`` colored by speaker) is identical.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

import numpy as np

import hypertools as hyp

SPEAKER_COLOR = {
    'Alice': '#E4572E', 'Hatter': '#3F72AF', 'March Hare': '#5B8C5A',
    'Dormouse': '#B5537F',
}

# The turns are CURATED, SPOKEN TEXT ONLY: each entry is exactly what that
# character says out loud, quoted verbatim from the Gutenberg text (emphasis
# underscores dropped), with every narrative attribution ("said the Hatter",
# "Alice replied") and all surrounding narration removed. Automatic extraction
# was tried and rejected: it mis-merged speakers across adjacent quotes (e.g.
# attributing "There's plenty of room!" to the March Hare) and dragged
# narration into the embedded text. What is embedded IS the dialogue.
TURNS = [
    ('Alice', "There's plenty of room!"),
    ('March Hare', "Have some wine."),
    ('Alice', "I don't see any wine."),
    ('March Hare', "There isn't any."),
    ('Alice', "Then it wasn't very civil of you to offer it."),
    ('March Hare', "It wasn't very civil of you to sit down without being invited."),
    ('Alice', "I didn't know it was your table; it's laid for a great many more than three."),
    ('Hatter', "Your hair wants cutting."),
    ('Alice', "You should learn not to make personal remarks; it's very rude."),
    ('Hatter', "Why is a raven like a writing-desk?"),
    ('Alice', "I'm glad they've begun asking riddles. I believe I can guess that."),
    ('March Hare', "Do you mean that you think you can find out the answer to it?"),
    ('Alice', "Exactly so."),
    ('March Hare', "Then you should say what you mean."),
    ('Alice', "I do; at least I mean what I say - that's the same thing, you know."),
    ('Hatter', "Not the same thing a bit! You might just as well say that 'I see what I eat' is the same thing as 'I eat what I see'!"),
    ('March Hare', "You might just as well say that 'I like what I get' is the same thing as 'I get what I like'!"),
    ('Dormouse', "You might just as well say that 'I breathe when I sleep' is the same thing as 'I sleep when I breathe'!"),
    ('Hatter', "It is the same thing with you. What day of the month is it?"),
    ('Alice', "The fourth."),
    ('Hatter', "Two days wrong! I told you butter wouldn't suit the works!"),
    ('March Hare', "It was the best butter."),
    ('Hatter', "Yes, but some crumbs must have got in as well; you shouldn't have put it in with the bread-knife."),
    ('Alice', "What a funny watch! It tells the day of the month, and doesn't tell what o'clock it is!"),
    ('Hatter', "Why should it? Does your watch tell you what year it is?"),
    ('Alice', "Of course not; but that's because it stays the same year for such a long time together."),
    ('Hatter', "Which is just the case with mine."),
    ('Alice', "I don't quite understand you."),
]


def embed(texts):
    """Embed a list of short strings. Prefer a sentence-transformer; fall back
    to a character n-gram TF-IDF vector if it (or its model) is unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return np.asarray(model.encode(texts, show_progress_bar=False),
                          dtype=float)
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                              min_df=1)
        return vec.fit_transform(texts).toarray().astype(float)


def word_spans(text, size=6, step=2, min_wins=3):
    """Sliding word windows over one turn, as ``(start, end)`` index pairs.

    Index pairs rather than joined strings, because the caption bolds the
    words of the window being drawn right now and so has to know which words
    each window covers. What gets embedded is still
    ``' '.join(words[start:end])``.

    ``min_wins`` prevents a real rendering artifact. ``hyp.plot`` draws a
    ONE-ROW dataset as a dot (``marker='.'``, ``linestyle='None'``), because
    there is no line through a single point. With a fixed 6-word window, every
    turn of six words or fewer ("Have some wine.", "Exactly so.", "The
    fourth.") collapses to a single window and shows up as a stray dot floating
    in the box; 12 of the 28 turns below do. Shrinking the window, and the step
    if needed, keeps every turn a real path, which is the whole point here.
    """
    w = text.split()
    n = len(w)
    size = max(1, min(size, n - min_wins + 1))
    step = step if (n - size) // step + 1 >= min_wins else 1
    return [(i, i + size) for i in range(0, n - size + 1, step)]


turn_words = [text.split() for _spk, text in TURNS]
turn_spans = [word_spans(text) for _spk, text in TURNS]
n_wins = [len(spans) for spans in turn_spans]
flat = [' '.join(words[a:b])
        for words, spans in zip(turn_words, turn_spans) for a, b in spans]
vecs = embed(flat)

# reduce every window into one shared 3-D space. The UMAP kwargs suit short
# turns: n_neighbors=8 keeps the neighbor graph local, min_dist=0.5 spreads the
# points so one turn's windows do not collapse onto the next turn's, and
# random_state=1 makes the stochastic embedding reproducible.
red = np.asarray(hyp.reduce(
    vecs, reduce={'model': 'UMAP',
                  'kwargs': {'n_neighbors': 8, 'min_dist': 0.5,
                             'random_state': 1}}, ndims=3))
# no rescaling here: hyp.plot already mean-centers every dataset and rescales
# them into [-1, 1] with ONE shared affine before drawing

# split back into one DISJOINT trajectory per turn, colored by speaker
trajectories, colors, speakers = [], [], []
k = 0
for (spk, _text), nw in zip(TURNS, n_wins):
    trajectories.append(red[k:k + nw])
    k += nw
    colors.append(SPEAKER_COLOR[spk])
    speakers.append(spk)

# THE hypertools call: disjoint per-turn paths, colored by speaker, revealed
# ONE TURN AT A TIME by animate='serial'. On top of it we add a recency fade
# (below): the current turn is opaque and earlier turns get progressively more
# transparent -- a fading tail across the conversation.
#
# duration/frame_rate are the clip's length in seconds and its frames per
# second, so the animation is duration * frame_rate = 192 frames long. That
# product (`total` below) is the frame index every custom per-frame hook here
# is driven by, which is why both are passed explicitly.
duration, fps = 12, 16
fig, ani = hyp.plot(trajectories, fmt='-', color=colors, linewidth=1.6,
                    animate='serial',
                    duration=duration, frame_rate=fps,
                    elev=16, size=(7.6, 7.4), show=False)

import matplotlib.patches as mpatches
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnchoredOffsetbox

present = [s for s in SPEAKER_COLOR if s in speakers]
fig.legend(handles=[mpatches.Patch(color=SPEAKER_COLOR[s], label=s)
                    for s in present],
           loc='upper left', bbox_to_anchor=(0.02, 0.93), frameon=False,
           fontsize=10)
fig.text(0.5, 0.965, "Alice's Mad Tea-Party", ha='center', va='top',
         fontsize=16, fontweight='bold', color='#1a1a1a')
# who is speaking right now, in that speaker's colour, under the title
speaker = fig.text(0.5, 0.923, '', ha='center', va='top', fontsize=13,
                   fontweight='bold')

n_turns = len(trajectories)
total = int(round(fps * duration))
lines = ani._args[1]                                       # one Line3D per turn
# NOTE: ``hyp.plot`` resamples every multi-row LINE dataset onto the frame
# grid, so the DRAWN per-turn row counts are not the original turn lengths
# (1-row turns are left as-is). The serial reveal is paced by those DRAWN
# lengths, so the active turn must be derived from them -- using the original
# lengths makes the opaque highlight lag the turn actually being drawn.
drawn_lens = [np.asarray(a).shape[0] for a in ani._args[0]]
starts = np.cumsum([0] + drawn_lens[:-1])
total_pts = int(sum(drawn_lens))
FLOOR, DECAY = 0.10, 0.45                                   # oldest-turn floor; per-turn fade
# Over the final stretch the whole conversation is lifted back up, so the clip
# ends on the shape it spent the whole run building rather than on one lit turn
# against near-invisible history.
FINALE = int(1.4 * fps)
FINALE_FLOOR = 0.62
_orig = ani._func


def shown_counts(num):
    """Per-turn drawn row counts at this frame, mirroring
    ``update_lines_serial``: ``revealed = total_points * num /
    (total_frames - 1)``."""
    revealed = total_pts * num / max(1, total - 1)
    return [int(np.clip(revealed - st, 0, n))
            for st, n in zip(starts, drawn_lens)]


def current_state(num):
    """The (turn, window) being revealed right now, mirroring
    ``update_lines_serial``: ``revealed = total_points * num /
    (total_frames - 1)``, and a turn is ACTIVE while ``0 < its shown-count <
    its row count``."""
    revealed = total_pts * num / max(1, total - 1)
    done = -1
    for j, (s, n) in enumerate(zip(starts, drawn_lens)):
        # int(clip(...)) mirrors update_lines_serial EXACTLY -- comparing the
        # un-truncated float instead disagrees with the backend on boundary
        # frames, which showed up as one frame where the turn being drawn was
        # faded as history.
        shown = int(np.clip(revealed - s, 0, n))
        if 0 < shown < n:
            # the drawn rows are a resampling of this turn's windows, so map
            # the drawn position back onto a window index
            frac = (shown - 1) / max(1, n - 1)
            return j, int(round(frac * (n_wins[j] - 1)))     # actively drawing
        if shown >= n:
            done = j                                        # fully revealed
    if done < 0:
        # nothing drawn yet (frame 0). Falling through to the "between turns"
        # branch here reported the LAST window of turn 0, so the very first
        # frame bolded the end of the line and frame 1 snapped back to its
        # start.
        return 0, 0
    return done, n_wins[done] - 1                           # between turns


def caption_lines(ti, wi, width=68):
    """The turn's words as ``[[(word, is_bold), ...], ...]`` -- one list per
    wrapped line -- with the words of the window being drawn RIGHT NOW bold."""
    words = list(turn_words[ti])
    a, b = turn_spans[ti][wi]
    words[0] = '“' + words[0]
    words[-1] = words[-1] + '”'
    rows, row, used = [], [], 0
    for k, word in enumerate(words):
        step = len(word) + (1 if row else 0)
        if row and used + step > width:
            rows.append(row)
            row, used, step = [], 0, len(word)
        row.append((word, a <= k < b))
        used += step
    if row:
        rows.append(row)
    return rows


# The caption mixes bold and regular runs on ONE line, which a single Text
# artist cannot do, so it is built from per-word TextAreas packed into rows and
# rebuilt every frame. ``sep`` is the width of a space at this font size
# (0.318 em in DejaVu Sans).
CAP_FS = 12
caption = [None]                                            # current artist


def set_caption(rows, color):
    if caption[0] is not None:
        caption[0].remove()
    packed = [HPacker(children=[
        TextArea(word, textprops=dict(color=color, fontsize=CAP_FS,
                                      style='italic',
                                      fontweight='bold' if bold else 'normal'))
        for word, bold in row], align='baseline', pad=0, sep=0.318 * CAP_FS)
        for row in rows]
    box = AnchoredOffsetbox(loc='lower center', pad=0, frameon=False,
                            child=VPacker(children=packed, align='center',
                                          pad=0, sep=4),
                            bbox_to_anchor=(0.5, 0.05),
                            bbox_transform=fig.transFigure)
    fig.add_artist(box)
    caption[0] = box


def _wrapped(num, *args):
    result = _orig(num, *args)
    ti, wi = current_state(num)
    # recency fade: the current turn is opaque; earlier turns get progressively
    # more transparent (a fading tail) down to a floor so the whole shape stays
    # visible; not-yet-spoken turns are hidden.
    counts = shown_counts(num)
    ramp = min(1.0, max(0.0, (num - (total - 1 - FINALE)) / max(1, FINALE)))
    floor = FLOOR + (FINALE_FLOOR - FLOOR) * ramp
    for j, ln in enumerate(lines):
        if j > ti or counts[j] < 2:
            # not yet spoken, or only ONE point drawn so far. hyp.plot renders
            # a single drawn point as a lone dot (there is no line through one
            # point), which flashes as a speck for the one frame at each turn
            # boundary.
            ln.set_alpha(0.0)
        elif j == ti:
            ln.set_alpha(1.0)
        else:
            ln.set_alpha(floor + (1.0 - floor) * DECAY ** (ti - j))
    # who is speaking, then the SPOKEN LINE only -- no attribution is tacked
    # onto the quote -- with the window being drawn right now in bold
    spk = speakers[ti]
    speaker.set_text(spk)
    speaker.set_color(SPEAKER_COLOR[spk])
    set_caption(caption_lines(ti, wi), SPEAKER_COLOR[spk])
    return result


ani._func = _wrapped
