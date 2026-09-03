# -*- coding: utf-8 -*-
"""
===========================================================================
The shape of a conversation: per-speaker paths, revealed one turn at a time
===========================================================================

A conversation as geometry, in one ``hyp.plot`` call on raw dialogue. Each
**turn** (a contiguous run of speech by one speaker) is cut into sliding
word windows, and the turns are handed to ``hyp.plot`` as a **list of lists
of strings**: the nesting is the grouping, so every turn becomes its own
disjoint trajectory through one shared 3-D space. The call embeds every
window with a sentence-transformer (``vectorizer='all-MiniLM-L6-v2'``),
reduces them together with UMAP, colours each path by **speaker** through a
categorical ``hue=`` with a native legend, and ``order='serial'`` reveals
the turns one at a time with ``chemtrails=True`` leaving the spoken path
behind each head. ``title=`` carries one string per turn, so the label
above the box always names who is speaking and what they say -- driven by
the library's own reveal schedule, not re-derived here.

The one bespoke effect left is a **recency fade** across turns: the current
turn is opaque, earlier turns recede, unspoken turns are hidden. Nothing in
1.1 fades across already-revealed datasets, so it is real custom work -- but
it runs on the public ``on_frame`` hook and reads the schedule the library
publishes (``ctx.current_index``, ``ctx.revealed_counts``). Before 1.1 this
example monkeypatched ``ani._func`` and re-derived that schedule by hand;
the hook replaces both.

Here the conversation is Lewis Carroll's *Mad Tea-Party* (Alice in
Wonderland). The turns are bundled inline -- quoted verbatim from the
`Project Gutenberg <https://www.gutenberg.org>`_ text -- so the example is
fully offline and deterministic.

**What gets embedded is SPOKEN TEXT ONLY.** Every narrative attribution
("said the Hatter", "Alice replied") and all surrounding narration has been
stripped, so the geometry reflects what the characters *say*, not how the
narrator introduces them -- otherwise every one of Alice's turns would be
pulled together by the repeated words "said Alice" rather than by their
content. Who is speaking is carried by the colour, the legend and the title.

**Embeddings & graceful degradation.** Text embedding needs the ``[text]``
extra (``pip install "hypertools[text]"``); without it,
``vectorizer='TfidfVectorizer'`` is used, and the pipeline (embed -> reduce
-> disjoint per-turn paths -> serial reveal coloured by speaker) is
identical either way.
"""

# Code source: Contextual Dynamics Laboratory
# License: MIT

from typing import NamedTuple

import hypertools as hyp

try:
    import sentence_transformers  # noqa: F401 -- only asks whether [text] is installed
    VECTORIZER = 'all-MiniLM-L6-v2'
except ImportError:
    VECTORIZER = 'TfidfVectorizer'

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

WINDOW, STEP, MIN_WINDOWS = 6, 2, 3
FLOOR, DECAY = 0.10, 0.45               # oldest-turn floor; per-turn fade


class Conversation(NamedTuple):
    turns: list                 # per turn, its list of word windows
    speakers: list              # per turn, who speaks it
    texts: list                 # per turn, the spoken line (for the title)
    vectorizer: str             # how the windows are embedded
    source: str                 # which path produced them


def windows(text, size=WINDOW, step=STEP, min_windows=MIN_WINDOWS):
    """Sliding word windows over one turn.

    ``min_windows`` prevents a real rendering artifact: ``hyp.plot`` draws a
    ONE-ROW dataset as a dot (there is no line through a single point), and
    with a fixed 6-word window, 12 of the 28 turns above collapse to a
    single window and would show up as stray specks. Shrinking the window,
    and the step if needed, keeps every turn a real path.
    """
    words = text.split()
    n = len(words)
    size = max(1, min(size, n - min_windows + 1))
    step = step if (n - size) // step + 1 >= min_windows else 1
    return [' '.join(words[i:i + size]) for i in range(0, n - size + 1, step)]


# --- the data half. Nothing here fetches: the text is inline, and the only
# --- network access an embedding can need (the model) is named, not run.
def embed_turns(spec=TURNS, vectorizer=VECTORIZER):
    """Window each turn and name the vectorizer the figure should use."""
    return Conversation([windows(text) for _speaker, text in spec],
                        [speaker for speaker, _text in spec],
                        [text for _speaker, text in spec],
                        vectorizer, 'inline Gutenberg text')


def fixture_data():
    """The same payload embedded with a deterministic sklearn TF-IDF fit:
    no model download. What the test-suite drives."""
    return embed_turns(TURNS, 'TfidfVectorizer')


# --- the figure half: no network, deterministic given its input -------------
def turn_alpha(i, revealed, current):
    """How visible turn `i` should be while turn `current` is being drawn.

    Assigns a value for EVERY dataset on EVERY frame, including turns not
    yet spoken -- the portable callback rule (animation.rst): put the
    condition in the VALUE, never around the assignment. A skipped
    assignment leaves matplotlib's shared artists at whatever the previous
    frame set, which is how a fade turns into a smear.
    """
    if i > current or revealed < 2:
        return 0.0                     # unspoken, or a single stray point
    if i == current:
        return 1.0
    return FLOOR + (1.0 - FLOOR) * DECAY ** (current - i)


def recency_fade(ctx):
    """The one bespoke effect left: earlier turns recede as the talk moves on.

    ``chemtrails``/``precog``/``bullettime`` fade WITHIN one trajectory;
    nothing in 1.1 fades ACROSS already-revealed datasets, so this is real
    custom work -- but it runs on the public per-frame hook and reads the
    library's own published schedule instead of re-deriving it.

    ``ctx.artists`` is NOT one artist per dataset. It is heads first, then
    trails (animation_context.FrameContext), so with ``chemtrails=True`` it
    holds 2N entries against ``revealed_counts``' N. Zipping the two
    directly walks off the end of the counts. Split by role first.
    """
    current = ctx.current_index
    if current is None:
        raise RuntimeError(
            "recency_fade needs a serial reveal: ctx.current_index is None, "
            "which means this plot is animating in parallel. Keep "
            "order='serial' (or animate='serial') on the plot() call.")
    n_datasets = len(ctx.revealed_counts)
    heads = ctx.artists[:n_datasets]
    trails = ctx.artists[n_datasets:]
    # chemtrails=True is broadcast to every dataset, so this holds here. It
    # is asserted rather than assumed because a dataset drawn marker-only
    # gets no trail artist, and the mismatch would otherwise show up as a
    # silently mis-paired head/trail rather than an error.
    if len(trails) != n_datasets:
        raise RuntimeError(
            f"expected one trail artist per dataset, got {len(trails)} "
            f"trails for {n_datasets} datasets")
    for i, (head, trail, revealed) in enumerate(
            zip(heads, trails, ctx.revealed_counts)):
        alpha = turn_alpha(i, revealed, current)
        head.set_alpha(alpha)
        # NOT the library's 0.3x trail convention. On a serial reveal the
        # trail IS the part of the current turn already spoken (measured:
        # 821 of its points against a 6-point head), so at 0.3x the turn
        # being spoken was the faintest thing on screen. It fades with its
        # head, one value per turn.
        trail.set_alpha(alpha)


def construct_artifact(data):
    """`data.turns` / `data.speakers` in, the animation out. Returns the
    HyperAnimation wrapper: `.on_frame()` lives on it, not on the
    FuncAnimation that `fig, ani = ...` would give."""
    # category order is FIRST APPEARANCE, not alphabetical, so the palette
    # must be listed in that order for each speaker to get their colour
    order = list(dict.fromkeys(data.speakers))
    # THE hypertools call: raw dialogue in, one disjoint trajectory per
    # turn, coloured by speaker, revealed ONE TURN AT A TIME.
    anim = hyp.plot(
        data.turns, '-',
        vectorizer=data.vectorizer, semantic=None, corpus=None,
        reduce={'model': 'UMAP', 'kwargs': {'n_neighbors': 8, 'min_dist': 0.5,
                                            'random_state': 1}},
        ndims=3,
        hue=[[speaker] * len(turn)
             for speaker, turn in zip(data.speakers, data.turns)],
        palette=[SPEAKER_COLOR[s] for s in order], legend=True,
        linewidth=1.6,
        animate=True, order='serial', chemtrails=True,
        title=[f'{speaker}   \u201c{text}\u201d'
               for speaker, text in zip(data.speakers, data.texts)],
        duration=12, frame_rate=16, elev=16, size=(8, 8), show=False)
    anim.on_frame(recency_fade)
    return anim


if __name__ == '__main__':
    conversation = embed_turns()
    print(f'conversation: {len(conversation.turns)} turns, '
          f'{len(set(conversation.speakers))} speakers, '
          f'{sum(len(t) for t in conversation.turns)} windows, embedded with '
          f'{conversation.vectorizer}')
    anim = construct_artifact(conversation)
    fig = anim.figure
