#!/usr/bin/env python
"""Sliding windows over text (GH #285).

Turning one document into a *trajectory* means cutting it into overlapping
pieces and treating each piece as an observation. Three different chunkers
were written by hand across the examples and tutorials -- word windows with
a min-windows guard, equal-length character chunks, and 3-sentence windows
-- and `text_windows` is the single implementation of all three.
"""

import re

__all__ = ['text_windows']

#: Sentence boundary: a `.`, `!` or `?` immediately followed by whitespace.
#: Deliberately dependency-free -- see the ``unit='sentences'`` limits in
#: `text_windows`'s Notes.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')

_UNITS = ('words', 'sentences', 'chars')


def _tokenize(text, unit):
    """Split `text` into its `unit`s, and say how to glue them back."""
    if unit == 'words':
        return text.split(), ' '
    if unit == 'sentences':
        pieces = [s.strip() for s in _SENTENCE_BOUNDARY.split(text)]
        return [s for s in pieces if s], ' '
    if unit == 'chars':
        return list(text), ''
    raise ValueError(
        f"unit= must be one of {_UNITS}; got {unit!r}.")


def _check_positive_int(value, name, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int,)):
        raise TypeError(f"{name}= must be an integer; got {value!r}.")
    if value < 1:
        raise ValueError(f"{name}= must be at least 1; got {value!r}.")
    return int(value)


def _windows_one(text, size, step, unit, min_windows, max_chars):
    """The single-string case: the loop every hand-written version ran."""
    if not isinstance(text, str):
        raise TypeError(
            "text_windows takes a string, or a list/tuple of strings (one "
            f"per document); got an element of type {type(text).__name__}.")

    if max_chars is not None:
        text = text[:max_chars]

    tokens, joiner = _tokenize(text, unit)
    n = len(tokens)
    if n == 0:
        # Nothing to window. `min_windows` cannot manufacture observations
        # out of an empty document, so this is the one case that returns
        # fewer than `min_windows` windows by construction.
        return []

    width = n if size is None else size
    # The min-windows guard, verbatim from examples/animate_conversation.py:
    # shrink the window until `min_windows` of them fit, and drop the step
    # to 1 if striding would still leave too few.
    width = max(1, min(width, n - min_windows + 1))
    if (n - width) // step + 1 < min_windows:
        step = 1
    return [joiner.join(tokens[i:i + width])
            for i in range(0, n - width + 1, step)]


def text_windows(text, size=None, step=1, unit='words', min_windows=1,
                 max_chars=None):
    """Cut text into sliding windows: one observation per window.

    A single string becomes a list of window strings -- a trajectory
    through whatever space the vectorizer defines. A list of strings becomes
    a list of such lists, one trajectory per document, which is exactly the
    "list of lists of strings" shape `hypertools.plot` reads as one disjoint
    trajectory per document.

    Parameters
    ----------
    text : str, or list/tuple of str
        The document(s) to window. A list returns a list of lists (one per
        document); the nesting is never flattened.

    size : int or None, optional
        Window width, in `unit`s. ``None`` (the default) means "the whole
        document": one window per document, which -- with `max_chars` -- is
        the truncation-only form used by the Wikipedia tutorial.

    step : int, optional
        How many `unit`s to advance between consecutive windows (default 1,
        i.e. maximally overlapping). ``step=size`` gives disjoint chunks.

    unit : {'words', 'sentences', 'chars'}, optional
        What a window is measured in. ``'words'`` splits on whitespace and
        rejoins with a single space (so runs of whitespace are normalized);
        ``'sentences'`` uses the regex described in the Notes; ``'chars'``
        slices the raw string, preserving it byte for byte.

    min_windows : int, optional
        The smallest number of windows a document should yield (default 1).
        A short document has its window (and, if that is not enough, its
        step) shrunk until at least this many windows fit. This is a real
        rendering guard: `hypertools.plot` draws a one-row dataset as a dot,
        so a turn that collapses to a single window shows up as a stray
        speck rather than a path. It is best-effort: a document of `n`
        `unit`s can never yield more than `n` windows, and an empty document
        yields none.

    max_chars : int or None, optional
        Truncate each document to its first `max_chars` characters *before*
        windowing. Sentence-embedding models read a bounded number of
        tokens, so this both fits the model and speeds encoding up.

    Returns
    -------
    windows : list of str, or list of list of str
        One list of window strings per input document; a bare string in
        gives a bare list out.

    Notes
    -----
    ``unit='sentences'`` splits on ``(?<=[.!?])\\s+`` -- a terminator
    immediately followed by whitespace -- and carries no abbreviation
    dictionary, so ``'Dr. Smith'``, ``'e.g. this'`` and ``'U.S. law'`` each
    split in the middle. A terminator followed by a closing quote or
    parenthesis (``'"Stop." He left.'``) does not split at the quote,
    because the character before the whitespace is not a terminator. Pass
    pre-split sentences (or use a real sentence segmenter) when those cases
    matter.

    The equal-chunk idiom -- "cut this article into 5 pieces" -- is
    ``text_windows(s, size=len(s) // 5, step=len(s) // 5, unit='chars')``.
    Like the hand-written version it drops a trailing remainder shorter than
    one window.

    Examples
    --------
    Overlapping word windows -- one observation per window:

    >>> from hypertools.tools import text_windows
    >>> text_windows('the quick brown fox jumps', size=3)
    ['the quick brown', 'quick brown fox', 'brown fox jumps']

    A stride, and the min-windows guard shrinking a window that does not
    fit:

    >>> text_windows('the quick brown fox jumps', size=3, step=2)
    ['the quick brown', 'brown fox jumps']
    >>> text_windows('one two three', size=6, min_windows=3)
    ['one', 'two', 'three']

    Sentences, characters, and truncation:

    >>> text_windows('A. B. C. D.', size=2, unit='sentences')
    ['A. B.', 'B. C.', 'C. D.']
    >>> text_windows('abcdef', size=2, step=2, unit='chars')
    ['ab', 'cd', 'ef']
    >>> text_windows(['a long article...'], max_chars=6)
    [['a long']]

    One trajectory per document:

    >>> [len(doc) for doc in
    ...  text_windows(['one two three four', 'five six'], size=2)]
    [3, 1]

    See Also
    --------
    hypertools.plot : accepts a list of lists of strings, one trajectory per
        document.
    """
    size = _check_positive_int(size, 'size', allow_none=True)
    step = _check_positive_int(step, 'step')
    min_windows = _check_positive_int(min_windows, 'min_windows')
    max_chars = _check_positive_int(max_chars, 'max_chars', allow_none=True)
    if unit not in _UNITS:
        raise ValueError(f"unit= must be one of {_UNITS}; got {unit!r}.")

    if isinstance(text, str):
        return _windows_one(text, size, step, unit, min_windows, max_chars)
    if isinstance(text, (list, tuple)):
        return [_windows_one(item, size, step, unit, min_windows, max_chars)
                for item in text]
    raise TypeError(
        "text_windows takes a string, or a list/tuple of strings (one per "
        f"document); got {type(text).__name__}.")
