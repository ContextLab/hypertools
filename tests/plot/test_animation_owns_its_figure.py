# -*- coding: utf-8 -*-
"""An animated `hyp.plot` builds its own figure -- say so, loudly.

Found while prototyping Plan 4's Market composition (review round 13): every
animated mode IGNORES an explicitly passed `ax=`. It builds a new figure,
draws there, returns that figure, and leaves the caller's axes empty. The
caller gets a perfectly good animation of the right data in the wrong
place, and nothing says a word.

Measured before this test existed, across every mode `hyp.plot` accepts:

    parallel serial spin chemtrails precog bullettime morph
      -> returned figure is the caller's:  False (all seven)
      -> the caller's axes has artists:    False (all seven)

so the rule is universal rather than mode-specific, which is what makes an
unconditional error the right response instead of a per-mode special case.
`ax=` with a 3-D static plot already raises in this file's neighbour
(`"If passing ax and the plot is 3D, ax must also be 3d"`), so refusing an
unsupportable `ax=` is the established convention here, not a new one.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
import pytest                                                  # noqa: E402

import hypertools as hyp                                       # noqa: E402


def _walk(rows=24, cols=2, seed=0):
    return np.cumsum(
        np.random.default_rng(seed).standard_normal((rows, cols)), axis=0)


ANIMATED_MODES = ['parallel', 'serial', 'spin', 'chemtrails', 'precog',
                  'bullettime', 'window']


@pytest.mark.parametrize('mode', ANIMATED_MODES)
def test_ax_with_an_animated_plot_RAISES_instead_of_drawing_elsewhere(mode):
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match='animated .* own figure'):
            hyp.plot(_walk(), animate=mode, duration=1, frame_rate=4,
                     ndims=2, reduce=None, ax=ax, show=False)
    finally:
        plt.close(fig)


def test_the_message_says_what_to_do_instead():
    """A refusal that does not name the fix just moves the confusion."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError) as caught:
            hyp.plot(_walk(), animate='parallel', duration=1, frame_rate=4,
                     ndims=2, reduce=None, ax=ax, show=False)
    finally:
        plt.close(fig)
    message = str(caught.value)
    assert 'ax=' in message
    assert '.figure' in message, 'point the caller at the figure they DO get'


def test_morph_refuses_it_too():
    """`animate='morph'` takes a different code path to the other modes."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match='animated .* own figure'):
            hyp.plot([_walk(), _walk(seed=1)], animate='morph', duration=1,
                     frame_rate=4, ndims=2, reduce=None, ax=ax, show=False)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('falsy', [None, False])
def test_a_STATIC_plot_still_honours_ax(falsy):
    """The guard must not cost the static path its `ax=`, which works and is
    used throughout the gallery and the tutorials."""
    fig, ax = plt.subplots()
    try:
        hyp.plot(_walk(), '-', animate=falsy, ndims=2, reduce=None,
                 ax=ax, show=False)
        assert len(ax.lines) + len(ax.collections) > 0, (
            'a static plot must still draw into the axes it was given')
    finally:
        plt.close(fig)


def test_an_animated_plot_WITHOUT_ax_is_untouched():
    """The guard fires on the combination, not on animation."""
    out = hyp.plot(_walk(), animate='parallel', duration=1,
                   frame_rate=4, ndims=2, reduce=None, show=False)
    try:
        assert out.n_frames == 4
        assert isinstance(out.figure, plt.Figure)
    finally:
        plt.close(out.figure)


def test_the_figure_an_animation_returns_is_its_own():
    """The positive statement behind the error message: the animation hands
    back a figure, and that figure is the one it drew on."""
    out = hyp.plot(_walk(), animate='parallel', duration=1,
                   frame_rate=4, ndims=2, reduce=None, show=False)
    try:
        assert out.figure is out.animation._fig
        assert len(out.figure.axes) == 1
    finally:
        plt.close(out.figure)
