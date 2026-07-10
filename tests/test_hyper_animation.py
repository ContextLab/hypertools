"""Animated hyp.plot returns a HyperAnimation (QC 2026-07, Jeremy's notes
sections 10-11).

Jeremy's "pure failure": `anim = hyp.plot(X, animate='spin', show=False)` then
`anim.to_html5_video()` raised "'Figure'/'tuple' object has no attribute
to_html5_video" -- animated plots returned a bare (fig, animation) tuple.
Animated plots now return a HyperAnimation: a tuple subclass that IS
(figure, animation) -- so every legacy pattern (unpacking, indexing,
isinstance-tuple) keeps working -- with .to_html5_video()/.to_jshtml()/.save()
and inline auto-play (_repr_html_).

Also section 10: animate='chemtrails'/'precog'/'bullettime' silently produced a
STATIC plot (they are trail-effect flags, not styles); they now animate. An
unknown animate style raises a clear error instead of a silent static plot.

Real data, no mocks; headless (Agg).
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import hypertools as hyp
from hypertools import HyperAnimation


def _traj(n=30, d=3):
    return np.random.default_rng(0).normal(size=(n, d)).cumsum(axis=0)


# --- HyperAnimation return -----------------------------------------------

def test_animated_plot_returns_hyperanimation():
    out = hyp.plot(_traj(), ndims=3, animate='spin', duration=2, show=False)
    assert isinstance(out, HyperAnimation)
    # Jeremy's exact failing call now works
    assert hasattr(out, 'to_html5_video')
    assert hasattr(out, 'save')
    assert out.figure is out[0]
    assert out.animation is out[1]


def test_hyperanimation_is_backwards_compatible_tuple():
    out = hyp.plot(_traj(), ndims=3, animate='spin', duration=2, show=False)
    # legacy code did all of these
    assert isinstance(out, tuple)
    assert len(out) == 2
    fig, ani = out
    import matplotlib.figure
    import matplotlib.animation
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ani, matplotlib.animation.Animation)


def test_hyperanimation_repr_html_returns_playable_markup():
    out = hyp.plot(_traj(), ndims=3, animate='spin', duration=2, show=False)
    html = out._repr_html_()
    assert isinstance(html, str) and len(html) > 0


def test_hyperanimation_save_writes_gif(tmp_path):
    out = hyp.plot(_traj(), ndims=3, animate='spin', duration=1, show=False)
    path = tmp_path / 'anim.gif'
    out.save(str(path), writer='pillow', fps=10)
    assert path.exists() and path.stat().st_size > 0


def test_static_plot_still_returns_bare_figure():
    import matplotlib.figure
    out = hyp.plot(_traj(), ndims=3, show=False)
    assert isinstance(out, matplotlib.figure.Figure)
    assert not isinstance(out, HyperAnimation)


def test_return_model_bundle_animation_is_raw_animation():
    import matplotlib.animation
    bundle = hyp.plot(_traj(), ndims=3, animate='spin', duration=2,
                      return_model=True, show=False)
    assert isinstance(bundle['animation'], matplotlib.animation.Animation)


# --- chemtrails/precog/bullettime as animate values ----------------------

@pytest.mark.parametrize('style', ['chemtrails', 'precog', 'bullettime'])
def test_trail_effect_as_animate_style_animates(style):
    # used to silently produce a STATIC Figure (not in the style whitelist)
    out = hyp.plot(_traj(), ndims=3, animate=style, duration=2, show=False)
    assert isinstance(out, HyperAnimation)
    assert hasattr(out, 'to_html5_video')


def test_unknown_animate_style_raises_clear_error():
    with pytest.raises(ValueError, match='unknown animate style'):
        hyp.plot(_traj(), ndims=3, animate='wobble', show=False)


@pytest.mark.parametrize('style', ['spin', 'parallel', 'window', 'serial'])
def test_valid_animate_styles_still_work(style):
    out = hyp.plot(_traj(), ndims=3, animate=style, duration=2, show=False)
    assert isinstance(out, HyperAnimation)
