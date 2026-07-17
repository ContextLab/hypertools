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


def test_return_model_bundle_animation_gc_does_not_warn():
    # release-1.0 audit follow-up to X4-warnings-012: the return_model bundle
    # hands back the RAW FuncAnimation (asserted raw by the test above), so
    # HyperAnimation.__del__'s silencing never applied to it. The animation is
    # kept in a reference cycle by its own canvas callbacks, so discarding the
    # bundle leaked matplotlib's "Animation was deleted without rendering
    # anything" UserWarning at the NEXT cyclic-gc pass -- attributed to
    # whatever test happened to be running then (29 scattered instances in the
    # full suite). plot.py now marks the bundled animation via
    # hyper_animation.mark_draw_started() at hand-off, exactly like the
    # wrapper's __del__ does.
    import gc
    import warnings
    bundle = hyp.plot(_traj(), ndims=3, animate='spin', duration=2,
                      return_model=True, show=False)
    del bundle
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        gc.collect()
    leaked = [w for w in rec
              if 'deleted without rendering' in str(w.message)]
    assert not leaked, f"unrendered-animation warning leaked at gc: {leaked}"


def test_failed_save_path_animation_gc_does_not_warn(tmp_path):
    # release-1.0 audit follow-up to X4-warnings-012 (second gap): when
    # plot(..., save_path=...) raises (e.g. unsupported extension), the
    # FuncAnimation already exists but the HyperAnimation wrapper is never
    # constructed, so the abandoned animation leaked the same "deleted
    # without rendering" warning at the next cyclic-gc pass. plot.py's
    # save-failure cleanup now marks it via mark_draw_started() before
    # re-raising.
    import gc
    import warnings
    with pytest.raises(ValueError, match='unsupported animation save'):
        hyp.plot(_traj(), ndims=3, animate='spin', duration=2, show=False,
                 save_path=str(tmp_path / 'x.html'))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        gc.collect()
    leaked = [w for w in rec
              if 'deleted without rendering' in str(w.message)]
    assert not leaked, f"unrendered-animation warning leaked at gc: {leaked}"


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
