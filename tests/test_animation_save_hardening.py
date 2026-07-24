"""HyperAnimation.save format coverage + animation kwarg validation
(QC 2026-07 release hunt).

- anim.save('x.svg') / .save('x.png') crashed (raw Animation.save tried to pipe
  h264 into an svg/png), though the same extensions work via save_path=;
- duration=0 / frame_rate=0 raised ZeroDivisionError; a negative duration a
  cryptic "zero-size array" error.

Real rendering (writes files), no mocks; headless (Agg).
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import hypertools as hyp
from hypertools import HyperAnimation


def _traj():
    return np.random.default_rng(0).normal(size=(20, 3)).cumsum(axis=0)


@pytest.mark.parametrize('ext', ['gif', 'svg', 'png'])
def test_hyperanimation_save_all_supported_formats(tmp_path, ext):
    anim = hyp.plot(_traj(), ndims=3, animate='spin', duration=1, show=False)
    assert isinstance(anim, HyperAnimation)
    out = tmp_path / f'anim.{ext}'
    anim.save(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_hyperanimation_save_explicit_writer_still_works(tmp_path):
    anim = hyp.plot(_traj(), ndims=3, animate='spin', duration=1, show=False)
    out = tmp_path / 'anim_writer.gif'
    anim.save(str(out), writer='pillow', fps=10)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize('kwargs,match', [
    ({'duration': 0}, 'duration must be'),
    ({'duration': -1}, 'duration must be'),
    ({'frame_rate': 0}, 'frame_rate must be'),
])
def test_animation_nonpositive_duration_frame_rate_clear_error(kwargs, match):
    with pytest.raises(ValueError, match=match):
        hyp.plot(_traj(), ndims=3, animate=True, show=False, **kwargs)


def test_valid_animation_still_works():
    anim = hyp.plot(_traj(), ndims=3, animate='spin', duration=2, frame_rate=15,
                    show=False)
    assert isinstance(anim, HyperAnimation)
