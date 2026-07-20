# -*- coding: utf-8 -*-
"""Animation export tests: real files, verified frame counts."""

import os
import shutil

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import hypertools as hyp


walk = np.cumsum(np.random.default_rng(0).standard_normal((50, 5)), axis=0)
overlap = np.vstack([np.random.default_rng(42).standard_normal((100, 5))
                     + 1.5 * i for i in range(3)])

HAS_FFMPEG = shutil.which('ffmpeg') is not None


def _animated_frames(path):
    with Image.open(path) as im:
        return getattr(im, 'n_frames', 1)


def _gif_playback(path):
    """(n_frames, total_playback_seconds) read from an exported gif's
    per-frame delays -- the real-time duration a viewer experiences."""
    with Image.open(path) as im:
        n = getattr(im, 'n_frames', 1)
        total_ms = 0
        for i in range(n):
            im.seek(i)
            total_ms += im.info.get('duration', 0)
    return n, total_ms / 1000.0


def test_matplotlib_gif_export(tmp_path):
    out = str(tmp_path / 'anim.gif')
    hyp.plot(walk, animate=True, duration=2, frame_rate=10,
             save_path=out, show=False)
    plt.close('all')
    assert os.path.getsize(out) > 0
    assert _animated_frames(out) > 1


def test_matplotlib_apng_export(tmp_path):
    out = str(tmp_path / 'anim.png')
    hyp.plot(walk, animate=True, duration=2, frame_rate=10,
             save_path=out, show=False)
    plt.close('all')
    assert _animated_frames(out) > 1  # animated PNG, not a static frame


@pytest.mark.skipif(not HAS_FFMPEG, reason='ffmpeg not installed')
def test_matplotlib_mp4_export(tmp_path):
    out = str(tmp_path / 'anim.mp4')
    hyp.plot(walk, animate=True, duration=2, frame_rate=10,
             save_path=out, show=False)
    plt.close('all')
    assert os.path.getsize(out) > 1000


def test_plotly_gif_export(tmp_path):
    out = str(tmp_path / 'anim.gif')
    hyp.plot(walk, animate=True, duration=2, backend='plotly',
             save_path=out, show=False)
    assert _animated_frames(out) > 1


def test_plotly_spin_gif_export(tmp_path):
    out = str(tmp_path / 'spin.gif')
    hyp.plot(walk, animate='spin', duration=2, backend='plotly',
             save_path=out, show=False)
    assert _animated_frames(out) > 1


@pytest.mark.skipif(not HAS_FFMPEG, reason='ffmpeg not installed')
def test_plotly_mp4_export(tmp_path):
    out = str(tmp_path / 'anim.mp4')
    hyp.plot(walk, animate=True, duration=2, backend='plotly',
             save_path=out, show=False)
    assert os.path.getsize(out) > 1000


# --- exported gifs must preserve real-time playback duration ---------------
# Regression guard for the "exported gifs play ~6x too fast" bug: an exported
# gif must contain the FULL frame set (frame_rate * duration frames) at the
# true per-frame delay (1000 / frame_rate ms), so total playback ~= duration
# seconds. frame_rate=10 -> 100 ms/frame, which the gif format's 10 ms
# (centisecond) delay granularity stores exactly, keeping the assertion tight.

def test_matplotlib_spin_gif_preserves_realtime_duration(tmp_path):
    duration, frame_rate = 2, 10
    out = str(tmp_path / 'spin.gif')
    hyp.plot(walk, animate='spin', duration=duration, frame_rate=frame_rate,
             save_path=out, show=False)
    plt.close('all')
    n, total = _gif_playback(out)
    assert n == duration * frame_rate       # full frame set, not subsampled
    assert abs(total - duration) <= 0.25 * duration   # ~real-time, not 6x fast


def test_plotly_spin_gif_preserves_realtime_duration(tmp_path):
    duration, frame_rate = 2, 10
    out = str(tmp_path / 'spin.gif')
    hyp.plot(walk, animate='spin', duration=duration, frame_rate=frame_rate,
             backend='plotly', save_path=out, show=False)
    n, total = _gif_playback(out)
    assert n == duration * frame_rate       # full frame set, not subsampled
    assert abs(total - duration) <= 0.25 * duration   # ~real-time, not 6x fast


# --- animated legends must not duplicate in-focus entries with their trails ---
# Regression guard for "each label appears twice": animated line plots draw a
# faint alpha=0.3 trail artist per dataset in addition to the in-focus window.
# Both used to carry the dataset's label, so `ax.legend()` collected each label
# twice (once for the window, once for the tail). Only the in-focus lines should
# appear in the legend.

def _animated_legend_labels(fig):
    lg = fig.axes[0].get_legend()
    assert lg is not None, 'no legend was drawn'
    return [t.get_text() for t in lg.get_texts()]


@pytest.mark.parametrize('extra', [{}, {'chemtrails': True}])
def test_animation_legend_has_no_duplicate_trail_entries(extra):
    a = np.cumsum(np.random.default_rng(1).standard_normal((40, 3)), axis=0)
    b = np.cumsum(np.random.default_rng(2).standard_normal((40, 3)), axis=0)
    fig, _ani = hyp.plot([a, b], animate=True, legend=['first', 'second'],
                         show=False, **extra)
    labels = _animated_legend_labels(fig)
    plt.close('all')
    assert labels == ['first', 'second']  # exactly one entry per dataset


def test_serial_animation_legend_is_static_union():
    """Serial animation brings datasets into focus one at a time, but the
    legend is built once from the upfront line artists, so it shows the
    union of all in-focus datasets and never changes across frames."""
    a = np.cumsum(np.random.default_rng(1).standard_normal((30, 3)), axis=0)
    b = np.cumsum(np.random.default_rng(2).standard_normal((30, 3)), axis=0)
    c = np.cumsum(np.random.default_rng(3).standard_normal((30, 3)), axis=0)
    fig, _ani = hyp.plot([a, b, c], animate='serial', duration=2,
                         legend=['x', 'y', 'z'], show=False)
    labels = _animated_legend_labels(fig)
    plt.close('all')
    assert labels == ['x', 'y', 'z']


def test_mixture_soft_membership_on_overlapping_clusters():
    """Overlapping blobs must yield genuinely MIXED memberships -- the
    mixture demo requirement: a substantial fraction of points belong
    meaningfully to more than one component."""
    props = hyp.cluster(overlap, cluster='GaussianMixture', n_clusters=3)
    mixed = np.mean(props.max(axis=1) < 0.9)
    assert mixed > 0.15, (
        f'only {mixed:.0%} of points show mixed membership; overlap data '
        'should produce substantially soft assignments')


# --- kaleido/Chrome wedge robustness (Windows-CI hang guard) ---------------
# kaleido 1.x can hang a to_image() call UNBOUNDED (its own timeout only wraps
# the figure calc, not browser launch/tab acquisition), which stalled Windows
# CI at the 20-min per-test limit. A blocked native/browser call cannot be
# interrupted from a Python thread, so frame rendering runs in a KILLABLE
# subprocess with a hard wall-clock deadline; on overrun the whole process tree
# (Chrome included) is killed and the export retried in a fresh subprocess.
# These exercise that lifecycle -- including a genuinely-blocked renderer, its
# bounded termination, and a real export afterwards in the same parent process.

import importlib
import subprocess
import sys
import time
# the `hypertools.plot` submodule name is shadowed by the `plot` function, so
# reach the backend module via importlib rather than a dotted import
_pb = importlib.import_module('hypertools.plot.plotly_backend')


def _small_plotly_anim_fig():
    """A real animated plotly Figure with a handful of frames (for the export
    subprocess to render)."""
    w3 = np.cumsum(np.random.default_rng(0).standard_normal((10, 3)), axis=0)
    return hyp.plot(w3, animate=True, duration=1, frame_rate=3,
                    backend='plotly', show=False)


def test_export_ceiling_is_a_generous_backstop_not_the_primary_guard():
    # the ceiling only catches pathological slow-drip progress; the real guard
    # is the stall watchdog, so the ceiling is deliberately far above cost
    assert _pb._export_ceiling(1) == _pb._KALEIDO_MIN_CEILING
    assert _pb._export_ceiling(10_000) == 10_000 * _pb._KALEIDO_CEILING_PER_FRAME


def test_kill_process_tree_terminates_a_hung_subprocess_bounded():
    # a genuinely blocked child (sleeps far past any deadline) must be killed
    # and reaped in bounded wall-clock time -- the core recovery primitive
    proc = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(600)'],
        start_new_session=(os.name != 'nt'))
    t0 = time.time()
    _pb._kill_process_tree(proc)
    assert proc.poll() is not None            # actually dead
    assert time.time() - t0 < 20              # ... and bounded


def test_wait_with_progress_kills_a_stalled_render():
    # no new frame ever completes -> wedged -> killed within the stall timeout,
    # regardless of how many frames the export has in total
    proc = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(600)'],
        start_new_session=(os.name != 'nt'))
    t0 = time.time()
    killed = _pb._wait_with_progress(proc, lambda: 0, stall_timeout=3, poll=0.5)
    assert killed is True
    assert proc.poll() is not None
    assert time.time() - t0 < 30


def test_wait_with_progress_does_not_kill_a_slow_but_progressing_render():
    # THE point of a progress watchdog: a render that keeps completing frames
    # is never killed, even though it runs far past the stall timeout. This is
    # what makes the DEFAULT ~900-frame animation safe -- a frame-count-scaled
    # whole-export deadline could not distinguish this from a wedge.
    proc = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(6)'],
        start_new_session=(os.name != 'nt'))
    progress = {'n': 0}

    def count():
        progress['n'] += 1        # a new frame lands on every poll
        return progress['n']

    killed = _pb._wait_with_progress(proc, count, stall_timeout=2, poll=0.5)
    assert killed is False        # ran ~6s > 2s stall timeout, still not killed
    assert proc.returncode == 0   # ... it exited on its own


def test_wait_with_progress_enforces_the_absolute_ceiling():
    # backstop: even with progress on every poll, the ceiling eventually trips
    proc = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(600)'],
        start_new_session=(os.name != 'nt'))
    progress = {'n': 0}

    def count():
        progress['n'] += 1
        return progress['n']

    t0 = time.time()
    killed = _pb._wait_with_progress(proc, count, stall_timeout=300,
                                     ceiling=3, poll=0.5)
    assert killed is True
    assert proc.poll() is not None
    assert time.time() - t0 < 30


def test_render_subprocess_stall_kills_and_raises_bounded(monkeypatch):
    # force an impossibly short stall timeout so even a healthy render is
    # treated as wedged: the parent must kill the process tree, retry, and
    # finally raise a bounded TimeoutError -- never the 20-min hang.
    monkeypatch.setattr(_pb, '_KALEIDO_STALL_TIMEOUT', 1)
    fig = _small_plotly_anim_fig()
    n = len(fig.frames)
    t0 = time.time()
    with pytest.raises(TimeoutError, match='stalled'):
        _pb._render_frames_via_subprocess(fig, 'png', 200, 200, n)
    assert time.time() - t0 < 120


def test_worker_skips_frames_already_rendered(tmp_path):
    # RESUME support: a retry must not redo renders that already landed. Every
    # frame is pre-created, so the worker should skip them all and exit without
    # ever launching Chrome, leaving the sentinel bytes untouched.
    fig = _small_plotly_anim_fig()
    fig_json = tmp_path / 'figure.json'
    fig_json.write_text(fig.to_json())
    frames_dir = tmp_path / 'frames'
    frames_dir.mkdir()
    for i in range(len(fig.frames)):
        (frames_dir / f'{i:06d}.png').write_bytes(b'SENTINEL')
    proc = subprocess.run(
        [sys.executable, '-m', 'hypertools.plot._kaleido_export_worker',
         str(fig_json), str(frames_dir), 'png', '200', '200'],
        capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stderr.decode('utf-8', 'replace')[-2000:]
    for i in range(len(fig.frames)):
        assert (frames_dir / f'{i:06d}.png').read_bytes() == b'SENTINEL'


def test_render_frames_via_subprocess_returns_all_frames():
    # the production render path spawns the worker as a KILLABLE subprocess
    # (deadline + retry) and returns one image blob per frame. This is the ONLY
    # safe way to exercise the worker end-to-end: invoking worker.main() in the
    # test process has no timeout, so a transient Chrome wedge would hang the
    # whole test (it did, on a CI runner) instead of being bounded + retried.
    fig = _small_plotly_anim_fig()
    n = len(fig.frames)
    blobs = _pb._render_frames_via_subprocess(fig, 'png', 200, 200, n)
    assert len(blobs) == n
    assert all(isinstance(b, bytes) and len(b) > 0 for b in blobs)


def test_real_export_succeeds_after_a_killed_export(tmp_path, monkeypatch):
    # THE lifecycle guard: a wedged-and-killed export must leave the PARENT
    # process uncorrupted (no leaked global-server state, no stale closer
    # thread), so a subsequent REAL export in the same process still works --
    # the whole reason for a process boundary over abandoning in-process
    # threads / mutating kaleido's private singleton.
    fig = _small_plotly_anim_fig()
    n = len(fig.frames)
    with monkeypatch.context() as m:            # force a stall+kill first
        m.setattr(_pb, '_KALEIDO_STALL_TIMEOUT', 1)
        with pytest.raises(TimeoutError):
            _pb._render_frames_via_subprocess(fig, 'png', 200, 200, n)
    # ... then a genuine export in the SAME parent process must succeed
    out = str(tmp_path / 'after.gif')
    hyp.plot(walk, animate=True, duration=1, frame_rate=5, backend='plotly',
             save_path=out, show=False)
    assert os.path.getsize(out) > 0
    assert _animated_frames(out) > 1
