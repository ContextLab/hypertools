# -*- coding: utf-8 -*-
"""Animation export tests: real files, verified frame counts."""

import importlib
import os
import shutil
import subprocess
import sys
import time

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


@pytest.mark.skipif(not HAS_FFMPEG, reason='ffmpeg not installed')
def test_mp4_export_is_quality_targeted_not_fixed_bitrate(tmp_path):
    """Until 1.1 every video was written at ``bitrate=1800`` kbit/s, so a
    2-second clip was ~450 KB whatever it showed. A CRF encode of a small,
    mostly-white line plot spends a small fraction of that. The bound is
    generous (a quarter of the old fixed budget) so codec build differences
    cannot flake it, and still impossible for a fixed 1800 kbit/s stream."""
    out = str(tmp_path / 'anim.mp4')
    hyp.plot(walk, animate=True, duration=2, frame_rate=10, size=(4, 4),
             save_path=out, show=False)
    plt.close('all')
    size = os.path.getsize(out)
    assert 1000 < size < 1800 * 1000 / 8 * 2 / 4, (
        f'{size} bytes for a 2 s clip -- a fixed 1800 kbit/s stream is '
        f'~450 KB; a CRF encode of this plot is far smaller')


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
    reason = _pb._wait_with_progress(proc, lambda: 0, stall_timeout=3, poll=0.5)
    assert reason == 'stalled'
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

    reason = _pb._wait_with_progress(proc, count, stall_timeout=2, poll=0.5)
    assert reason == 'exited'     # ran ~6s > 2s stall timeout, still not killed
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
    reason = _pb._wait_with_progress(proc, count, stall_timeout=300,
                                     ceiling=3, poll=0.5)
    # reported as a CEILING hit, not a stall: frames were arriving the whole
    # time, so this needs different diagnosis than a wedged browser
    assert reason == 'ceiling'
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


def test_worker_starts_no_browser_when_every_frame_already_exists(tmp_path):
    # RESUME, complete case: a previous attempt may have rendered EVERY frame
    # and then wedged during browser TEARDOWN. Starting Chrome again just to
    # discover there is nothing to draw would let a broken Kaleido fail an
    # export whose frames are all present -- so the worker must return before
    # touching a browser at all. A stub `kaleido` on PYTHONPATH records any
    # attempt to start one (and removes the need for a real browser here).
    fig = _small_plotly_anim_fig()
    fig_json = tmp_path / 'figure.json'
    fig_json.write_text(fig.to_json())
    frames_dir = tmp_path / 'frames'
    frames_dir.mkdir()
    for i in range(len(fig.frames)):
        (frames_dir / f'{i:06d}.png').write_bytes(b'SENTINEL')

    stub_dir = tmp_path / 'stub'
    stub_dir.mkdir()
    marker = tmp_path / 'browser-was-started'
    (stub_dir / 'kaleido.py').write_text(
        'import os\n'
        'class _S:\n'
        '    def is_running(self):\n'
        '        return False\n'
        '_global_server = _S()\n'
        'def start_sync_server(*a, **k):\n'
        f'    open(r"{marker}", "w").close()\n'
        'def stop_sync_server(*a, **k):\n'
        '    pass\n')
    env = dict(os.environ, PYTHONPATH=str(stub_dir))
    proc = subprocess.run(
        [sys.executable, '-m', 'hypertools.plot._kaleido_export_worker',
         str(fig_json), str(frames_dir), 'png', '200', '200'],
        capture_output=True, timeout=180, env=env)
    assert proc.returncode == 0, proc.stderr.decode('utf-8', 'replace')[-2000:]
    assert not marker.exists(), 'worker started a browser on a complete resume'
    for i in range(len(fig.frames)):
        assert (frames_dir / f'{i:06d}.png').read_bytes() == b'SENTINEL'


def test_export_accepts_a_complete_frame_set_from_a_killed_worker(monkeypatch):
    # Parent side of the same hazard: if the worker is killed (here: an
    # impossibly short stall timeout) but every frame already landed, the export
    # must ACCEPT that complete set rather than reporting failure. Frames are
    # pre-seeded into the workdir the parent creates, via a patched mkdtemp.
    import tempfile as _tempfile
    fig = _small_plotly_anim_fig()
    n = len(fig.frames)
    real_mkdtemp = _tempfile.mkdtemp

    def seeded_mkdtemp(*a, **k):
        d = real_mkdtemp(*a, **k)
        frames_dir = os.path.join(d, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        for i in range(n):                     # export is already complete
            with open(os.path.join(frames_dir, f'{i:06d}.png'), 'wb') as fh:
                fh.write(b'SEEDED')
        return d

    monkeypatch.setattr(_tempfile, 'mkdtemp', seeded_mkdtemp)
    monkeypatch.setattr(_pb, '_KALEIDO_STALL_TIMEOUT', 1)   # kill the worker
    blobs = _pb._render_frames_via_subprocess(fig, 'png', 200, 200, n)
    assert blobs == [b'SEEDED'] * n


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


# ------------------------------------------- animation Play/Pause controls
# Maintainer report (Andy): in 2-D the controls were drawn ON TOP of the
# chart's bottom-left corner, and their styling was rough ("Play is a little
# off center inside the button"). They used to sit at paper (0, 0) anchored
# bottom-left -- fine in 3-D, where the scene floats above that corner, but in
# 2-D the axes fill the paper area. They now hang BELOW the plotting area.

@pytest.mark.parametrize('ndims', [2, 3])
def test_animation_controls_sit_below_the_plot_area(ndims):
    fig = hyp.plot(walk, animate=True, ndims=ndims, backend='plotly',
                   show=False)
    menus = fig.layout.updatemenus
    assert len(menus) == 1
    menu = menus[0]
    # anchored by its TOP edge at a NEGATIVE paper y => entirely below the
    # plotting area (paper y=0 is its bottom edge), so it cannot overlap
    assert menu.yanchor == 'top'
    assert menu.y < 0, 'controls must hang below the plot, not overlap it'
    # ... and the bottom margin is opened up so they are not clipped off
    assert fig.layout.margin.b >= _pb._ANIM_BUTTON_MARGIN_B


@pytest.mark.parametrize('ndims', [2, 3])
def test_animation_controls_are_themed(ndims):
    fig = hyp.plot(walk, animate=True, ndims=ndims, backend='plotly',
                   show=False)
    menu = fig.layout.updatemenus[0]
    assert menu.direction == 'right', 'controls should lay out horizontally'
    # symmetric padding centers each label in its button
    assert menu.pad.l == menu.pad.r and menu.pad.t == menu.pad.b
    assert menu.pad.l > 0 and menu.pad.t > 0
    assert menu.bgcolor and menu.bordercolor and menu.borderwidth >= 1
    # same font stack as the rest of the figure, not plotly's default face
    assert menu.font.family == _pb._PLOTLY_SANS_STACK
    assert [b.label for b in menu.buttons] == ['Play', 'Pause']


def test_animation_controls_preserve_other_margins():
    # update_layout merges nested dicts -- opening up margin.b must not clobber
    # the left/right/top margins the static layout computed
    static = hyp.plot(walk, ndims=2, backend='plotly', show=False)
    anim = hyp.plot(walk, animate=True, ndims=2, backend='plotly', show=False)
    for side in ('l', 'r', 't'):
        assert getattr(anim.layout.margin, side) == \
            getattr(static.layout.margin, side)


# --- HyperAnimation.save() forwards dpi= and refuses what it cannot honour ---
# Found 2026-09-03: `anim.save('x.gif', dpi=75)` wrote the GIF at the figure's
# dpi because save() popped `fps` and silently discarded every other keyword.
# The launch notebooks had been passing dpi= for a month with no effect.

def _gif_size(path):
    with Image.open(path) as im:
        return im.size


def test_hyper_animation_save_honours_dpi(tmp_path):
    anim = hyp.plot(walk, animate=True, duration=0.5, frame_rate=4,
                    size=(4, 3), show=False)
    anim.save(tmp_path / 'lo.gif', dpi=50)
    anim.save(tmp_path / 'hi.gif', dpi=100)
    lo, hi = _gif_size(tmp_path / 'lo.gif'), _gif_size(tmp_path / 'hi.gif')
    assert lo == (200, 150) and hi == (400, 300), (lo, hi)
    plt.close('all')


def test_hyper_animation_save_default_dpi_is_the_figures(tmp_path):
    anim = hyp.plot(walk, animate=True, duration=0.5, frame_rate=4,
                    size=(4, 3), show=False)
    anim.save(tmp_path / 'default.gif')
    expected = (round(4 * anim.figure.dpi), round(3 * anim.figure.dpi))
    assert _gif_size(tmp_path / 'default.gif') == expected
    plt.close('all')


def test_hyper_animation_save_refuses_unknown_keywords(tmp_path):
    anim = hyp.plot(walk, animate=True, duration=0.5, frame_rate=4,
                    show=False)
    with pytest.raises(TypeError, match=r"unexpected keyword.*'bitrate'"):
        anim.save(tmp_path / 'x.gif', bitrate=1800)
    assert not (tmp_path / 'x.gif').exists()
    plt.close('all')
