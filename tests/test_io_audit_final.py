# -*- coding: utf-8 -*-
"""Regression tests for the FINAL 2026-07 release-audit fix wave on the io
unit: file-mode semantics of atomic writes, permission-error wrapping,
gzip-bomb caps, ASCII (protocol-0) pickle sniffing, head-phase stream
salvage, the streaming FFmpeg precheck, and LSL/save documentation
honesty. Real files, real streams, real figures (Agg) -- no mocks."""

import gzip
import os
import pickle
import stat
import threading
import time

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import hypertools as hyp
from hypertools._shared.exceptions import HypertoolsIOError
from hypertools.io import sources

try:
    import pylsl
    PYLSL_AVAILABLE = True
except ImportError:
    PYLSL_AVAILABLE = False

FFMPEG_AVAILABLE = matplotlib.animation.writers.is_available('ffmpeg')


def _current_umask():
    umask = os.umask(0)
    os.umask(umask)
    return umask


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def walk_gen(n=120, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield rng.standard_normal(dim)


def _tiny_animation():
    """A real 3-frame matplotlib FuncAnimation (no hypertools plumbing
    needed to exercise the writer/rename path)."""
    from matplotlib import animation as mpl_animation
    fig, ax = plt.subplots(figsize=(2, 2))
    (ln,) = ax.plot([0, 1], [0, 1])

    def update(i):
        ln.set_ydata([0, 1 + 0.1 * i])
        return (ln,)

    ani = mpl_animation.FuncAnimation(fig, update, frames=3, blit=False)
    return fig, ani


# ------------------------------------------------------------- item 1
# atomic writes via mkstemp must not leak the temp file's private 0600
# mode: new files honor the umask, overwrites preserve the target's mode.

def test_save_new_file_honors_umask(tmp_path):
    target = tmp_path / 'fresh.pkl'
    hyp.save(np.arange(10.0), str(target))
    expected = 0o666 & ~_current_umask()
    assert _mode(target) == expected, (
        f'new file mode {oct(_mode(target))} != {oct(expected)} '
        '(mkstemp 0600 leaked onto the saved file?)')


def test_save_overwrite_preserves_existing_mode(tmp_path):
    target = tmp_path / 'existing.pkl'
    hyp.save(np.arange(5.0), str(target))
    os.chmod(target, 0o604)  # deliberately unusual mode
    hyp.save(np.arange(50.0), str(target))
    assert _mode(target) == 0o604, \
        'overwriting demoted the existing file mode'
    np.testing.assert_allclose(hyp.load(str(target)), np.arange(50.0))


def test_save_csv_overwrite_preserves_existing_mode(tmp_path):
    # the non-pickle formats go through the same mkstemp/replace path
    target = tmp_path / 'existing.csv'
    hyp.save(pd.DataFrame({'a': [1, 2]}), str(target))
    os.chmod(target, 0o604)
    hyp.save(pd.DataFrame({'a': [3, 4]}), str(target))
    assert _mode(target) == 0o604


def test_animation_apng_new_file_honors_umask(tmp_path):
    from hypertools.plot.animate import _save_animation
    out = tmp_path / 'anim.apng'
    fig, ani = _tiny_animation()
    try:
        _save_animation(ani, str(out), 5)
    finally:
        plt.close(fig)
    assert _mode(out) == 0o666 & ~_current_umask()
    with Image.open(out) as im:
        assert im.n_frames >= 2


def test_animation_apng_overwrite_preserves_mode(tmp_path):
    from hypertools.plot.animate import _save_animation
    out = tmp_path / 'anim.apng'
    out.write_bytes(b'placeholder')
    os.chmod(out, 0o604)
    fig, ani = _tiny_animation()
    try:
        _save_animation(ani, str(out), 5)
    finally:
        plt.close(fig)
    assert _mode(out) == 0o604


def test_stream_apng_overwrite_preserves_mode(tmp_path):
    out = tmp_path / 'stream.apng'
    out.write_bytes(b'placeholder')
    os.chmod(out, 0o604)
    hyp.plot(walk_gen(80), stream_init=30, stream_chunk=25,
             save_path=str(out), show=False)
    plt.close('all')
    assert _mode(out) == 0o604
    with Image.open(out) as im:
        assert im.n_frames >= 1


def test_stream_apng_new_file_honors_umask(tmp_path):
    out = tmp_path / 'stream_new.apng'
    hyp.plot(walk_gen(80), stream_init=30, stream_chunk=25,
             save_path=str(out), show=False)
    plt.close('all')
    assert _mode(out) == 0o666 & ~_current_umask()


# ------------------------------------------------------------- item 2
# save() into a write-protected directory must raise the documented
# HypertoolsIOError, not a raw PermissionError from mkstemp.

@pytest.mark.skipif(os.geteuid() == 0,
                    reason='root bypasses directory write protection')
def test_save_write_protected_dir_raises_hypertools_ioerror(tmp_path):
    locked = tmp_path / 'locked'
    locked.mkdir()
    os.chmod(locked, 0o500)
    try:
        with pytest.raises(HypertoolsIOError, match='write permission'):
            hyp.save(np.arange(3.0), str(locked / 'x.pkl'))
    finally:
        os.chmod(locked, 0o700)


# ------------------------------------------------------------- item 3
# transparent gzip decompression must be capped (gzip-bomb DoS).

def test_gzip_cap_constant_is_generous_2gib():
    assert sources._MAX_GZIP_INFLATED_BYTES == 2 * 1024 ** 3


def test_gzip_bomb_rejected(tmp_path, monkeypatch):
    # a REAL gzip payload inflating far past the (temporarily lowered)
    # cap: lowering the constant keeps the test fast/light while the
    # decompression, capping, and error paths all run for real
    monkeypatch.setattr(sources, '_MAX_GZIP_INFLATED_BYTES', 1024 * 1024)
    bomb = tmp_path / 'bomb.csv.gz'
    bomb.write_bytes(gzip.compress(b'0' * (4 * 1024 * 1024)))
    with pytest.raises(HypertoolsIOError, match='gzip bomb'):
        hyp.load(str(bomb))


def test_gzipped_csv_still_loads_transparently(tmp_path):
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    p = tmp_path / 'data.csv.gz'
    p.write_bytes(gzip.compress(df.to_csv(index=False).encode()))
    pd.testing.assert_frame_equal(hyp.load(str(p)), df)


def test_truncated_gzip_raises_corruption_error(tmp_path):
    payload = gzip.compress(b'a,b\n1,2\n3,4\n5,6\n7,8\n')
    bad = tmp_path / 'trunc.csv.gz'
    bad.write_bytes(payload[:len(payload) - 6])
    with pytest.raises(HypertoolsIOError, match='corrupted'):
        hyp.load(str(bad))


# ------------------------------------------------------------- item 4
# extensionless protocol-0 (ASCII) pickles have no magic prefix and used
# to be silently CSV-parsed into a garbage DataFrame.

def test_extensionless_protocol0_pickle_loads_not_csv_garbage(tmp_path):
    obj = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    p = tmp_path / 'proto0_payload'
    p.write_bytes(pickle.dumps(obj, protocol=0))
    # sanity-check the bug's precondition: protocol-0 pickles are plain
    # ASCII, so they decode as UTF-8 and reach the text parser
    p.read_bytes().decode('utf-8')
    assert hyp.load(str(p)) == obj


def test_unknown_extension_protocol0_pickle_roundtrips(tmp_path):
    p = tmp_path / 'payload.xyz'
    hyp.save({'a': [1, 2]}, str(p), protocol=0)
    assert hyp.load(str(p)) == {'a': [1, 2]}


def test_extensionless_csv_still_parses_as_csv(tmp_path):
    # guard against pickle-sniff false positives on genuine text data
    p = tmp_path / 'plain_text_data'
    p.write_text('a,b\n1,2\n3,4\n')
    loaded = hyp.load(str(p))
    assert list(loaded.columns) == ['a', 'b']
    assert loaded.shape == (2, 2)


def test_extensionless_numeric_csv_not_misdetected_as_pickle(tmp_path):
    # '0.5' begins with valid pickle opcodes (POP + STOP), so the sniff
    # must anchor STOP to the payload's end or numeric CSVs would be
    # misdetected as pickles
    p = tmp_path / 'decimal_data'
    p.write_text('0.5,1.5\n2.5,3.5\n')
    loaded = hyp.load(str(p))
    assert loaded.shape[1] == 2
    assert float(loaded.iloc[0, 0]) == 2.5  # first row is the header


def test_extensionless_wordy_text_not_misdetected_as_pickle(tmp_path):
    p = tmp_path / 'wordy_data'
    p.write_text('Name,Age\nNina,33\nIvan.Petrov,41\n')
    loaded = hyp.load(str(p))
    assert list(loaded.columns) == ['Name', 'Age']
    assert loaded.shape == (2, 2)


def test_extensionless_corrupt_ascii_pickle_raises_friendly_error(tmp_path):
    # a complete-looking pickle stream that fails to unpickle must raise
    # a HypertoolsIOError, not come back as CSV garbage
    p = tmp_path / 'broken_proto0'
    p.write_bytes(b'cnot_a_real_module\nnot_a_real_name\n.')
    with pytest.raises(HypertoolsIOError, match='unpickle|pickle'):
        hyp.load(str(p))


# ------------------------------------------------------------- item 5
# a stream error DURING the head phase must salvage the samples already
# received (data, figure, and save file), like the mid-stream salvage.

def test_head_phase_error_salvages_data_figure_and_file(tmp_path):
    out = tmp_path / 'head_partial.gif'

    def dying_in_head():
        rng = np.random.default_rng(1)
        for i in range(200):
            if i == 60:
                raise RuntimeError('amplifier unplugged during head')
            yield rng.standard_normal(3)

    with pytest.warns(RuntimeWarning, match='streaming stopped early'):
        fig = hyp.plot(dying_in_head(), stream_init=100, stream_chunk=25,
                       save_path=str(out), show=False)
    assert fig is not None
    assert fig.stream_info['n_samples'] == 60
    assert fig.stream_info['truncated']
    assert isinstance(fig.stream_info['error'], RuntimeError)
    assert fig.stream_info['data'][0].shape == (60, 3)
    assert out.exists(), 'animation file was lost on a head-phase error'
    with Image.open(out) as im:
        assert im.n_frames >= 1, 'animation was not finalized'
    plt.close('all')


def test_head_phase_error_warning_names_the_head_phase():
    def dying_in_head():
        for i in range(30):
            yield [float(i), float(i % 5), float(i % 3)]
        raise RuntimeError('died before stream_init was reached')

    with pytest.warns(RuntimeWarning, match='stream_init'):
        fig = hyp.plot(dying_in_head(), stream_init=100, stream_chunk=25,
                       show=False)
    assert fig.stream_info['n_samples'] == 30
    assert isinstance(fig.stream_info['error'], RuntimeError)
    plt.close('all')


def test_head_phase_error_with_no_samples_reraises():
    # nothing was consumed, so there is nothing to salvage: the stream's
    # own exception propagates
    def dead():
        raise RuntimeError('never produced a sample')
        yield  # pragma: no cover -- makes this a generator function

    with pytest.raises(RuntimeError, match='never produced'):
        hyp.plot(dead(), stream_init=50, show=False)
    plt.close('all')


def test_head_phase_bad_sample_salvages_good_prefix():
    def bad_row_in_head():
        rng = np.random.default_rng(2)
        for i in range(200):
            if i == 40:
                yield ['not', 'numeric', 'row']
            else:
                yield rng.standard_normal(3)

    with pytest.warns(RuntimeWarning, match='streaming stopped early'):
        fig = hyp.plot(bad_row_in_head(), stream_init=100, stream_chunk=25,
                       show=False)
    assert fig.stream_info['n_samples'] == 40
    assert fig.stream_info['error'] is not None
    assert fig.stream_info['data'][0].shape == (40, 3)
    plt.close('all')


def test_midstream_salvage_still_works():
    # the pre-existing F22-003 behavior must survive the head-phase
    # extension: an error AFTER the head still salvages everything
    def dying_later():
        rng = np.random.default_rng(3)
        for i in range(200):
            if i == 120:
                raise RuntimeError('sensor unplugged')
            yield rng.standard_normal(3)

    with pytest.warns(RuntimeWarning, match='streaming stopped early'):
        fig = hyp.plot(dying_later(), stream_init=50, stream_chunk=25,
                       show=False)
    assert fig.stream_info['n_samples'] == 120
    assert isinstance(fig.stream_info['error'], RuntimeError)
    plt.close('all')


# ------------------------------------------------------------- item 6
# a video save_path must fail fast when FFmpeg is unavailable, BEFORE any
# samples are consumed.

def test_stream_video_save_precheck_no_ffmpeg(tmp_path):
    # point matplotlib at a nonexistent ffmpeg binary (real matplotlib
    # configuration, not a mock) so writer availability is genuinely False
    consumed = {'n': 0}

    def counting():
        for i in range(50):
            consumed['n'] += 1
            yield [float(i), float(i % 7), float(i % 3)]

    old = matplotlib.rcParams['animation.ffmpeg_path']
    matplotlib.rcParams['animation.ffmpeg_path'] = '/nonexistent/ffmpeg'
    try:
        with pytest.raises(RuntimeError, match='FFmpeg'):
            hyp.plot(counting(), stream_init=20, stream_chunk=10,
                     save_path=str(tmp_path / 'out.mp4'), show=False)
    finally:
        matplotlib.rcParams['animation.ffmpeg_path'] = old
    assert consumed['n'] == 0, \
        'samples were consumed before the ffmpeg precheck'
    plt.close('all')


def test_stream_video_precheck_error_names_remedy(tmp_path):
    old = matplotlib.rcParams['animation.ffmpeg_path']
    matplotlib.rcParams['animation.ffmpeg_path'] = '/nonexistent/ffmpeg'
    try:
        with pytest.raises(RuntimeError) as excinfo:
            hyp.plot(walk_gen(30), stream_init=10, stream_chunk=10,
                     save_path=str(tmp_path / 'out.mp4'), show=False)
    finally:
        matplotlib.rcParams['animation.ffmpeg_path'] = old
    message = str(excinfo.value)
    assert 'install ffmpeg' in message.lower()
    assert '.gif' in message  # points at the no-ffmpeg alternative
    plt.close('all')


@pytest.mark.skipif(not FFMPEG_AVAILABLE,
                    reason='ffmpeg is not installed on this machine')
def test_stream_video_save_works_when_ffmpeg_available(tmp_path):
    out = tmp_path / 'stream.mp4'
    fig = hyp.plot(walk_gen(60), stream_init=25, stream_chunk=20,
                   save_path=str(out), show=False)
    assert fig.stream_info['n_samples'] == 60
    assert out.exists() and out.stat().st_size > 0
    plt.close('all')


# ------------------------------------------------------------- item 7
# the LSL multi-match warning is race-dependent by nature; both the
# warning and the docstring must say so honestly.

def test_lsl_docstring_documents_multimatch_best_effort():
    doc = hyp.io.lsl_stream.__doc__
    assert 'best-effort' in doc
    assert 'minimum=2' in doc


@pytest.mark.skipif(not PYLSL_AVAILABLE,
                    reason='pylsl is not installed -- install it with '
                           '`pip install "hypertools[lsl]"`')
def test_lsl_multimatch_warning_admits_best_effort():
    stamp = time.time_ns()
    stream_type = f'HypFinalMulti{stamp}'
    infos = [pylsl.StreamInfo(f'HypFinalA-{stamp}', stream_type, 4, 100.0,
                              'float32', f'hyp-final-a-{stamp}'),
             pylsl.StreamInfo(f'HypFinalB-{stamp}', stream_type, 4, 100.0,
                              'float32', f'hyp-final-b-{stamp}')]
    outlets = [pylsl.StreamOutlet(i) for i in infos]
    try:
        with pytest.warns(RuntimeWarning, match='best-effort'):
            stream = hyp.io.lsl_stream(type=stream_type, timeout=5.0,
                                       minimum=2)
        stream.close()
    finally:
        del outlets


# ------------------------------------------------------------- item 8
# io/lsl.py docstrings must cite the canonical hypertools.core.exceptions
# path (hypertools._shared.exceptions is an internal back-compat shim).

def test_lsl_docstring_cites_core_exceptions_path_only():
    doc = hyp.io.lsl_stream.__doc__
    assert 'hypertools._shared.exceptions' not in doc
    assert doc.count('hypertools.core.exceptions') >= 2


# ------------------------------------------------------------- item 9
# the save docstring must plainly state the 1.0 behavior changes: file
# permission semantics and the **kwargs -> protocol= signature narrowing.

def test_save_docstring_documents_mode_semantics():
    doc = hyp.save.__doc__
    assert 'umask' in doc
    assert '0600' in doc
    assert 'preserve' in doc.lower()


def test_save_docstring_documents_signature_narrowing():
    doc = hyp.save.__doc__
    assert 'versionchanged' in doc
    assert 'TypeError' in doc
    assert '**kwargs' in doc


def test_save_unknown_kwarg_still_raises_typeerror(tmp_path):
    with pytest.raises(TypeError):
        hyp.save(np.arange(3.0), str(tmp_path / 'x.pkl'), compression='xz')
