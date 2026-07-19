# -*- coding: utf-8 -*-
"""Regression tests for the release-1.0 audit's save/return findings
(fix batch B3-animation-save; unit F09-plot-save-return, plus
F10-plot-kwargs-sweep-001).

Every test drives real hypertools plots and real file I/O (no mocks):
MPLBACKEND=Agg, show=False, seeded data.
"""

import os
import pathlib
import warnings

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _walk(n=40, d=4, seed=0):
    return np.cumsum(np.random.RandomState(seed).randn(n, d), axis=0)


# ---------------------------------------------------------------------------
# F09-001: save_path with a user ax= saves THE PLOTTED figure, not whatever
# figure happens to be pyplot-current
# ---------------------------------------------------------------------------

def test_user_ax_save_saves_plotted_figure(tmp_path):
    from PIL import Image
    userfig = plt.figure(figsize=(4, 4))
    userax = userfig.add_subplot(111, projection='3d')
    plt.figure(figsize=(2, 2))  # decoy becomes pyplot-current
    plt.plot([0, 1], [0, 1])
    out = str(tmp_path / 'out.png')
    fig = hyp.plot(_walk(60), ax=userax, save_path=out, show=False)
    assert fig is userfig
    assert Image.open(out).size == (400, 400)  # decoy was 200x200


# ---------------------------------------------------------------------------
# F09-002: .apng saves never clobber a pre-existing sibling .png
# ---------------------------------------------------------------------------

def test_apng_save_preserves_sibling_png(tmp_path):
    from PIL import Image
    sibling = tmp_path / 'movie.png'
    sibling.write_text('PRECIOUS USER DATA')
    target = str(tmp_path / 'movie.apng')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hyp.plot(_walk(30), animate=True, duration=1, frame_rate=5,
                 save_path=target, show=False)
    assert sibling.read_text() == 'PRECIOUS USER DATA'
    assert Image.open(target).n_frames == 5
    # and no stray temp files left behind
    leftovers = [p for p in os.listdir(tmp_path)
                 if p not in ('movie.png', 'movie.apng')]
    assert leftovers == []


# ---------------------------------------------------------------------------
# F09-003: show=False leaks no pyplot figures -- including 12 sequential
# ANIMATED saves -- and the returned HyperAnimation stays fully usable
# ---------------------------------------------------------------------------

def test_twelve_animated_saves_leak_no_figures(tmp_path):
    from PIL import Image
    data = _walk(30)
    plt.close('all')
    keep = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(12):
            keep = hyp.plot(data, animate=True, duration=1, frame_rate=5,
                            save_path=str(tmp_path / f'a{i}.gif'),
                            show=False)
    assert plt.get_fignums() == []
    # the returned handle still exports after its figure was deregistered
    later = str(tmp_path / 'later.gif')
    keep.save(later)
    assert Image.open(later).n_frames == 5
    assert len(keep.to_jshtml()) > 1000
    # redrawing the returned figure must not blow up either
    keep.figure.canvas.draw()


def test_animated_show_false_without_save_leaks_no_figures():
    data = _walk(30)
    plt.close('all')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(4):
            hyp.plot(data, animate=True, duration=1, frame_rate=5,
                     show=False)
    assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# F09-004: pathlib.Path works everywhere str does
# ---------------------------------------------------------------------------

def test_pathlib_path_animated_matplotlib(tmp_path):
    from PIL import Image
    p = pathlib.Path(tmp_path) / 'anim.gif'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hyp.plot(_walk(30), animate=True, duration=1, frame_rate=5,
                 save_path=p, show=False)
    assert Image.open(p).n_frames == 5


def test_pathlib_path_static_and_plotly(tmp_path):
    from PIL import Image
    p = pathlib.Path(tmp_path) / 'static.png'
    hyp.plot(_walk(30), save_path=p, show=False)
    assert Image.open(p).format == 'PNG'
    html = pathlib.Path(tmp_path) / 'fig.html'
    hyp.plot(_walk(30), backend='plotly', save_path=html, show=False)
    assert html.stat().st_size > 1000


def test_hyperanimation_save_accepts_path(tmp_path):
    from PIL import Image
    r = hyp.plot(_walk(30), animate='spin', duration=1, frame_rate=5,
                 show=False)
    p = pathlib.Path(tmp_path) / 'ha.gif'
    r.save(p)
    assert Image.open(p).n_frames == 5


# ---------------------------------------------------------------------------
# F09-005: animated and static exports of the same size= produce the same
# pixel dimensions (no machine-dependent retina 2x)
# ---------------------------------------------------------------------------

def test_animated_and_static_export_same_pixels(tmp_path):
    from PIL import Image
    gif = str(tmp_path / 'a.gif')
    png = str(tmp_path / 's.png')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hyp.plot(_walk(30), animate=True, duration=1, frame_rate=5,
                 save_path=gif, show=False, size=[4, 3])
    hyp.plot(_walk(30), save_path=png, show=False, size=[4, 3])
    assert Image.open(gif).size == Image.open(png).size == (400, 300)


# ---------------------------------------------------------------------------
# F09-006: a failing save leaks no figure with show=False
# ---------------------------------------------------------------------------

def test_failed_save_leaks_no_figures(tmp_path):
    plt.close('all')
    # missing directory now fails FAST (before any figure exists)
    for _ in range(2):
        with pytest.raises(FileNotFoundError, match='save_path directory'):
            hyp.plot(_walk(20), save_path=str(tmp_path / 'ghost/f.png'),
                     show=False)
    assert plt.get_fignums() == []


@pytest.mark.skipif(hasattr(os, 'geteuid') and os.geteuid() == 0,
                    reason='root bypasses file permission bits')
def test_failed_midsave_leaks_no_figures(tmp_path):
    # a MID-save failure (after the figure is drawn, inside fig.savefig)
    # must clean up too. A read-only TARGET file makes savefig's
    # open(..., 'wb') raise PermissionError on every platform:
    # os.chmod(0o444) drops the write bit on POSIX and sets the read-only
    # attribute on Windows (the one chmod semantic Windows honors). The
    # previous fixture -- a chmod-0o500 parent directory -- does not
    # restrict writes on Windows, which inverted the expectation there.
    plt.close('all')
    target = tmp_path / 'locked.png'
    target.write_bytes(b'placeholder')
    os.chmod(target, 0o444)
    try:
        for _ in range(2):
            with pytest.raises(PermissionError):
                hyp.plot(_walk(20), save_path=str(target), show=False)
        assert plt.get_fignums() == []
    finally:
        os.chmod(target, 0o666)


# ---------------------------------------------------------------------------
# F09-007: save_path misuse fails fast with named, actionable errors
# ---------------------------------------------------------------------------

def test_save_path_type_error():
    with pytest.raises(TypeError, match='save_path must be a str'):
        hyp.plot(_walk(20), save_path=123, show=False)


def test_save_path_empty_string():
    with pytest.raises(ValueError, match='save_path is an empty string'):
        hyp.plot(_walk(20), save_path='', show=False)


def test_save_path_existing_directory(tmp_path):
    with pytest.raises(ValueError, match='existing directory'):
        hyp.plot(_walk(20), save_path=str(tmp_path), show=False)


def test_save_path_tilde_expanded(tmp_path, monkeypatch):
    from PIL import Image
    # both variables: POSIX expanduser reads HOME, Windows prefers
    # USERPROFILE -- setting only HOME would silently write into the real
    # Windows profile directory and fail the assert below
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    hyp.plot(_walk(20), save_path='~/fig.png', show=False)
    assert Image.open(tmp_path / 'fig.png').format == 'PNG'


# ---------------------------------------------------------------------------
# F10-plot-kwargs-sweep-001: labels= survive animations whose frame count is
# SMALLER than the sample count (down-sampling used to drop them silently)
# ---------------------------------------------------------------------------

def test_labels_survive_short_animation():
    from hypertools.plot import matplotlib_backend as mb
    A = np.cumsum(np.random.RandomState(50).randn(30, 5), axis=0)
    labels = [None] * 30
    labels[0], labels[15] = 'L0', 'L15'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = hyp.plot(A, animate=True, duration=1.0, frame_rate=5,
                     labels=labels, show=False)
    drawn = [entry[0].get_text() for entry in mb.labels_and_points]
    assert sorted(drawn) == ['L0', 'L15']
    assert r.animation._save_count == 5
