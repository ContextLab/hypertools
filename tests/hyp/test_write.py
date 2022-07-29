# noinspection PyPackageRequirements
import datawrangler as dw
import os
import hashlib

import pytest
import hypertools as hyp


data = hyp.load('weights')[:10]

umap2d = {'model': 'UMAP', 'args': [], 'kwargs': {'n_components': 2}}
umap3d = {'model': 'UMAP', 'args': [], 'kwargs': {'n_components': 3}}
post =   [{'model': 'Smooth', 'args': [], 'kwargs': {'kernel_width': 25}},
          {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 100}},
          'ZScore']
hyperalign = {'model': 'HyperAlign', 'args': [], 'kwargs': {'n_iter': 2}}

duration = 10
focused = 2
zoom = 1.0


# check that arbitrary processing pipelines are supported in animations and also that staggered timelines are animated
# correctly
def compare_files(a, b):
    a_hash = hashlib.md5(open(a,'rb').read()).hexdigest()
    b_hash = hashlib.md5(open(b,'rb').read()).hexdigest()
    assert a_hash == b_hash, f'file mismatch: {a} vs. {b}'


def write_test_helper(fig, fname):
    # comment out next line after debugging
    # hyp.write(fig, fname)

    # FIXME: pdfs and gifs need to be manually checked for now...
    if not (fname[-3:].lower() in ['pdf', 'gif']):
        tmp_file = fname.replace('write', 'tmp')
        hyp.write(fig, tmp_file)
        
        try:
            compare_files(fname, tmp_file)
            passed = True
        except AssertionError:
            passed = False
        os.remove(tmp_file)
        assert passed, f'Figure did not match template: {fname}'


def test_write_static_2d(fig_dir):
    fig = hyp.plot(data, reduce=umap2d, post=post, align=hyperalign)

    # pdf
    write_test_helper(fig, os.path.join(fig_dir, 'write2d_static.pdf'))

    # png
    write_test_helper(fig, os.path.join(fig_dir, 'write2d_static.png'))


def test_write_animated_2d(fig_dir):
    fig = hyp.plot(data, reduce=umap2d, post=post, align=hyperalign,
               animate='window', duration=duration, zoom=zoom, focused=focused)

    write_test_helper(fig, os.path.join(fig_dir, 'write2d_animated.gif'))


def test_write_static_3d(fig_dir):
    fig = hyp.plot(data, post=post, reduce=umap3d, align=hyperalign)

    # pdf
    write_test_helper(fig, os.path.join(fig_dir, 'write3d_static.pdf'))

    # png
    write_test_helper(fig, os.path.join(fig_dir, 'write3d_static.png'))


def test_write_animated_3d(fig_dir):
    # FIXME: when data are not 0-centered, the camera and zoom are miscalibrated

    fig = hyp.plot(data, align=hyperalign, reduce=umap3d, post=post,
               animate='chemtrails', duration=duration, zoom=zoom, focused=focused)

    write_test_helper(fig, os.path.join(fig_dir, 'write3d_animated.gif'))
