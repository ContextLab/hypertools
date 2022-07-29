# noinspection PyPackageRequirements
import datawrangler as dw
import os
import hashlib
from PIL import Image
import numpy as np
import random
from pdf2image import convert_from_path

import pytest
import hypertools as hyp


data = hyp.load('weights')[:10]

umap2d = {'model': 'UMAP', 'args': [], 'kwargs': {'n_components': 2,
          'random_state': 10}}
umap3d = {'model': 'UMAP', 'args': [], 'kwargs': {'n_components': 3,
          'random_state': 10}}
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
    def compare_images(x, y, tol=0.95):
        if type(x) is list:
            return all([compare_images(i, j) for i, j in zip(x, y)])
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        return np.corrcoef(x_array.ravel(), y_array.ravel())[0, 1] >= tol
    
    def gif2imlist(g):
        x = Image.open(g)
        frames = []
        try:
            while True:
                x.seek(x.tell() + 1)
                frames.append(x)
        except EOFError:
            return frames

    a_hash = hashlib.md5(open(a,'rb').read()).hexdigest()
    b_hash = hashlib.md5(open(b,'rb').read()).hexdigest()

    if a_hash == b_hash:
        return True
    else:
        if a[-3:].lower() == 'png':
            a_img = Image.open(a)
            b_img = Image.open(b)
        elif a[-3:].lower() == 'pdf':
            a_img = convert_from_path(a)
            b_img = convert_from_path(b)
        elif a[-3:].lower() == 'gif':
            a_img = gif2imlist(a)
            b_img = gif2imlist(b)
        return compare_images(a_img, b_img)

def plot_wrapper(*args, **kwargs):
        np.random.seed(1234)
        return hyp.plot(*args, **kwargs)


def write_test_helper(fig, fname):
    # comment out next line after debugging
    # hyp.write(fig, fname)
    
    tmp_file = fname.replace('write', 'tmp')
    hyp.write(fig, tmp_file)
    match = compare_files(fname, tmp_file)
    os.remove(tmp_file)
    assert match, f'File did not match template: {fname}'


def test_write_static_2d(fig_dir):
    fig = plot_wrapper(data, reduce=umap2d, post=post, align=hyperalign)

    # pdf
    write_test_helper(fig, os.path.join(fig_dir, 'write2d_static.pdf'))

    # png
    write_test_helper(fig, os.path.join(fig_dir, 'write2d_static.png'))


def test_write_animated_2d(fig_dir):
    fig = plot_wrapper(data, reduce=umap2d, post=post, align=hyperalign,
               animate='window', duration=duration, zoom=zoom, focused=focused)

    write_test_helper(fig, os.path.join(fig_dir, 'write2d_animated.gif'))


def test_write_static_3d(fig_dir):
    fig = plot_wrapper(data, post=post, reduce=umap3d, align=hyperalign)

    # pdf
    write_test_helper(fig, os.path.join(fig_dir, 'write3d_static.pdf'))

    # png
    write_test_helper(fig, os.path.join(fig_dir, 'write3d_static.png'))


def test_write_animated_3d(fig_dir):
    # FIXME: when data are not 0-centered, the camera and zoom are miscalibrated
    fig = plot_wrapper(data, align=hyperalign, reduce=umap3d, post=post,
               animate='chemtrails', duration=duration, zoom=zoom, focused=focused)

    write_test_helper(fig, os.path.join(fig_dir, 'write3d_animated.gif'))
