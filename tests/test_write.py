# noinspection PyPackageRequirements
import datawrangler as dw
import os
import hashlib

import hypertools as hyp


data = hyp.load('weights')
fig_dir = os.path.join(os.path.dirname(__file__), 'reference_figures')

umap2d = {'model': 'UMAP', 'args': [], 'kwargs': {'n_components': 2}}
umap3d = {'model': 'UMAP', 'args': [], 'kwargs': {'n_components': 3}}
manip = [{'model': 'Smooth', 'args': [], 'kwargs': {'kernel_width': 25}},
         {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 1000}},
         'ZScore']
hyperalign = {'model': 'HyperAlign', 'args': [], 'kwargs': {'n_iter': 2}}

duration = 30
focused = 4
zoom = 1.0


# check that arbitrary processing pipelines are supported in animations and also that staggered timelines are animated
# correctly
def compare_files(a, b):
    a_hash = hashlib.md5(open(a,'rb').read()).hexdigest()
    b_hash = hashlib.md5(open(b,'rb').read()).hexdigest()
    assert a_hash == b_hash, f'file mismatch: {a} vs. {b}'


def write_test_helper(fig, fname):
    # comment out next line after debugging
    hyp.write(fig, fname)

    # FIXME: pdfs need to be manually checked for now...
    if not (fname[-3:].lower() == 'pdf'):
        tmp_file = fname.replace('write', 'tmp')
        hyp.write(fig, tmp_file)

        compare_files(fname, tmp_file)
        os.remove(tmp_file)


def test_write_static_2d():
    fig = hyp.plot(data, reduce=umap2d, manip=manip, align=hyperalign)

    # pdf
    write_test_helper(fig, os.path.join(fig_dir, 'write2d_static.pdf'))

    # png
    write_test_helper(fig, os.path.join(fig_dir, 'write2d_static.png'))


def test_write_animated_2d():
    fig = hyp.plot(data, reduce=umap2d, manip=manip, align=hyperalign,
                   animate='window', duration=duration, zoom=zoom, focused=focused)

    write_test_helper(fig, os.path.join(fig_dir, 'write2d_animated.gif'))


def test_write_static_3d():
    fig = hyp.plot(data, manip=manip, reduce=umap3d, align=hyperalign)

    # pdf
    write_test_helper(fig, os.path.join(fig_dir, 'write3d_static.pdf'))

    # png
    write_test_helper(fig, os.path.join(fig_dir, 'write3d_static.png'))


def test_write_animated_3d():
    fig = hyp.plot(data, pipeline=umap3d, manip=manip, align=hyperalign, reduce=umap3d,
                   animate='window', duration=duration, zoom=zoom, focused=focused)

    write_test_helper(fig, os.path.join(fig_dir, 'write3d_animated.gif'))


# test_write_static_2d()
# test_write_animated_2d()
# test_write_static_3d()
test_write_animated_3d()