# -*- coding: utf-8 -*-
"""GH #94 (feature, glue): `hyp.plot(..., resample=N)` runs the existing
`hypertools.manip` ``Resample`` manipulator (PCHIP resampling) on each
dataset BEFORE the normalize/reduce/align pipeline, so the resampled row
count is what everything downstream (normalize/reduce/align/cluster/hue,
and later the GH #141 line-smoothing interpolation) sees.

Real `hyp.plot()`/`hyp.manip()` calls only (no mocks): every assertion
exercises the actual PCHIP resampling + rendering pipeline.
"""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp


def test_resample_produces_n_row_line_artist():
    """resample=500 on a 100-row dataset -> the plotted (marker-only, so
    NOT further densified by GH #141 line-smoothing interpolation) artist
    carries exactly 500 vertices."""
    rng = np.random.default_rng(0)
    data = np.cumsum(rng.standard_normal((100, 3)), axis=0)

    fig = hyp.plot([data.copy()], 'o', animate=False, show=False,
                   resample=500)
    lines = fig.axes[0].get_lines()
    plt.close('all')

    assert len(lines) == 1
    xdata, ydata, zdata = lines[0].get_data_3d()
    assert xdata.shape[0] == 500


def test_resample_values_match_hyp_manip_exactly():
    """The resampled (pre-reduce) values fed into the pipeline must match
    `hyp.manip(data, model='Resample', n_samples=500)` on the SAME input
    EXACTLY -- verified with normalize=None and the default
    reduce='IncrementalPCA' (a no-op on already-3D data), so `xform_data`
    (return_model=True) is untouched apart from the resample step itself."""
    rng = np.random.default_rng(1)
    data = np.cumsum(rng.standard_normal((100, 3)), axis=0)

    expected = np.asarray(
        hyp.manip(data, model='Resample', n_samples=500))
    assert expected.shape == (500, 3)

    result = hyp.plot([data.copy()], 'o', animate=False, show=False,
                      return_model=True, resample=500, normalize=None)
    xform_data = result['xform_data'][0]
    plt.close('all')

    assert xform_data.shape == (500, 3)
    assert np.allclose(xform_data, expected)


def test_resample_smaller_than_original_downsamples():
    rng = np.random.default_rng(2)
    data = np.cumsum(rng.standard_normal((200, 2)), axis=0)

    expected = np.asarray(hyp.manip(data, model='Resample', n_samples=50))
    result = hyp.plot([data.copy()], 'o', animate=False, show=False,
                      return_model=True, resample=50, normalize=None,
                      ndims=2)
    xform_data = result['xform_data'][0]
    plt.close('all')

    assert xform_data.shape == (50, 2)
    assert np.allclose(xform_data, expected)


def test_resample_none_default_unchanged():
    """resample=None (the default) must not alter row count at all."""
    rng = np.random.default_rng(3)
    data = np.cumsum(rng.standard_normal((37, 3)), axis=0)

    result = hyp.plot([data.copy()], 'o', animate=False, show=False,
                      return_model=True)
    assert result['xform_data'][0].shape[0] == 37
    plt.close('all')


def test_resample_false_equivalent_to_none():
    rng = np.random.default_rng(4)
    data = np.cumsum(rng.standard_normal((25, 3)), axis=0)

    result = hyp.plot([data.copy()], 'o', animate=False, show=False,
                      return_model=True, resample=False)
    assert result['xform_data'][0].shape[0] == 25
    plt.close('all')


def test_resample_combines_with_line_smoothing_interpolation():
    """resample= (row-count change, applied before analyze) and the GH
    #141 line-smoothing interpolation (applied to whatever row count
    survives analyze/reduce, for line-style fmts) compose: a line-style
    plot with resample=200 should show a SMOOTHED, interpolated line
    (more than 200 vertices, since interp_val > 1 for 200 points at the
    default frame_rate/duration), not exactly 200 raw vertices."""
    rng = np.random.default_rng(5)
    data = np.cumsum(rng.standard_normal((50, 3)), axis=0)

    fig = hyp.plot([data.copy()], '-', animate=False, show=False,
                   resample=200)
    lines = fig.axes[0].get_lines()
    plt.close('all')

    assert len(lines) == 1
    xdata = lines[0].get_data_3d()[0]
    assert xdata.shape[0] > 200


@pytest.mark.parametrize('bad', [1, 0, -5, 1.5, '500', [500], True])
def test_resample_bad_values_raise_value_error(bad):
    rng = np.random.default_rng(6)
    data = np.cumsum(rng.standard_normal((20, 3)), axis=0)

    with pytest.raises(ValueError):
        hyp.plot([data.copy()], 'o', animate=False, show=False,
                 resample=bad)
    plt.close('all')


def test_resample_multiple_datasets_each_independently_resampled():
    rng = np.random.default_rng(7)
    data = [np.cumsum(rng.standard_normal((30, 3)), axis=0),
           np.cumsum(rng.standard_normal((80, 3)), axis=0)]

    result = hyp.plot([d.copy() for d in data], 'o', animate=False,
                      show=False, return_model=True, resample=60,
                      normalize=None)
    for xi in result['xform_data']:
        assert xi.shape[0] == 60
    plt.close('all')
