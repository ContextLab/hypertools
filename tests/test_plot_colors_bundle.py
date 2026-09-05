"""`bundle['colors']` and `HyperAnimation.colors` (GH #285).

`examples/animate_weather_decades.py` rebuilds `Normalize(mean.min(),
mean.max())` + `plt.get_cmap('RdBu_r')` by hand to reproduce hypertools'
own hue mapping in its companion panels -- a transcription that can silently
drift from what the figure actually drew. `plot()` already computes exactly
that mapping for `colorbar=`; the bundle now hands it back, always, whether
or not a colorbar was asked for.

Every assertion compares the exposed mapping against the colours actually
drawn on the figure, so "it matches" is measured, not asserted by fiat.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import hypertools as hyp


def _datasets(n=3, rows=25, cols=5, seed=6):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.normal(size=(rows, cols)), axis=0) for _ in range(n)]


# --- presence / shape ---------------------------------------------------

def test_bundle_always_carries_a_colors_key():
    bundle = hyp.plot(_datasets(3), reduce='PCA', return_model=True,
                      show=False)
    assert 'colors' in bundle
    colors = bundle['colors']
    assert set(colors) >= {'kind', 'palette', 'cmap', 'norm', 'vmin',
                           'vmax', 'colors', 'labels', 'categories'}


def test_colors_is_present_without_asking_for_a_colorbar():
    """The colorbar is opt-in; the resolved mapping is not."""
    bundle = hyp.plot(_datasets(3), reduce='PCA', return_model=True,
                      show=False)
    assert bundle['colors']['kind'] == 'discrete'
    assert np.asarray(bundle['colors']['colors']).shape == (3, 3)


def test_single_ungrouped_dataset_does_not_raise():
    """`colorbar=True` on one ungrouped dataset raises by design; the
    colours dict must still resolve."""
    with pytest.raises(ValueError):
        hyp.plot(_datasets(1), colorbar=True, reduce='PCA', show=False)
    bundle = hyp.plot(_datasets(1), reduce='PCA', return_model=True,
                      show=False)
    assert bundle['colors']['kind'] == 'discrete'
    assert np.asarray(bundle['colors']['colors']).shape == (1, 3)


# --- continuous hue -----------------------------------------------------

def test_continuous_hue_exposes_the_real_value_range_and_cmap():
    data = _datasets(2, rows=25)
    hue = np.linspace(-3.5, 8.25, 50)
    bundle = hyp.plot(data, hue=hue, palette='RdBu_r', reduce='PCA',
                      return_model=True, show=False)
    colors = bundle['colors']
    assert colors['kind'] == 'continuous'
    assert colors['vmin'] == pytest.approx(-3.5)
    assert colors['vmax'] == pytest.approx(8.25)
    assert colors['norm'](hue[0]) == pytest.approx(0.0)
    assert colors['norm'](hue[-1]) == pytest.approx(1.0)
    assert colors['palette'] == 'RdBu_r'


def test_continuous_mapping_reproduces_the_drawn_line_colours():
    """The point of the feature: `cmap(norm(value))` must be the colour
    hypertools actually drew for that observation."""
    from matplotlib.collections import LineCollection
    data = _datasets(1, rows=30)
    hue = np.linspace(0.0, 1.0, 30)
    bundle = hyp.plot(data, hue=hue, palette='RdBu_r', reduce='PCA',
                      return_model=True, show=False)
    colors = bundle['colors']
    drawn = [c for c in bundle['fig'].axes[0].collections
             if isinstance(c, LineCollection)
             and getattr(c, '_hyp_trace_index', None) is not None]
    assert drawn, 'no multicolour line collection was drawn'
    segment_colors = np.asarray(drawn[0].get_colors())
    predicted = np.asarray(colors['cmap'](colors['norm'](hue)))
    # the first drawn segment starts at the first observation's colour
    np.testing.assert_allclose(segment_colors[0][:3], predicted[0][:3],
                               atol=0.02)
    np.testing.assert_allclose(segment_colors[-1][:3], predicted[-1][:3],
                               atol=0.02)


# --- categorical hue ----------------------------------------------------

def test_categorical_hue_exposes_a_category_to_colour_map():
    data = _datasets(3, rows=20)
    bundle = hyp.plot(data, hue=['a', 'b', 'c'], reduce='PCA',
                      return_model=True, show=False)
    colors = bundle['colors']
    assert colors['kind'] == 'discrete'
    assert list(colors['categories']) == ['a', 'b', 'c']
    drawn = [line.get_color() for line in bundle['fig'].axes[0].lines]
    for category, drawn_color in zip(['a', 'b', 'c'], drawn):
        np.testing.assert_allclose(colors['categories'][category],
                                   matplotlib.colors.to_rgb(drawn_color),
                                   atol=1e-6)


def test_discrete_norm_indexes_the_groups():
    data = _datasets(3, rows=20)
    colors = hyp.plot(data, hue=['a', 'b', 'c'], reduce='PCA',
                      return_model=True, show=False)['colors']
    assert [int(colors['norm'](i)) for i in range(3)] == [0, 1, 2]
    np.testing.assert_allclose(colors['cmap'](colors['norm'](1))[:3],
                               colors['colors'][1], atol=1e-6)


def test_cluster_labels_reach_the_categories_map():
    data = _datasets(3, rows=20)
    colors = hyp.plot(data, cluster='KMeans', n_clusters=2, reduce='PCA',
                      return_model=True, show=False)['colors']
    assert colors['kind'] == 'discrete'
    assert len(colors['colors']) >= 2


def test_colors_agree_with_the_colorbar_when_both_are_asked_for():
    data = _datasets(2, rows=25)
    hue = np.linspace(0, 10, 50)
    bundle = hyp.plot(data, hue=hue, colorbar=True, palette='viridis',
                      reduce='PCA', return_model=True, show=False)
    colors = bundle['colors']
    cbar_axes = [a for a in bundle['fig'].axes if a is not bundle['fig'].axes[0]]
    assert cbar_axes, 'no colorbar was drawn'
    assert (colors['vmin'], colors['vmax']) == (0.0, 10.0)


# --- animations ---------------------------------------------------------

def test_hyper_animation_exposes_the_same_dict():
    data = _datasets(2, rows=20)
    anim = hyp.plot(data, hue=np.linspace(0, 1, 40), palette='RdBu_r',
                    animate=True, reduce='PCA', show=False, duration=1,
                    frame_rate=2)
    assert anim.colors['kind'] == 'continuous'
    assert anim.colors['vmin'] == pytest.approx(0.0)
    assert anim.colors['vmax'] == pytest.approx(1.0)


def test_animated_bundle_and_wrapper_agree():
    data = _datasets(2, rows=20)
    hue = np.linspace(2, 9, 40)
    bundle = hyp.plot(data, hue=hue, animate=True, reduce='PCA',
                      return_model=True, show=False, duration=1,
                      frame_rate=2)
    anim = hyp.plot(data, hue=hue, animate=True, reduce='PCA', show=False,
                    duration=1, frame_rate=2)
    assert bundle['colors']['vmin'] == anim.colors['vmin']
    assert bundle['colors']['vmax'] == anim.colors['vmax']
    assert bundle['colors']['kind'] == anim.colors['kind']


# --- matrix hue ---------------------------------------------------------

def test_matrix_hue_reports_a_blend():
    rng = np.random.default_rng(7)
    weights = rng.random((40, 3))
    weights /= weights.sum(axis=1, keepdims=True)
    bundle = hyp.plot(_datasets(2, rows=20), hue=weights, reduce='PCA',
                      return_model=True, show=False)
    assert bundle['colors']['kind'] == 'blend'
    assert bundle['colors']['cmap'] is None


def test_matrix_hue_blend_carries_the_legend_swatches():
    rng = np.random.default_rng(8)
    weights = rng.random((40, 3))
    weights /= weights.sum(axis=1, keepdims=True)
    bundle = hyp.plot(_datasets(2, rows=20), hue=weights,
                      legend=['a', 'b', 'c'], reduce='PCA',
                      return_model=True, show=False)
    assert list(bundle['colors']['categories']) == ['a', 'b', 'c']


# --- plotly parity ------------------------------------------------------

def test_colors_bundle_under_plotly():
    pytest.importorskip('plotly')
    data = _datasets(2, rows=20)
    bundle = hyp.plot(data, hue=np.linspace(0, 4, 40), palette='RdBu_r',
                      reduce='PCA', backend='plotly', return_model=True,
                      show=False)
    colors = bundle['colors']
    assert colors['kind'] == 'continuous'
    assert (colors['vmin'], colors['vmax']) == (0.0, 4.0)
