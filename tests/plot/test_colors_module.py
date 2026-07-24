import numpy as np


def test_plot_colors_mat2colors():
    from hypertools.plot.colors import mat2colors
    out = mat2colors(np.array([0.0, 0.5, 1.0]))
    assert np.asarray(out).shape == (3, 3)


def test_plot_colors_reexports_legacy():
    from hypertools.plot.colors import vals2colors, vals2bins, colors2groups
    assert callable(vals2colors) and callable(vals2bins) and callable(colors2groups)
