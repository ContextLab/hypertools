def test_matplotlib_backend_module():
    from hypertools.plot.matplotlib_backend import _draw
    assert callable(_draw)


def test_plotly_backend_module():
    from hypertools.plot.plotly_backend import detect_environment, resolve_backend, plotly_draw
    assert callable(detect_environment) and callable(resolve_backend) and callable(plotly_draw)


def test_old_paths_still_work_via_shim():
    from hypertools.plot.draw import _draw as d
    from hypertools.plot.interactive import plotly_draw as p
    from hypertools.plot.matplotlib_backend import _draw as d2
    from hypertools.plot.plotly_backend import plotly_draw as p2
    assert d is d2 and p is p2
