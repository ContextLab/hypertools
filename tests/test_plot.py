import numpy as np
import pandas as pd

import hypertools as hyp


weights = hyp.load('weights')


def test_static_plot2d():
    # test lines, markers
    # test different strategies for managing color
    # test various manipulations (align, cluster, manip, reduce)
    fig = hyp.plot(weights)
    pass


test_static_plot2d()


def test_animated_plot2d():
    pass


def test_static_plot3d():
    pass


def test_animated_plot3d():
    # test each type of animation
    # also test saving in various formats
    pass


def test_backend_management():
    # not implemented
    pass
