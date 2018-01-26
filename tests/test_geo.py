# -*- coding: utf-8 -*-

import pytest
import numpy as np

from hypertools.datageometry import DataGeometry
from hypertools.plot.plot import plot

data = [np.random.multivariate_normal(np.zeros(4), np.eye(4), size=100) for i
        in range(2)]

geo = plot(data, show=False)

def test_geo():
    assert isinstance(geo, DataGeometry)

def test_geo_data():
    assert isinstance(geo.data, list)

def test_geo_data_dims():
    assert (geo.data[0].shape[0]==100) and (geo.data[0].shape[1]==4)

def test_geo_kwargs():
    assert isinstance(geo.kwargs, dict)

def test_geo_reduce():
    assert isinstance(geo.reduce, dict)

def test_geo_xform_data_dims1():
    assert (geo.xform_data[0].shape[0]==100) and (geo.xform_data[0].shape[1]==3)

def test_geo_xform_data_dims2():
    geo = plot(data, ndims=4, show=False)
    assert (geo.xform_data[0].shape[0]==100) and (geo.xform_data[0].shape[1]==4)

def test_geo_transform():
    assert isinstance(geo.transform(data), list)

def test_geo_transform_dims():
    assert geo.transform(data)[0].shape[1]==3

def test_geo_plot():
    assert isinstance(geo.plot(show=False), DataGeometry)

def test_geo_text_data():
    data = [['i like cats alot', 'cats r pretty cool', 'cats are better than dogs'],
            ['dogs rule the haus', 'dogs are my jam', 'dogs are a mans best friend']]
    geo = plot(data, show=False)
    assert isinstance(geo, DataGeometry)
    assert geo.transform(data)[0].shape[1]==3
    assert geo.text['model'] is 'LatentDirichletAllocation'
    assert geo.text['params']=={'n_components' : 20}
    assert isinstance(geo.plot(show=False), DataGeometry)
