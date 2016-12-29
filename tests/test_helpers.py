# # -*- coding: utf-8 -*-

import pytest

## LIBRARIES ##

import numpy as np
import pandas as pd

## LOAD TEST DATA ##

# import scipy.io.loadmat as load
# test_dataset = load('examples/test_data.mat')
# test_data = test_dataset['spiral']

## HELPERS ##

import hypertools._shared.helpers as helpers

def test_center():
    assert np.array_equal(helpers.center([np.array([[0,0,0],[1,1,1]])]),[np.array([[-0.5,-0.5,-0.5],[0.5,0.5,0.5]])])

def test_group_by_category_ints():
    assert helpers.group_by_category([1, 1, 2, 3])==[0, 0, 1, 2]

def test_group_by_category_str():
    assert helpers.group_by_category(['a', 'a', 'c', 'b'])==[0, 0, 1, 2]

def test_vals2colors_list():
    assert helpers.vals2colors([0, .5, 1])==[(0.2009432271103454, 0.20707420255623613, 0.20941176489287733), (0.23065488108622481, 0.4299115830776738, 0.50588235901851286), (0.26537486525142889, 0.65373320018543912, 0.79918494084302116)]

def test_vals2colors_list_of_lists():
    assert helpers.vals2colors([[0],[.5],[1]])==[(0.2009432271103454, 0.20707420255623613, 0.20941176489287733), (0.23065488108622481, 0.4299115830776738, 0.50588235901851286), (0.26537486525142889, 0.65373320018543912, 0.79918494084302116)]

def test_vals2bins():
    assert helpers.vals2bins([0,1,2])==[0, 33, 66]

def test_interp_array():
    assert np.allclose(helpers.interp_array(np.array([1,2,3])),np.linspace(1,2.9,20))

def test_interp_array_list():
    assert np.allclose(helpers.interp_array_list(np.array([[1,2,3],[1,2,3]])),[np.linspace(1,2.9,20)] * 2)

def test_check_data_list_of_arrays():
    helpers.check_data([np.random.random((3,3))]*2)=='list'

def test_check_data_list_of_other():
    with pytest.raises(Exception) as e_info:
        helpers.check_data([1,2,3])

def test_check_data_array():
    helpers.check_data(np.array([[0,1,2],[1,2,3]]))=='array'

def test_check_data_df():
    helpers.check_data(pd.DataFrame([0,1,2]))=='df'

def test_check_data_int():
    with pytest.raises(Exception) as e_info:
        helpers.check_data(int(1))

def test_check_data_str():
    with pytest.raises(Exception) as e_info:
        helpers.check_data(str(1))

def test_parse_args_array():
    x = [np.random.random((3,3))]
    args=('o',)
    assert helpers.parse_args(x, args)==[('o',)]

def test_parse_args_list():
    x = [np.random.random((3,3))]*2
    args=('o',)
    assert helpers.parse_args(x, args)==[('o',),('o',)]

def test_parse_kwargs_array():
    x = [np.random.random((3,3))]
    kwargs={'label': ['Group A']}
    assert helpers.parse_kwargs(x, kwargs)==[{'label': 'Group A'}]

def test_parse_kwargs_list():
    x = [np.random.random((3,3))]*2
    kwargs={'label': ['Group A', 'Group B']}
    assert helpers.parse_kwargs(x, kwargs)==[{'label': 'Group A'}, {'label': 'Group B'}]

def test_reshape_data():
    x = [[1,2],[3,4]]*2
    labels = ['a','b','a','b']
    assert np.array_equal(helpers.reshape_data(x,labels),[np.array([[1,2],[1,2]]),np.array([[3,4],[3,4]])])

def test_pandas_to_list_dummy():
    df = pd.DataFrame(['a','b'])
    assert np.array_equal(helpers.pandas_to_list(df),np.array([[1,0],[0,1]]))
