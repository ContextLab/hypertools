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

def test__getAplus():
    assert np.allclose(helpers._getAplus(np.array([[1,2,3],[4,5,6],[7,8,9]])),np.matrix([[0.86725382,1.96398776,3.0607217 ], [1.96398776,   4.44765748,   6.93132719],[  3.0607217 ,   6.93132719,  10.80193268]]))

def test__getPs():
    mtx = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
    n = mtx.shape[0]
    W = np.identity(n)
    assert np.allclose(helpers._getPs(mtx, W=W),np.matrix([[0.86725382,1.96398776,3.0607217 ], [1.96398776,   4.44765748,   6.93132719],[  3.0607217 ,   6.93132719,  10.80193268]]))

def test__getPu():
    mtx = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
    n = mtx.shape[0]
    W = np.identity(n)
    assert np.allclose(helpers._getPs(mtx,W=W), np.matrix([[  0.86725382,   1.96398776,   3.0607217 ],[  1.96398776,   4.44765748,   6.93132719],[  3.0607217 ,   6.93132719,  10.80193268]]))

def test_nearPD():
    mtx = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
    assert np.allclose(helpers.nearPD(mtx), np.matrix([[ 1.        ,  1.09304495,  1.12692214],[ 1.09304495,1.        ,  1.14165415],[ 1.12692214,  1.14165415,  1.        ]]))

def test_is_pos_def_true():
    mtx = np.matrix([[1,0],[0,1]])
    assert helpers.is_pos_def(mtx)==True

def test_is_pos_def_false():
    mtx = np.matrix([[-3,2,0],[2,-3,0],[0,0,-5]])
    assert helpers.is_pos_def(mtx)==False
