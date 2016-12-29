# # -*- coding: utf-8 -*-

import pytest

## LIBRARIES ##

import numpy as np

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
    assert np.allclose(helpers.interp_array(np.array([1,2,3])),np.array([ 1. ,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9]))
