# # -*- coding: utf-8 -*-

import numpy as np
import hypertools._shared.helpers as helpers


def test_center():
    assert np.array_equal(helpers.center([np.array([[0,0,0],[1,1,1]])]),[np.array([[-0.5,-0.5,-0.5],[0.5,0.5,0.5]])])


def test_group_by_category_ints():
    assert helpers.group_by_category([1, 1, 2, 3])==[0, 0, 1, 2]


def test_group_by_category_str():
    assert helpers.group_by_category(['a', 'a', 'c', 'b'])==[0, 0, 1, 2]


def test_vals2colors_list():
    assert np.allclose(helpers.vals2colors([0, .5, 1]),[(0.9629680891964629, 0.9860207612456747, 0.9360092272202999), (0.7944636678200693, 0.9194156093810073, 0.7700884275278739), (0.4740484429065744, 0.7953863898500577, 0.7713956170703576)])


def test_vals2colors_list_of_lists():
    assert np.allclose(helpers.vals2colors([[0],[.5],[1]]),[(0.9629680891964629, 0.9860207612456747, 0.9360092272202999), (0.7944636678200693, 0.9194156093810073, 0.7700884275278739), (0.4740484429065744, 0.7953863898500577, 0.7713956170703576)])


def test_vals2bins():
    assert helpers.vals2bins([0,1,2])==[0, 33, 66]


def test_interp_array():
    assert np.allclose(helpers.interp_array(np.array([1,2,3])),np.linspace(1,2.9,20))


def test_interp_array_list():
    assert np.allclose(helpers.interp_array_list(np.array([[1,2,3],[1,2,3]])),[np.linspace(1,2.9,20)] * 2)


def test_interp_array_list_interpval():
    assert helpers.interp_array_list([np.array([[1,2,3],[1,2,3],[1,2,3]])],interp_val=10)[0].shape[0]==20

# def test_check_data_list_of_arrays():
#     helpers.check_data([np.random.random((3,3))]*2)=='list'
#
# def test_check_data_list_of_other():
#     with pytest.raises(ValueError) as e_info:
#         helpers.check_data([1,2,3])
#
# def test_check_data_array():
#     helpers.check_data(np.array([[0,1,2],[1,2,3]]))=='array'
#
# def test_check_data_df():
#     helpers.check_data(pd.DataFrame([0,1,2]))=='df'
#
# def test_check_data_df_list():
#     helpers.check_data([pd.DataFrame([0,1,2]),pd.DataFrame([0,1,2])])=='dflist'
#
# def test_check_data_int():
#     with pytest.raises(Exception) as e_info:
#         helpers.check_data(int(1))
#
# def test_check_data_str():
#     with pytest.raises(Exception) as e_info:
#         helpers.check_data(str(1))


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
    assert np.array_equal(helpers.reshape_data(x, labels, labels)[0],[np.array([[1,2],[1,2]]),np.array([[3,4],[3,4]])])
