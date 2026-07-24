# # -*- coding: utf-8 -*-

import numpy as np
import hypertools._shared.helpers as helpers


def test_center():
    assert np.array_equal(helpers.center([np.array([[0,0,0],[1,1,1]])]),[np.array([[-0.5,-0.5,-0.5],[0.5,0.5,0.5]])])


def test_group_by_category_ints():
    assert helpers.group_by_category([1, 1, 2, 3])==[0, 0, 1, 2]


def test_group_by_category_str():
    assert helpers.group_by_category(['a', 'a', 'c', 'b'])==[0, 0, 1, 2]


# vals2colors/vals2bins expected values updated for the release-1.0 audit
# fix (commit 5ddbbf3b): bin edges now span [vmin, vmax] exactly so the
# FULL palette range is used -- the old expected values encoded the stray
# max+1 edge that left the top third of the colormap unused (see the
# comments in hypertools/_shared/helpers.py). min -> first palette slot,
# max -> last palette slot, midpoint -> middle slot.
_GNBU_LO = (0.9629680891964629, 0.9860207612456747, 0.9360092272202999)
_GNBU_MID = (0.4740484429065744, 0.7953863898500577, 0.7713956170703576)
_GNBU_HI = (0.03137254901960784, 0.2608227604767397, 0.5164628988850442)


def test_vals2colors_list():
    assert np.allclose(helpers.vals2colors([0, .5, 1]),
                       [_GNBU_LO, _GNBU_MID, _GNBU_HI])


def test_vals2colors_list_of_lists():
    assert np.allclose(helpers.vals2colors([[0], [.5], [1]]),
                       [_GNBU_LO, _GNBU_MID, _GNBU_HI])


def test_vals2bins():
    assert helpers.vals2bins([0,1,2])==[0, 50, 99]


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


def test_patch_lines_bridges_by_default():
    # without breaks, each group is extended with the first point of the
    # next group so a line renders continuously across the groups
    A = np.array([[0., 0.], [1., 0.]])
    B = np.array([[0., 5.], [1., 5.]])
    out = helpers.patch_lines([A.copy(), B.copy()])
    assert out[0].shape[0] == 3                      # A gained B's first point
    assert np.array_equal(out[0][-1], B[0])          # bridge point == B start


def test_patch_lines_breaks_skip_bridge():
    # a break at group 1 means "do not bridge INTO group 1": the two
    # groups stay disjoint, so no spurious A->B connecting segment (GH #291)
    A = np.array([[0., 0.], [1., 0.]])
    B = np.array([[0., 5.], [1., 5.]])
    out = helpers.patch_lines([A.copy(), B.copy()], breaks={1})
    assert out[0].shape[0] == 2                      # A unchanged
    assert not np.array_equal(out[0][-1], B[0])      # no bridge point


def test_segment_by_run_two_datasets_same_category():
    # GH #291: two datasets sharing ONE category must NOT be merged into a
    # single run -- each dataset is its own run and they are not bridgeable
    A = np.array([[0., 0.], [1., 0.], [2., 0.]])
    B = np.array([[9., 0.], [8., 0.], [7., 0.]])
    segs, seg_labels, seg_cat, seg_bridge, seg_ds = helpers.segment_by_run(
        [A, B], ['x'] * 6)
    assert len(segs) == 2
    assert [s.shape[0] for s in segs] == [3, 3]
    assert seg_cat == ['x', 'x']
    assert seg_bridge == [False]                     # dataset boundary between
    assert seg_ds == [0, 1]                           # one run per dataset


def test_segment_by_run_repeated_category_within_dataset():
    # A A B B A A in ONE dataset -> three runs in source order, each a
    # bridgeable neighbour of the next (same dataset, colour transitions)
    T = np.arange(12, dtype=float).reshape(6, 2)
    segs, seg_labels, seg_cat, seg_bridge, seg_ds = helpers.segment_by_run(
        [T], ['A', 'A', 'B', 'B', 'A', 'A'])
    assert seg_cat == ['A', 'B', 'A']
    assert [s.shape[0] for s in segs] == [2, 2, 2]
    assert seg_bridge == [True, True]                # all within one dataset
    assert seg_ds == [0, 0, 0]                        # all from dataset 0
    # runs preserve original row order (no 1->4 style jumps)
    assert np.array_equal(segs[0], T[0:2])
    assert np.array_equal(segs[1], T[2:4])
    assert np.array_equal(segs[2], T[4:6])


def test_segment_by_run_bridge_flags_across_datasets():
    # runs within a dataset bridge; the boundary between datasets does not
    A = np.arange(8, dtype=float).reshape(4, 2)   # cats a a b b
    B = np.arange(4, dtype=float).reshape(2, 2)   # cat  c c
    segs, seg_labels, seg_cat, seg_bridge, seg_ds = helpers.segment_by_run(
        [A, B], ['a', 'a', 'b', 'b', 'c', 'c'], labels=[0, 1, 2, 3, 4, 5])
    assert seg_cat == ['a', 'b', 'c']
    assert seg_bridge == [True, False]            # a->b same ds; b->c boundary
    assert seg_labels == [[0, 1], [2, 3], [4, 5]]
    assert seg_ds == [0, 0, 1]                    # a,b from ds0; c from ds1
