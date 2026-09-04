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


def test_patch_lines_bridges_labels_in_lockstep_with_data():
    # regression test: patch_lines grows x[idx] by one row (a duplicate of
    # x[idx+1]'s first point) whenever it bridges, but historically left a
    # parallel `labels` list untouched -- permanently under-counting it by
    # one entry per bridge point relative to the (now longer) data array.
    # This crashed `annotate_plot` (matplotlib_backend.py) with a bare
    # IndexError whenever nothing downstream happened to rebuild `labels`
    # from scratch afterward (animate='morph', or a static plot with
    # antialias=False) -- and silently misplaced real point labels even
    # when something DID (the `_expand_labels` interpolation remap), since
    # it sliced each segment's labels using the BRIDGED (wrong) length.
    # `labels=`, when given, must grow in lockstep with `x`: the synthetic
    # bridge row is a DUPLICATE observation (not a new one), so it gets a
    # `None` placeholder, exactly like `_expand_labels` marks synthetic
    # interpolated points.
    A = np.array([[0., 0.], [1., 0.]])
    B = np.array([[0., 5.], [1., 5.]])
    labels = [["a0", "a1"], ["b0", "b1"]]
    out = helpers.patch_lines([A.copy(), B.copy()], labels=labels)
    assert out[0].shape[0] == 3                      # A gained B's first point
    assert len(labels[0]) == 3                        # labels[0] grew to match
    assert labels[0] == ["a0", "a1", None]            # bridge slot is None
    assert labels[1] == ["b0", "b1"]                  # untouched (last group)


def test_patch_lines_breaks_skip_bridge_labels_too():
    # when a break skips bridging the DATA, the parallel `labels` list must
    # not be extended either -- they must always stay the same length.
    A = np.array([[0., 0.], [1., 0.]])
    B = np.array([[0., 5.], [1., 5.]])
    labels = [["a0", "a1"], ["b0", "b1"]]
    out = helpers.patch_lines([A.copy(), B.copy()], breaks={1}, labels=labels)
    assert out[0].shape[0] == 2                       # A unchanged
    assert labels[0] == ["a0", "a1"]                  # labels unchanged too


def test_segment_by_run_then_patch_lines_keeps_labels_length_matched():
    # the exact call sequence `_regroup_categorical_lines` (plot.py) uses:
    # segment_by_run() followed by patch_lines(..., labels=seg_labels).
    # This pins the root-cause invariant directly: every segment's label
    # list must stay the SAME length as its (possibly bridged) data array,
    # for every segment, not just the first.
    T = np.arange(12, dtype=float).reshape(6, 2)   # one dataset, A A B B A A
    segs, seg_labels, seg_cat, seg_bridge, seg_ds = helpers.segment_by_run(
        [T], ['A', 'A', 'B', 'B', 'A', 'A'], labels=[0, 1, 2, 3, 4, 5])
    breaks = {i + 1 for i in range(len(segs) - 1) if not seg_bridge[i]}
    segs = helpers.patch_lines(segs, breaks=breaks, labels=seg_labels)
    assert [s.shape[0] for s in segs] == [len(lab) for lab in seg_labels]
    # the two bridged runs (A->B, B->A) each gained one None-labeled point;
    # the real labels keep their original values and relative order
    assert seg_labels == [[0, 1, None], [2, 3, None], [4, 5]]


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
