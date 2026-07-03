from hypertools.core.shared import get


def test_get_indexes_lists():
    assert get([10, 20, 30], 1) == 20


def test_get_passes_scalars_through():
    assert get(5, 2) == 5
    assert get("x", 0) == "x"


def test_get_out_of_range_returns_value():
    # a list shorter than the index returns the whole value (broadcast semantics)
    assert get([1], 3) == [1]
