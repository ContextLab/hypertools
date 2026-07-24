import pytest

from hypertools.core.shared import get


def test_get_indexes_lists():
    assert get([10, 20, 30], 1) == 20


def test_get_passes_scalars_through():
    assert get(5, 2) == 5
    assert get("x", 0) == "x"


def test_get_negative_index_python_conventional():
    # 2026-07 audit F23-012: negative indices used to return the whole list
    assert get([10, 20, 30], -1) == 30


def test_get_out_of_range_warns_and_returns_value():
    # a list shorter than the index still returns the whole value (broadcast
    # semantics), but now warns about the length mismatch (2026-07 audit
    # F23-012: this used to silently hand later datasets the whole list)
    with pytest.warns(UserWarning, match="no entry for dataset index"):
        assert get([1], 3) == [1]
