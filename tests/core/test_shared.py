import pytest
from hypertools.core.shared import RobustDict, unpack_model


class Base:
    pass


class Child(Base):
    pass


def test_robustdict_missing_key_returns_default():
    d = RobustDict({"a": 1})
    assert d["a"] == 1
    assert d["missing"] is None
    d2 = RobustDict({"a": 1}, __default_value__={})
    assert d2["missing"] == {}


def test_unpack_model_resolves_valid_name_to_class():
    assert unpack_model("Child", valid=[Child, Base]) is Child


def test_unpack_model_passes_through_subclass():
    assert unpack_model(Child, valid=[], parent_class=Base) is Child


def test_unpack_model_unmatched_string_returns_string():
    assert unpack_model("KMeans", valid=[Child]) == "KMeans"


def test_unpack_model_dict_unpacks_inner_model():
    spec = {"model": "Child", "args": [], "kwargs": {}}
    out = unpack_model(spec, valid=[Child])
    assert out["model"] is Child and out["args"] == [] and out["kwargs"] == {}


def test_unpack_model_list_maps_elementwise():
    out = unpack_model(["Child", "Base"], valid=[Child, Base])
    assert out == [Child, Base]


class Unrelated:
    pass


def test_unpack_model_wrong_type_instance_with_parent_class_raises():
    with pytest.raises(ValueError, match="unknown model"):
        unpack_model(Unrelated(), valid=[], parent_class=Base)


def test_unpack_model_instance_passes_through_when_parent_class_none():
    obj = Unrelated()
    assert unpack_model(obj, valid=[], parent_class=None) is obj
