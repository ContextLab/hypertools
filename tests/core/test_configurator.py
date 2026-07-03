from hypertools.core.configurator import get_default_options, apply_defaults


def test_get_default_options_reads_config_ini():
    opts = get_default_options()
    assert "reduce" in opts
    assert opts["reduce"]["algorithm"] == "IncrementalPCA"
    assert int(opts["reduce"]["ndims"]) == 3


def test_get_default_options_unknown_section_returns_empty():
    opts = get_default_options()
    assert opts["does_not_exist"] == {}


def test_apply_defaults_overrides_with_caller_kwargs():
    merged = apply_defaults("reduce", {"ndims": 5})
    assert merged["ndims"] == 5
    assert merged["algorithm"] == "IncrementalPCA"
