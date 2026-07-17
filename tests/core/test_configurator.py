import pytest

from hypertools.core.configurator import get_default_options, apply_defaults
from hypertools.core.exceptions import HypertoolsIOError


def test_get_default_options_reads_config_ini():
    # config.ini mirrors the dispatcher signature defaults with option names
    # matching the real kwargs (2026-07 audit F23-001: the old [reduce]
    # section claimed algorithm=IncrementalPCA/ndims=3, which matched no
    # actual kwarg or default)
    opts = get_default_options()
    assert "reduce" in opts
    assert opts["reduce"]["reduce"] == "IncrementalPCA"
    assert opts["reduce"]["ndims"] is None


def test_get_default_options_values_are_typed():
    # values are literal-eval'd into real Python values (2026-07 audit
    # F23-002: they used to be raw INI strings like '3'/'False')
    opts = get_default_options()
    assert opts["cluster"]["n_clusters"] == 3
    assert isinstance(opts["cluster"]["n_clusters"], int)


def test_get_default_options_unknown_section_returns_empty():
    opts = get_default_options()
    assert opts["does_not_exist"] == {}


def test_get_default_options_missing_fname_raises():
    # missing custom config files fail loudly (2026-07 audit F23-003: they
    # used to be silently ignored by configparser.read)
    with pytest.raises(HypertoolsIOError, match="not found"):
        get_default_options(fname="/nonexistent/path/to/config.ini")


def test_apply_defaults_overrides_with_caller_kwargs():
    merged = apply_defaults("reduce", {"ndims": 5})
    assert merged["ndims"] == 5
    assert merged["reduce"] == "IncrementalPCA"
