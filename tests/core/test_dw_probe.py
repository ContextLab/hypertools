"""datawrangler API-surface probe.

Verification-first gate for HyperTools 2.0 (spec step 0). Asserts the installed
datawrangler exposes every symbol and behavior the class-based refactor builds
on. A failure means dw drifted from what we verified at 0.4.0: file a
ContextLab/data-wrangler issue, then mark the specific check xfail with the
issue link (see notes/datawrangler_coordination.md). Real calls only.
"""
import importlib

import numpy as np
import pandas as pd
import pytest

import datawrangler as dw


# (module path, attribute) pairs the refactor depends on.
REQUIRED_SYMBOLS = [
    ("datawrangler.decorate", "funnel"),
    ("datawrangler.decorate", "apply_stacked"),
    ("datawrangler.decorate", "list_generalizer"),
    ("datawrangler", "stack"),
    ("datawrangler", "unstack"),
    ("datawrangler", "wrangle"),
    ("datawrangler.core", "update_dict"),
    ("datawrangler.core", "apply_defaults"),
    ("datawrangler.core", "get_default_options"),
    ("datawrangler.zoo", "is_dataframe"),
    ("datawrangler.zoo", "is_array"),
    ("datawrangler.zoo", "is_multiindex_dataframe"),
    ("datawrangler.zoo.text", "wrangle_text"),
    ("datawrangler.zoo.text", "apply_text_model"),
    ("datawrangler.zoo.text", "is_hugging_face_model"),
    ("datawrangler.zoo.text", "get_text_model"),
    ("datawrangler.zoo.text", "get_corpus"),
]


@pytest.mark.parametrize("module_path,attr", REQUIRED_SYMBOLS)
def test_dw_symbols_exist(module_path, attr):
    module = importlib.import_module(module_path)
    assert hasattr(module, attr), (
        f"datawrangler {dw.__version__} is missing {module_path}.{attr}; "
        f"file a ContextLab/data-wrangler issue and xfail this param with the link"
    )
