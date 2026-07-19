"""datawrangler API-surface probe.

Verification-first gate for HyperTools 1.0 (spec step 0). Asserts the installed
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


# upstream: datawrangler calls pd.concat(copy=...), deprecated in pandas 4.
# the filter names the DeprecationWarning BASE class on purpose:
# pandas.errors.Pandas4Warning subclasses it but does not exist on
# pandas 2.x, and pytest aborts (UsageError, exit 4) on filter
# categories it cannot import
@pytest.mark.filterwarnings(
    'ignore:The copy keyword is deprecated:DeprecationWarning')
def test_stack_unstack_roundtrip():
    a = pd.DataFrame(np.arange(6).reshape(3, 2), columns=["x", "y"])
    b = pd.DataFrame(np.arange(6, 14).reshape(4, 2), columns=["x", "y"])
    stacked = dw.stack([a, b])
    assert dw.zoo.is_multiindex_dataframe(stacked), "stack should yield a MultiIndex frame"
    restored = dw.unstack(stacked)
    assert isinstance(restored, list) and len(restored) == 2
    assert restored[0].shape == (3, 2) and restored[1].shape == (4, 2)
    np.testing.assert_array_equal(restored[0].to_numpy(), a.to_numpy())
    np.testing.assert_array_equal(restored[1].to_numpy(), b.to_numpy())


def test_funnel_generalizes_over_input_types():
    @dw.decorate.funnel
    def n_columns(data, **kwargs):
        if isinstance(data, list):
            return [d.shape[1] for d in data]
        return data.shape[1]

    arr = np.arange(6).reshape(3, 2)
    df = pd.DataFrame(arr, columns=["x", "y"])
    assert n_columns(arr) == 2
    assert n_columns(df) == 2
    assert n_columns([arr, arr]) == [2, 2]


def test_funnel_accepts_polars():
    pl = pytest.importorskip("polars")

    @dw.decorate.funnel
    def n_rows(data, **kwargs):
        if isinstance(data, list):
            return [d.shape[0] for d in data]
        return data.shape[0]

    pdf = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert n_rows(pdf) == 3


def test_dw_text_sklearn_embedding():
    docs = ["the cat sat", "the dog ran", "cats and dogs"]
    # dw.wrangle routes text through its text zoo; 'CountVectorizer' is the pure-sklearn
    # path (no torch/hf required). Returns one row per document.
    embedded = dw.wrangle(docs, model="CountVectorizer")
    frame = embedded[0] if isinstance(embedded, list) else embedded
    assert dw.zoo.is_dataframe(frame)
    assert frame.shape[0] == 3, "one row per document"
    assert frame.shape[1] >= 1, "at least one feature column"
