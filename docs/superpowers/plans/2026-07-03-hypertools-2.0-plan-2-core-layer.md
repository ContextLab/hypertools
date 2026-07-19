# HyperTools 2.0 — Plan 2: Core Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Stand up the `hypertools/core/` package — the eval-free model-dispatch, config, and shared utilities every functional module will build on — as **additive** scaffolding that keeps the existing 242-test suite green.

**Architecture:** Strangler step 1. Create `core/` with: `exceptions.py` (moved from `_shared/`), `shared.py` (`unpack_model` + `RobustDict`, eval-free, adapted from the fork), `configurator.py` + `config.ini` (central defaults via `dw.core.get_default_options`), and `model.py` (the canonical `apply_model` — the *working* dev-2.0 implementation relocated verbatim, then extended with the fork's `{'model','args','kwargs'}` dict form). Old locations (`_shared/exceptions.py`, `tools/apply_model.py`) become thin shims re-exporting from `core/`, so every current import path and test keeps working. The deeper arrays→DataFrames / dw-stacking conversion is deliberately deferred to the per-module plans (reduce/align/cluster), where each caller migrates behind the dw funnel while staying green.

**Tech Stack:** Python 3.12 (`.venv`), pytest, datawrangler 0.4.0, scikit-learn.

## Global Constraints

- **Interpreter:** ALL commands use `/Users/jmanning/hypertools/.venv/bin/python` / `.venv/bin/pip`. Never bare `python`/`pip`/`pytest` (those = broken anaconda 3.9).
- **pandas pinned `>=2.2,<3`** (dw#30). Don't change it.
- **Branch:** `dev-2.0-refactor`; never push master.
- **Strangler:** old import paths keep working via shims; the existing suite (baseline 242 passed + 21 probe = ~263) must stay green at every commit. Run focused tests per task; the full suite runs at plan close.
- **eval-free:** never resolve model strings with `eval`/`exec`. Use explicit registries / `importlib` + `getattr`.
- **datawrangler coordination:** dw bug/gap → `gh issue create -R ContextLab/data-wrangler` + log in `notes/datawrangler_coordination.md`. (dw#30 already open.)
- **No mocks.** Real calls only.
- **Commits:** after each task; trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Don't push unless asked.

## File Structure

- `hypertools/core/__init__.py` — re-exports the core public names. **Create.**
- `hypertools/core/exceptions.py` — the 3 exception classes. **Create** (content moved from `_shared/exceptions.py`).
- `hypertools/_shared/exceptions.py` — becomes a shim: `from ..core.exceptions import *`. **Modify.**
- `hypertools/core/shared.py` — `RobustDict`, `unpack_model`. **Create.**
- `hypertools/core/configurator.py` — `get_default_options`, `apply_defaults`. **Create.**
- `hypertools/core/config.ini` — central plot/reduce/align/cluster/data defaults. **Create.**
- `hypertools/core/model.py` — canonical `apply_model`, `supported_models`, `_build_registry` (relocated from `tools/apply_model.py`, then extended). **Create.**
- `hypertools/tools/apply_model.py` — becomes a shim re-exporting from `..core.model`. **Modify.**
- `tests/core/test_shared.py`, `test_configurator.py`, `test_model.py` — new unit tests. **Create.**

---

### Task 1: core package + exceptions (move + shim)

**Files:**
- Create: `hypertools/core/__init__.py`, `hypertools/core/exceptions.py`
- Modify: `hypertools/_shared/exceptions.py`
- Test: `tests/core/test_core_exceptions.py`

**Interfaces:**
- Produces: `hypertools.core.exceptions.{HypertoolsError,HypertoolsBackendError,HypertoolsIOError}`; `hypertools._shared.exceptions` re-exports the same objects (identity preserved).

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_core_exceptions.py`:

```python
def test_exceptions_importable_from_core():
    from hypertools.core.exceptions import (
        HypertoolsError, HypertoolsBackendError, HypertoolsIOError,
    )
    assert issubclass(HypertoolsBackendError, HypertoolsError)
    assert issubclass(HypertoolsIOError, HypertoolsError)
    assert issubclass(HypertoolsIOError, OSError)


def test_shared_exceptions_are_the_same_objects():
    # the _shared shim must re-export the SAME class objects, not copies
    from hypertools.core import exceptions as core_exc
    from hypertools._shared import exceptions as shared_exc
    assert core_exc.HypertoolsError is shared_exc.HypertoolsError
    assert core_exc.HypertoolsIOError is shared_exc.HypertoolsIOError
```

- [ ] **Step 2: Run it — expect failure**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/core/test_core_exceptions.py -q -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError: hypertools.core.exceptions`).

- [ ] **Step 3: Create `hypertools/core/exceptions.py`**

```python
class HypertoolsError(Exception):
    pass


class HypertoolsBackendError(HypertoolsError):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class HypertoolsIOError(HypertoolsError, OSError):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
```

- [ ] **Step 4: Create `hypertools/core/__init__.py`**

```python
"""HyperTools 2.0 core: model dispatch, configuration, and shared utilities."""

from .exceptions import (
    HypertoolsError,
    HypertoolsBackendError,
    HypertoolsIOError,
)
```

- [ ] **Step 5: Turn `hypertools/_shared/exceptions.py` into a shim**

Replace its entire contents with:

```python
# Moved to hypertools.core.exceptions (HyperTools 2.0 refactor). Kept as a shim
# so existing imports (hypertools._shared.exceptions) keep working.
from ..core.exceptions import (  # noqa: F401
    HypertoolsError,
    HypertoolsBackendError,
    HypertoolsIOError,
)
```

- [ ] **Step 6: Run the new test + the exceptions consumers, expect pass**

Run:
```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/core/test_core_exceptions.py -q -p no:cacheprovider
grep -rl "_shared.exceptions\|_shared import exceptions" hypertools tests | head
MPLBACKEND=Agg .venv/bin/python -c "import hypertools; print('import OK')"
```
Expected: tests pass; `import hypertools` still works.

- [ ] **Step 7: Commit**

```bash
git add hypertools/core/__init__.py hypertools/core/exceptions.py hypertools/_shared/exceptions.py tests/core/test_core_exceptions.py
git commit -m "$(printf 'refactor(core): move exceptions to hypertools.core; shim _shared\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 2: core/shared.py — RobustDict + unpack_model (eval-free)

**Files:**
- Create: `hypertools/core/shared.py`, `tests/core/test_shared.py`
- Modify: `hypertools/core/__init__.py` (export the new names)

**Interfaces:**
- Produces:
  - `RobustDict(dict)` — indexing a missing key returns `__default_value__` (default None) instead of raising.
  - `unpack_model(m, valid=None, parent_class=None)` — resolves a model spec to a concrete class/instance/dict WITHOUT eval: a string matching a `valid` class name → that class; an object that is a subclass/instance of `parent_class` → itself; a `{'model','args','kwargs'}` dict → same dict with its `'model'` recursively unpacked; an unmatched string → returned as-is (later resolved by the registry). Lists map elementwise. Consumed by every module dispatcher in Plans 4–6.

- [ ] **Step 1: Write failing tests**

Create `tests/core/test_shared.py`:

```python
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
```

- [ ] **Step 2: Run — expect failure** (`ModuleNotFoundError: hypertools.core.shared`).

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/core/test_shared.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement `hypertools/core/shared.py`**

```python
"""Shared core helpers: a forgiving dict and an eval-free model-spec resolver.

`unpack_model` is the eval-free replacement for the fork's string→eval model
lookup: names are matched against an explicit whitelist of classes, objects are
checked against a parent class, and dict specs have their inner model unpacked
recursively. Anything unmatched is returned unchanged for the registry to
resolve later.
"""


class RobustDict(dict):
    """dict whose missing keys return a default value instead of raising."""

    def __init__(self, *args, **kwargs):
        self.default_value = kwargs.pop("__default_value__", None)
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.default_value

    def __missing__(self, key):
        return self.default_value


def unpack_model(m, valid=None, parent_class=None):
    """Resolve a model specification without eval.

    Parameters
    ----------
    m : str, class, instance, dict, or list of these
    valid : list of classes whose ``__name__`` a string may match
    parent_class : if given, an ``m`` that is/instantiates a subclass passes through

    Returns
    -------
    The matched class, the object itself, a dict with its inner ``'model'``
    unpacked, or (for an unmatched string) the string unchanged.
    """
    if isinstance(m, list):
        return [unpack_model(x, valid=valid, parent_class=parent_class) for x in m]

    if valid is None:
        valid = []

    if isinstance(m, str) and m in [v.__name__ for v in valid]:
        return next(v for v in valid if v.__name__ == m)

    if parent_class is not None:
        try:
            if issubclass(m, parent_class):
                return m
        except TypeError:
            if isinstance(m, parent_class):
                return m

    if isinstance(m, dict) and all(k in m for k in ("model", "args", "kwargs")):
        resolved = dict(m)
        resolved["model"] = unpack_model(m["model"], valid=valid, parent_class=parent_class)
        return resolved

    if isinstance(m, str):
        return m

    raise ValueError(f"unknown model: {m!r}")
```

- [ ] **Step 4: Export from `core/__init__.py`** — append:

```python
from .shared import RobustDict, unpack_model
```

- [ ] **Step 5: Run tests — expect pass.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/core/test_shared.py -q -p no:cacheprovider`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add hypertools/core/shared.py hypertools/core/__init__.py tests/core/test_shared.py
git commit -m "$(printf 'feat(core): add RobustDict and eval-free unpack_model\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 3: core/configurator.py + config.ini — central defaults

**Files:**
- Create: `hypertools/core/config.ini`, `hypertools/core/configurator.py`, `tests/core/test_configurator.py`
- Modify: `hypertools/core/__init__.py`

**Interfaces:**
- Consumes: `dw.core.get_default_options`, `dw.core.update_dict`; `RobustDict` (Task 2).
- Produces:
  - `get_default_options(fname=None) -> RobustDict` — parses `core/config.ini` (merged over dw's own defaults); indexing an unknown section returns `{}`.
  - `apply_defaults(func_name, kwargs) -> dict` — returns `config[func_name]` merged with (overridden by) the caller's `kwargs`.

- [ ] **Step 1: Write failing tests**

Create `tests/core/test_configurator.py`:

```python
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
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/core/test_configurator.py -q -p no:cacheprovider`

- [ ] **Step 3: Create `hypertools/core/config.ini`**

```ini
[plot]
style = .
linestyle = -
linewidth = 1
markersize = 4
bigmarkersize = 6
opacity = 0.7
cmap = mako

[reduce]
ndims = 3
algorithm = IncrementalPCA

[align]
algorithm = hyper

[cluster]
n_clusters = 3
verbose = False
```

- [ ] **Step 4: Create `hypertools/core/configurator.py`**

```python
"""Central default options for hypertools, parsed from core/config.ini.

Uses datawrangler's config machinery so hypertools and dw share one defaults
mechanism (single source of truth). Values are returned in a RobustDict so a
lookup for an unconfigured function/section yields {} instead of KeyError.
"""
import os

import datawrangler as dw

from .shared import RobustDict


def get_default_options(fname=None):
    """Parse config.ini into a RobustDict of {section: {option: value}}."""
    if fname is None:
        fname = os.path.join(os.path.dirname(__file__), "config.ini")
    merged = dw.core.update_dict(dw.core.get_default_options(),
                                 dw.core.get_default_options(fname))
    return RobustDict(merged, __default_value__={})


def apply_defaults(func_name, kwargs=None):
    """Return the config defaults for ``func_name`` overridden by ``kwargs``."""
    defaults = dict(get_default_options()[func_name])
    if kwargs:
        defaults.update(kwargs)
    return defaults
```

- [ ] **Step 5: Export from `core/__init__.py`** — append:

```python
from .configurator import get_default_options, apply_defaults
```

- [ ] **Step 6: Run tests — expect pass.** If `dw.core.get_default_options(fname)` returns values in an unexpected shape (e.g. strings not coerced), adjust the test's `int(...)` casting to match dw's real return (record any delta in `notes/datawrangler_coordination.md`); do NOT change the config.ini values.

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/core/test_configurator.py -q -p no:cacheprovider`

- [ ] **Step 7: Commit**

```bash
git add hypertools/core/config.ini hypertools/core/configurator.py hypertools/core/__init__.py tests/core/test_configurator.py
git commit -m "$(printf 'feat(core): central config.ini defaults via datawrangler configurator\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 4: core/model.py — relocate apply_model + shim (behavior-preserving)

**Files:**
- Create: `hypertools/core/model.py`
- Modify: `hypertools/tools/apply_model.py` (→ shim), `hypertools/core/__init__.py`
- Test: `tests/core/test_model.py` (new, additive) + existing `tests/test_apply_model.py` (must stay green)

**Interfaces:**
- Produces: `hypertools.core.model.{apply_model, supported_models, _build_registry}` with behavior IDENTICAL to the current `tools/apply_model.apply_model`, PLUS acceptance of the fork's `{'model','args','kwargs'}` dict form in addition to dev-2.0's `{'model','params'}`. `hypertools.tools.apply_model` re-exports these (shim).

- [ ] **Step 1: Relocate the implementation**

Copy the ENTIRE current contents of `hypertools/tools/apply_model.py` into a new `hypertools/core/model.py`, changing only the one relative import at the top from:

```python
from .format_data import format_data as formatter
```
to:
```python
from ..tools.format_data import format_data as formatter
```
and update the two lazy imports inside `_build_registry` from `from .reduce import ...` / `from .cluster import ...` to `from ..tools.reduce import ...` / `from ..tools.cluster import ...`. Leave all logic byte-for-byte otherwise.

- [ ] **Step 2: Replace `hypertools/tools/apply_model.py` with a shim**

```python
# Relocated to hypertools.core.model (HyperTools 2.0). Shim keeps the old import
# path (hypertools.tools.apply_model) working for existing callers and tests.
from ..core.model import (  # noqa: F401
    apply_model,
    supported_models,
    _build_registry,
)
```

- [ ] **Step 3: Export from `core/__init__.py`** — append:

```python
from .model import apply_model, supported_models
```

- [ ] **Step 4: Confirm no regression (existing apply_model contract preserved)**

Run:
```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_apply_model.py -q -p no:cacheprovider
MPLBACKEND=Agg .venv/bin/python -c "import hypertools; print(hypertools.apply_model)"
```
Expected: all existing apply_model tests pass unchanged; `hypertools.apply_model` resolves.

- [ ] **Step 5: Add the fork dict-form (additive) — write the failing test**

Append to a new `tests/core/test_model.py`:

```python
import numpy as np
from hypertools.core.model import apply_model


def test_apply_model_accepts_fork_dict_form():
    # fork spec: {'model', 'args', 'kwargs'} must work the same as {'model','params'}
    data = np.random.RandomState(0).rand(20, 5)
    out = apply_model(data, {"model": "PCA", "args": [], "kwargs": {"n_components": 2}},
                      format_data=False)
    assert np.asarray(out).shape == (20, 2)


def test_apply_model_devtwo_dict_form_still_works():
    data = np.random.RandomState(0).rand(20, 5)
    out = apply_model(data, {"model": "PCA", "params": {"n_components": 2}},
                      format_data=False)
    assert np.asarray(out).shape == (20, 2)
```

- [ ] **Step 6: Run — expect the fork-form test to FAIL** (current `_resolve_model` only reads `model['params']`).

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/core/test_model.py -q -p no:cacheprovider`
Expected: `test_apply_model_accepts_fork_dict_form` FAILS (KeyError/params ignored), the devtwo one passes.

- [ ] **Step 7: Extend `_resolve_model` in `core/model.py` to accept both dict forms**

In `_resolve_model`, replace the dict-handling block:

```python
    if isinstance(model, dict):
        params = dict(model.get('params', {}))
        model = model['model']
```
with (accept fork `kwargs` as an alias for dev-2.0 `params`; ignore fork `args` for now — models take kwargs):

```python
    if isinstance(model, dict):
        # dev-2.0 form: {'model', 'params'}; fork form: {'model', 'args', 'kwargs'}
        params = dict(model.get('params', model.get('kwargs', {})))
        model = model['model']
```

- [ ] **Step 8: Run tests — expect pass.**

Run:
```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/core/test_model.py tests/test_apply_model.py -q -p no:cacheprovider
```
Expected: all pass (new fork-form + existing contract).

- [ ] **Step 9: Full-suite regression at plan close**

Run:
```bash
MPLBACKEND=Agg .venv/bin/python -m pytest -q -p no:cacheprovider
```
Expected: ~263 passed (242 baseline + 21 probe + new core tests), no failures.

- [ ] **Step 10: Commit**

```bash
git add hypertools/core/model.py hypertools/tools/apply_model.py hypertools/core/__init__.py tests/core/test_model.py
git commit -m "$(printf 'refactor(core): relocate apply_model to core.model; accept fork dict form\n\ntools/apply_model.py becomes a shim; core.model is now the source of truth.\nAdds {model,args,kwargs} spec support alongside {model,params}. Behavior\notherwise identical; existing apply_model tests unchanged.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Self-Review

**1. Spec coverage:** core/model.py (eval-free apply_model, source of truth) → Task 4; core/configurator + config.ini → Task 3; core/shared unpack_model + RobustDict → Task 2; core/exceptions → Task 1. `core/util.py` (helpers split) is intentionally DEFERRED — `_shared/helpers.py` is consumed widely and its split belongs with the plot/reduce migrations that touch those helpers; splitting it now would be churn with no consumer. Deep dw-stacking conversion of apply_model deferred to per-module plans (documented in Architecture).

**2. Placeholder scan:** every step has complete code or an exact move instruction; no TBD/"handle errors".

**3. Type consistency:** `RobustDict`/`unpack_model` names match across Tasks 2/3 and the `core/__init__.py` exports; `get_default_options` returns a `RobustDict` consumed by `apply_defaults`; `apply_model` signature is the current one (unchanged) plus dict-form tolerance.

## Execution Handoff

After Plan 2, Plan 3 (external + manip) migrates the vendored algorithms into `external/` and builds the `Manipulator` base + normalize/smooth/resample/zscore, consuming `core.unpack_model` / `core.apply_defaults` / `core.apply_model`.
