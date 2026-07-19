# HyperTools 2.0 — Plan 3: external/ + manip/ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Quarantine the vendored algorithms in `hypertools/external/` and build the `hypertools/manip/` package — a `Manipulator` base plus DataFrame-based `Normalize`/`ZScore`/`Smooth`/`Resample` classes and a `hyp.manip` dispatcher — while keeping the array-based `hyp.normalize` compat API and the full suite green.

**Architecture:** Strangler steps 2–3, additive. `external/` receives `_externals/ppca.py`→`external/ppca.py` and `_externals/srm.py`→`external/brainiak.py` (license headers retained; `_externals` becomes shims). `manip/` adopts the fork's dw-based `Manipulator` classes, which are scikit-learn-compatible (`fit`/`transform`/`fit_transform`). The `manip` dispatcher is `@dw.decorate.funnel`-wrapped so it coerces any input (array/DataFrame/list/text/polars) to DataFrames, then calls the resolved Manipulator's `fit_transform` **directly** — NOT through the array-based `core.apply_model` (whose numpy coercion is incompatible with these DataFrame-column operations). `hyp.normalize` (dev-2.0, array/mode-based) is left untouched and coexists.

**Tech Stack:** Python 3.12 (`.venv`), pytest, datawrangler 0.4.0, scipy, scikit-learn.

## Global Constraints

- **Interpreter:** ALL commands use `/Users/jmanning/hypertools/.venv/bin/python`. Never bare `python`/`pip`/`pytest`.
- **pandas pinned `>=2.2,<3`** (dw#30). Don't change it.
- **Branch:** `dev-2.0-refactor`; never push master.
- **Strangler:** old import paths keep working via shims; the full suite (currently 276 passed) stays green. Focused tests per task; full suite at plan close.
- **Source-of-truth rule:** dev-2.0 code is the trusted template; **fork-only code (`Smooth`/`Resample`/`ZScore`, and the fork `Normalize`) is a starting point to VALIDATE and FIX, not trust** — prove each with real tests over array/list/DataFrame/polars inputs.
- **Vendored code:** keep the Apache-2.0 (brainiak) and pca-magic license headers in `external/`; add a `# Vendored from <url>` banner.
- **eval-free.** No mocks. Real calls only.
- **Commits:** after each task; trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Don't push unless asked.

## File Structure

- `hypertools/external/__init__.py`, `external/ppca.py`, `external/brainiak.py` — vendored. **Create** (content moved).
- `hypertools/_externals/ppca.py`, `_externals/srm.py` — become shims. **Modify.**
- `hypertools/core/shared.py` — add `get`. **Modify.**
- `hypertools/manip/__init__.py`, `common.py`, `normalize.py`, `zscore.py`, `smooth.py`, `resample.py`, `manip.py`. **Create.**
- `hypertools/__init__.py` — add `manip` (keep `normalize`). **Modify** (Task 6).
- `tests/manip/…`, `tests/core/test_get.py`, `tests/external/…`. **Create.**

---

### Task 1: external/ package (move vendored ppca + srm; shim _externals)

**Files:**
- Create: `hypertools/external/__init__.py`, `hypertools/external/ppca.py`, `hypertools/external/brainiak.py`
- Modify: `hypertools/_externals/ppca.py`, `hypertools/_externals/srm.py`
- Test: `tests/external/test_external_move.py`

**Interfaces:**
- Produces: `hypertools.external.ppca.PPCA`, `hypertools.external.brainiak.{SRM,DetSRM}`; `_externals.*` re-export the same objects.

- [ ] **Step 1: Write the failing test**

Create `tests/external/__init__.py` (empty) and `tests/external/test_external_move.py`:

```python
def test_external_ppca_importable():
    from hypertools.external.ppca import PPCA
    assert hasattr(PPCA, "fit")


def test_external_brainiak_importable():
    from hypertools.external.brainiak import SRM, DetSRM
    assert hasattr(SRM, "fit") and hasattr(DetSRM, "fit")


def test_externals_shims_are_same_objects():
    from hypertools.external.ppca import PPCA as new_ppca
    from hypertools._externals.ppca import PPCA as old_ppca
    from hypertools.external.brainiak import SRM as new_srm
    from hypertools._externals.srm import SRM as old_srm
    assert new_ppca is old_ppca
    assert new_srm is old_srm
```

- [ ] **Step 2: Run — expect failure** (`hypertools.external` missing).

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/external/test_external_move.py -q -p no:cacheprovider`

- [ ] **Step 3: Move the files with git (preserve history + license headers)**

```bash
cd /Users/jmanning/hypertools
git mv hypertools/_externals/ppca.py hypertools/external/ppca.py
git mv hypertools/_externals/srm.py hypertools/external/brainiak.py
```
Create `hypertools/external/__init__.py`:
```python
"""Vendored third-party algorithms (kept under their original licenses).

- ppca.py     : Probabilistic PCA (replaces the unmaintained pca-magic dependency)
- brainiak.py : Shared Response Model family (Apache-2.0, Intel Corporation)
"""
```
Add a banner as the FIRST line of each moved file (above the existing license header):
- `external/brainiak.py`: `# Vendored from https://github.com/brainiak/brainiak (Apache-2.0). See header below.`
- `external/ppca.py`: `# Vendored from the pca-magic project. See license in-repo.`

- [ ] **Step 4: Recreate `_externals` shims**

`hypertools/_externals/ppca.py`:
```python
# Moved to hypertools.external.ppca (HyperTools 2.0). Shim preserves the old path.
from ..external.ppca import *  # noqa: F401,F403
from ..external.ppca import PPCA  # noqa: F401
```
`hypertools/_externals/srm.py`:
```python
# Moved to hypertools.external.brainiak (HyperTools 2.0). Shim preserves the old path.
from ..external.brainiak import *  # noqa: F401,F403
from ..external.brainiak import SRM, DetSRM  # noqa: F401
```
(If `external/brainiak.py` defines `RSRM` too, add it to the shim's explicit import.)

- [ ] **Step 5: Update in-repo importers to the new path**

```bash
grep -rn "_externals" hypertools --include=*.py | grep -v "hypertools/_externals/"
```
For each hit (e.g. `hypertools/tools/format_data.py: from .._externals.ppca import PPCA`), the shim keeps it working — leave it, since Plan 3 is additive. Do NOT rewrite callers yet.

- [ ] **Step 6: Run tests + import + a focused consumer test**

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/external/test_external_move.py tests/test_reduce.py -q -p no:cacheprovider
.venv/bin/python -c "import hypertools; print('OK')"
```
Expected: pass; PPCA-using reduce path still works via the shim.

- [ ] **Step 7: Commit**

```bash
git add hypertools/external hypertools/_externals tests/external
git commit -m "$(printf 'refactor(external): quarantine vendored ppca+srm in hypertools.external; shim _externals\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 2: core.get — elementwise list indexer (Resample prerequisite)

**Files:**
- Modify: `hypertools/core/shared.py`, `hypertools/core/__init__.py`
- Test: `tests/core/test_get.py`

**Interfaces:**
- Produces: `get(value, i)` → `value[i]` when `value` is a list/tuple (and `i` in range), else `value`. Used by `Resample.transformer` to broadcast scalar-or-per-dataset kwargs.

- [ ] **Step 1: Failing test** — create `tests/core/test_get.py`:

```python
from hypertools.core.shared import get


def test_get_indexes_lists():
    assert get([10, 20, 30], 1) == 20


def test_get_passes_scalars_through():
    assert get(5, 2) == 5
    assert get("x", 0) == "x"


def test_get_out_of_range_returns_value():
    # a list shorter than the index returns the whole value (broadcast semantics)
    assert get([1], 3) == [1]
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/core/test_get.py -q -p no:cacheprovider`

- [ ] **Step 3: Add `get` to `hypertools/core/shared.py`** (append):

```python
def get(value, i):
    """Return value[i] for a list/tuple (if in range), else value itself.

    Lets a manipulator accept either one shared parameter or a per-dataset list.
    """
    if isinstance(value, (list, tuple)):
        if 0 <= i < len(value):
            return value[i]
        return value
    return value
```

- [ ] **Step 4: Export** — add `get` to the `from .shared import ...` line in `hypertools/core/__init__.py`.

- [ ] **Step 5: Run — expect pass.** Commit:

```bash
git add hypertools/core/shared.py hypertools/core/__init__.py tests/core/test_get.py
git commit -m "$(printf 'feat(core): add get() elementwise list indexer (Resample prereq)\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 3: manip package + Manipulator base + dispatcher skeleton

**Files:**
- Create: `hypertools/manip/__init__.py`, `hypertools/manip/common.py`, `hypertools/manip/manip.py`
- Test: `tests/manip/test_manip_base.py`

**Interfaces:**
- Produces:
  - `Manipulator(BaseEstimator)` — `__init__(**kwargs)` pops `data`/`fitter`/`transformer`/`required`; `fit(data)` calls `fitter` and sets returned params as attributes; `transform()` calls `transformer`; `fit_transform(data)`.
  - `manip(data, model='ZScore', **kwargs)` — `@dw.decorate.funnel`; resolves `model` via `unpack_model` against the manip whitelist; returns the Manipulator's `fit_transform(data)` output. Consumed by `hyp.manip` (Task 6).

- [ ] **Step 1: Failing test** — create `tests/manip/__init__.py` (empty) and `tests/manip/test_manip_base.py`:

```python
import numpy as np
import pandas as pd
from hypertools.manip.common import Manipulator


def _mean_center_fitter(data, **kwargs):
    return {"mean": data.mean(axis=0)}


def _mean_center_transformer(data, **kwargs):
    return data - kwargs["mean"]


def test_manipulator_fit_transform_roundtrip():
    df = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3), columns=list("abc"))
    m = Manipulator(fitter=_mean_center_fitter, transformer=_mean_center_transformer,
                    required=["mean"], data=None)
    out = m.fit_transform(df)
    assert np.allclose(out.mean(axis=0).to_numpy(), 0.0)
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/manip/test_manip_base.py -q -p no:cacheprovider`

- [ ] **Step 3: Create `hypertools/manip/common.py`** (ported from `jeremy/master:hypertools/manip/common.py`, using stdlib assertions; no dw dependency in the base itself):

```python
"""Base class for hypertools manipulators (scikit-learn compatible).

A Manipulator wraps a (fitter, transformer, required-params) triple: `fit`
runs the fitter and stores the returned dict as attributes; `transform` runs
the transformer with those params. Child classes (Normalize, ZScore, Smooth,
Resample) supply the three pieces plus their defaults.
"""
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class Manipulator(BaseEstimator):
    def __init__(self, **kwargs):
        self.data = kwargs.pop("data", None)
        self.fitter = kwargs.pop("fitter", None)
        self.transformer = kwargs.pop("transformer", None)
        self.required = kwargs.pop("required", [])
        self.kwargs = kwargs

    def fit(self, data):
        assert data is not None, ValueError("cannot manipulate an empty dataset")
        self.data = data
        if self.fitter is None:
            return
        params = self.fitter(data, **self.kwargs)
        assert isinstance(params, dict), ValueError("fit function must return a dictionary")
        assert all(r in params for r in self.required), \
            ValueError("one or more required fields not returned")
        for k, v in params.items():
            setattr(self, k, v)

    def transform(self, *_):
        if self.data is None:
            raise NotFittedError("must fit manipulator before transforming data")
        for r in self.required:
            if not hasattr(self, r):
                raise NotFittedError(f"missing fitted attribute: {r}")
        if self.transformer is None:
            return self.data
        required_params = {r: getattr(self, r) for r in self.required}
        merged = {**required_params, **self.kwargs}
        return self.transformer(self.data, **merged)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform()
```

- [ ] **Step 4: Create `hypertools/manip/manip.py`** (dispatcher; funnel coerces to DataFrames, then applies the Manipulator directly):

```python
"""hyp.manip dispatcher: resolve a manipulator spec and fit_transform it.

Wrapped by datawrangler's funnel so any input (array / DataFrame / list / text
/ polars) arrives as DataFrame(s); the resolved Manipulator (sklearn-compatible,
DataFrame-based) is applied directly rather than via the array-based
core.apply_model.
"""
import datawrangler as dw

from .common import Manipulator
from .normalize import Normalize
from .zscore import ZScore
from .smooth import Smooth
from .resample import Resample
from ..core.shared import unpack_model
from ..core.configurator import apply_defaults


MANIPULATORS = [Normalize, ZScore, Smooth, Resample]


@dw.decorate.funnel
def manip(data, model="ZScore", **kwargs):
    resolved = unpack_model(model, valid=MANIPULATORS, parent_class=Manipulator)
    if isinstance(resolved, type):
        resolved = resolved(**kwargs)
    elif isinstance(resolved, dict):
        cls = resolved["model"]
        resolved = cls(*resolved.get("args", []), **resolved.get("kwargs", {}))
    return resolved.fit_transform(data)
```
(Note: `manip.py` imports Normalize/ZScore/Smooth/Resample, created in Tasks 4–5. Until then this import fails — so create `manip.py` in Task 5's final step, OR create stub child modules now. To keep this task independently testable, create `manip.py` in Task 5 Step 6; for THIS task, `__init__.py` exports only `Manipulator`.)

- [ ] **Step 5: Create `hypertools/manip/__init__.py`**:

```python
from .common import Manipulator
```

- [ ] **Step 6: Run the base test — expect pass. Commit:**

```bash
git add hypertools/manip/__init__.py hypertools/manip/common.py tests/manip
git commit -m "$(printf 'feat(manip): add Manipulator base class\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 4: manip Normalize + ZScore (port + validate)

**Files:**
- Create: `hypertools/manip/normalize.py`, `hypertools/manip/zscore.py`, `tests/manip/test_normalize_zscore.py`
- Modify: `hypertools/manip/__init__.py`

**Interfaces:**
- Produces `Normalize(min=0, max=1, axis=0)` (min-max scaling) and `ZScore(axis=0)` (mean/std), both `Manipulator` subclasses with dw-`apply_stacked` fitter/transformer. Consumed by `manip.py` (Task 5) and `hyp.manip`.

- [ ] **Step 1: Failing tests** — create `tests/manip/test_normalize_zscore.py`:

```python
import numpy as np
import pandas as pd
from hypertools.manip.normalize import Normalize
from hypertools.manip.zscore import ZScore


def test_normalize_scales_to_unit_range():
    df = pd.DataFrame(np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]]), columns=["a", "b"])
    out = Normalize(min=0, max=1, axis=0).fit_transform(df)
    assert np.isclose(out["a"].min(), 0.0) and np.isclose(out["a"].max(), 1.0)
    assert np.isclose(out["b"].min(), 0.0) and np.isclose(out["b"].max(), 1.0)


def test_zscore_zero_mean_unit_std():
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.rand(50, 3), columns=list("abc"))
    out = ZScore(axis=0).fit_transform(df)
    assert np.allclose(out.mean(axis=0).to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(out.std(axis=0, ddof=1).to_numpy(), 1.0, atol=1e-6)
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/manip/test_normalize_zscore.py -q -p no:cacheprovider`

- [ ] **Step 3: Create `hypertools/manip/normalize.py`** — port from `jeremy/master:hypertools/manip/normalize.py` (the fitter/transformer with `@dw.decorate.apply_stacked` and the `Normalize(Manipulator)` class). Validate against pandas 2.3.3: if `pd.Series(index=data.columns)` raises a dtype FutureWarning/error, construct with `dtype=float`. Fix any such issue and note it in the report.

- [ ] **Step 4: Create `hypertools/manip/zscore.py`** — port from `jeremy/master:hypertools/manip/zscore.py` similarly (apply the same `pd.Series(..., dtype=float)` fix if needed).

- [ ] **Step 5: Export** — update `hypertools/manip/__init__.py`:

```python
from .common import Manipulator
from .normalize import Normalize
from .zscore import ZScore
```

- [ ] **Step 6: Run tests — expect pass** (fix ports until green; do not weaken assertions). Also test list + array inputs:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/manip/test_normalize_zscore.py -q -p no:cacheprovider
.venv/bin/python -c "
import numpy as np, pandas as pd
from hypertools.manip.zscore import ZScore
out = ZScore().fit_transform([pd.DataFrame(np.random.rand(10,2)), pd.DataFrame(np.random.rand(8,2))])
print('list-in type:', type(out).__name__)
"
```

- [ ] **Step 7: Commit**

```bash
git add hypertools/manip/normalize.py hypertools/manip/zscore.py hypertools/manip/__init__.py tests/manip/test_normalize_zscore.py
git commit -m "$(printf 'feat(manip): add Normalize (min-max) and ZScore manipulators\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 5: manip Smooth + Resample (port + FIX + wire dispatcher)

**Files:**
- Create: `hypertools/manip/smooth.py`, `hypertools/manip/resample.py`, `tests/manip/test_smooth_resample.py`
- Modify: `hypertools/manip/__init__.py`, and create `hypertools/manip/manip.py` (from Task 3 Step 4)

**Interfaces:**
- Produces `Smooth(axis=0, kernel_width=11, order=3, maintain_bounds=True)` (Savitzky–Golay) and `Resample(axis=0, n_samples=100)` (pchip); plus the `manip` dispatcher wired to all four manipulators.

- [ ] **Step 1: Failing tests** — create `tests/manip/test_smooth_resample.py`:

```python
import numpy as np
import pandas as pd
from hypertools.manip.smooth import Smooth
from hypertools.manip.resample import Resample
from hypertools.manip.manip import manip


def test_smooth_reduces_variance_of_noisy_signal():
    rng = np.random.RandomState(0)
    t = np.linspace(0, 4 * np.pi, 200)
    clean = np.sin(t)
    noisy = clean + rng.normal(0, 0.5, size=t.shape)
    df = pd.DataFrame({"x": noisy})
    out = Smooth(kernel_width=21, order=3).fit_transform(df)
    # smoothed signal is closer to the clean signal than the noisy input
    assert np.mean((out["x"].to_numpy() - clean) ** 2) < np.mean((noisy - clean) ** 2)


def test_resample_changes_row_count():
    df = pd.DataFrame({"x": np.linspace(0, 1, 50), "y": np.linspace(1, 2, 50)})
    out = Resample(n_samples=17).fit_transform(df)
    assert out.shape[0] == 17


def test_manip_dispatcher_by_name():
    df = pd.DataFrame(np.random.RandomState(1).rand(20, 3))
    out = manip(df, model="ZScore")
    assert np.allclose(np.asarray(out).mean(axis=0), 0.0, atol=1e-9)
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/manip/test_smooth_resample.py -q -p no:cacheprovider`

- [ ] **Step 3: Create `hypertools/manip/smooth.py`** — port from `jeremy/master:hypertools/manip/smooth.py`. FIX the known fork issues while porting: remove the unused `scipy.interpolate` import; replace the `maintain_bounds` chained `.loc` assignment with `np.clip` (`smoothed[c] = np.clip(smoothed[c].to_numpy(), kwargs['min'][c], kwargs['max'][c])`) to avoid pandas chained-assignment warnings; construct `pd.Series(..., dtype=float)` where the fork omits dtype. Keep the Savitzky–Golay behavior. Document each fix in the report.

- [ ] **Step 4: Create `hypertools/manip/resample.py`** — port from `jeremy/master:hypertools/manip/resample.py`; its `transformer` uses `from ..core import get` — import instead `from ..core.shared import get` (added in Task 2). Validate pchip resampling; fix `pd.Series`/`pd.DataFrame` dtype omissions if they warn.

- [ ] **Step 5: Export** — update `hypertools/manip/__init__.py`:

```python
from .common import Manipulator
from .normalize import Normalize
from .zscore import ZScore
from .smooth import Smooth
from .resample import Resample
from .manip import manip
```

- [ ] **Step 6: Create `hypertools/manip/manip.py`** — exactly the dispatcher from Plan-3 Task 3 Step 4.

- [ ] **Step 7: Run tests — expect pass** (fix ports until green; do not weaken assertions).

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/manip/ -q -p no:cacheprovider`

- [ ] **Step 8: GAUSSIAN-SMOOTH NOTE (do not implement here)** — the classic weights-trajectory gif needs gaussian smoothing (var=300), which Savitzky–Golay `Smooth` does NOT provide. Record in `notes/session_2026-07-03_refactor_plans1-2.md` under "Plan 3" that a gaussian smoothing mode/option is still owed and will be added when the plot/weights pipeline is migrated (Plan 6). Do not block Plan 3 on it.

- [ ] **Step 9: Commit**

```bash
git add hypertools/manip tests/manip
git commit -m "$(printf 'feat(manip): add Smooth (savgol) and Resample (pchip) + manip dispatcher\n\nFork ports validated/fixed: np.clip bounds, core.shared.get import, Series dtypes.\nGaussian-smooth mode still owed (Plan 6, weights pipeline).\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

### Task 6: wire hyp.manip (keep hyp.normalize compat) + plan-close regression

**Files:**
- Modify: `hypertools/__init__.py`
- Test: `tests/manip/test_public_api.py`

**Interfaces:**
- Produces: `hypertools.manip` (new) alongside the untouched `hypertools.normalize` (dev-2.0 array/mode-based compat).

- [ ] **Step 1: Failing test** — create `tests/manip/test_public_api.py`:

```python
import numpy as np
import hypertools as hyp


def test_hyp_manip_exposed():
    out = hyp.manip(np.random.RandomState(0).rand(20, 3), model="ZScore")
    assert np.allclose(np.asarray(out).mean(axis=0), 0.0, atol=1e-9)


def test_hyp_normalize_compat_still_present():
    # dev-2.0 array/mode API must be unchanged
    out = hyp.normalize(np.random.RandomState(0).rand(10, 4), normalize="across")
    assert np.asarray(out).shape == (10, 4)
```

- [ ] **Step 2: Run — expect failure** (`hyp.manip` missing).

- [ ] **Step 3: Add the export** — in `hypertools/__init__.py`, add after the existing imports:

```python
from .manip.manip import manip
```
Do NOT remove or change the existing `from .tools.normalize import normalize` (or however `normalize` is currently exported) — verify it is still present.

- [ ] **Step 4: Run the API test + full suite (plan close)**

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/manip/test_public_api.py -q -p no:cacheprovider
MPLBACKEND=Agg .venv/bin/python -m pytest -q -p no:cacheprovider
```
Expected: API test passes; full suite green (276 + new manip/external/core tests), no regressions.

- [ ] **Step 5: Commit**

```bash
git add hypertools/__init__.py tests/manip/test_public_api.py
git commit -m "$(printf 'feat(api): expose hyp.manip; hyp.normalize compat unchanged\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Self-Review

**1. Spec coverage:** external/ quarantine → Task 1; manip Manipulator + Normalize/ZScore/Smooth/Resample + dispatcher → Tasks 3–5; `hyp.manip` alias + `hyp.normalize` compat → Task 6; `core.get` prereq → Task 2. Fork-only Smooth/Resample/ZScore validated+fixed with real tests (source-of-truth rule). Gaussian-smooth gap explicitly tracked, not silently dropped.

**2. Placeholder scan:** every step has complete code or an exact port instruction + the specific fixes to apply; no "handle errors"/TBD.

**3. Type consistency:** `Manipulator` API (`fit`/`transform`/`fit_transform`) consistent across Tasks 3–6; `unpack_model`/`apply_defaults`/`get` names match core exports; `manip(data, model=...)` signature consistent between `manip.py` and the tests.

## Execution Handoff

After Plan 3, Plan 4 (reduce + align + cluster) performs the core arrays→DataFrames migration behind the dw funnel, consuming `external.ppca`/`external.brainiak`, `core.apply_model`, `core.unpack_model`, and the manip smoothing where the weights pipeline needs it.
