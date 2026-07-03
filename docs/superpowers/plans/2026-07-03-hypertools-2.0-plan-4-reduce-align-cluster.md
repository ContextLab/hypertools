# HyperTools 2.0 — Plan 4: reduce/ + align/ + cluster/ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Re-home dev-2.0's working `reduce`/`describe`, `align`/`procrustes`, and `cluster` into the fork's class-based module layout (`hypertools/reduce/`, `hypertools/align/`, `hypertools/cluster/`) — an `Aligner` base + `HyperAlign`/`Procrustes`/`SharedResponseModel`/`DeterministicSharedResponseModel`/`NullAlign` children + dispatchers — while keeping the classic `hyp.reduce/align/cluster/describe` APIs and the full suite green.

**Architecture:** Strangler steps 4–6, additive. dev-2.0's implementations are the **trusted algorithm source**; the fork supplies **organization/shape only**. reduce & cluster keep their complete dev-2.0 logic (the "children" are sklearn estimators resolved eval-free by `core.apply_model`'s registry — no hypertools base class is warranted there, matching the fork, which ships no `reduce/common.py` or `cluster/common.py`). **align** gets the full class trio because its algorithms (hyperalignment, Procrustes, SRM adapters) are hypertools-owned; the align dispatcher applies a resolved `Aligner` **directly via `fit_transform`** (like `manip`), NOT through `core.apply_model` (which stacks-and-fits-once — wrong for list alignment). Old `tools/{reduce,describe,align,procrustes,cluster}.py` become thin shims re-exporting the new locations (this preserves `core.model._build_registry`'s `from ..tools.reduce import models` / `from ..tools.cluster import models, mixture_models` — **no `core/model.py` change**).

**Tech Stack:** Python 3.12 (`.venv`), pytest, datawrangler 0.4.0, scikit-learn, scipy, numpy.

## Global Constraints

- **Interpreter:** ALL commands use `/Users/jmanning/hypertools/.venv/bin/python`. Never bare `python`/`pip`/`pytest`.
- **pandas `>=2.2.0`** (no upper pin — dw 0.5.0 fixed dw#30). Validated on pandas 3.0.3 / dw 0.5.0 / numpy 2.3.5.
- **Branch:** `dev-2.0-refactor`; never push master.
- **Strangler / green:** old import paths keep working via shims; the full suite (currently **293 passed**) stays green. Focused tests per task; full suite only at plan close.
- **Source-of-truth rule:** dev-2.0 code is the trusted template. Preserve its behavior exactly — the existing `tests/test_reduce.py`, `tests/test_describe.py`, `tests/test_align.py`, `tests/test_procrustes.py`, `tests/test_cluster.py` are **acceptance gates that must stay green** through their `tools/` shims. Fork-only capabilities (DetSRM, NullAlign, Procrustes-as-model, per-dataset `return_model`) are ports to VALIDATE with real tests, not trust.
- **RSRM not carried:** `external.brainiak` vendors `SRM` + `DetSRM` only (Plan 3 decision). `RobustSharedResponseModel` is therefore **out of scope for Plan 4** — document it as owed (would require vendoring RSRM into `external.brainiak` first). Do not import a non-existent `RSRM`.
- **eval-free.** No `eval()`/`exec()` for model loading. No mocks. Real calls only.
- **Commits:** after each task; trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Don't push unless asked.

## File Structure

- `hypertools/reduce/__init__.py`, `reduce/reduce.py` (← tools/reduce.py), `reduce/describe.py` (← tools/describe.py). **Create.**
- `hypertools/align/__init__.py`, `align/common.py` (Aligner + pad/trim_and_pad), `align/procrustes.py` (procrustes fn + Procrustes child), `align/hyperalign.py` (HyperAlign, dev-2.0 rescale algo), `align/srm.py` (SharedResponseModel + DeterministicSharedResponseModel), `align/null.py` (NullAlign), `align/align.py` (dispatcher). **Create.**
- `hypertools/cluster/__init__.py`, `cluster/cluster.py` (← tools/cluster.py). **Create.**
- `hypertools/tools/{reduce,describe,align,procrustes,cluster}.py` — become **shims**. **Modify.**
- `hypertools/__init__.py` — expose `hyp.reduce`/`align`/`cluster`/`describe` from new locations (keep classic names). **Modify** (Task 7).
- `tests/reduce/…`, `tests/align/…`, `tests/cluster/…`. **Create.**

---

### Task 1: reduce/ package (move reduce + describe; shim tools/)

**Files:**
- Create: `hypertools/reduce/__init__.py`, `hypertools/reduce/reduce.py`, `hypertools/reduce/describe.py`
- Modify: `hypertools/tools/reduce.py`, `hypertools/tools/describe.py` (→ shims)
- Test: `tests/reduce/__init__.py`, `tests/reduce/test_reduce_module.py`

**Interfaces:**
- Produces: `hypertools.reduce.reduce.reduce(x, reduce='IncrementalPCA', ndims=None, internal=False, format_data=True)`, `hypertools.reduce.reduce.models` (dict), `hypertools.reduce.reduce.reduce_list`, `hypertools.reduce.describe.describe(...)`. `tools.reduce` / `tools.describe` re-export these (registry import `from ..tools.reduce import models` keeps resolving).

- [ ] **Step 1: Write the failing test** — create `tests/reduce/__init__.py` (empty) and `tests/reduce/test_reduce_module.py`:

```python
import numpy as np
import pandas as pd


def test_reduce_new_path_importable_and_reduces():
    from hypertools.reduce.reduce import reduce
    data = [np.random.RandomState(0).rand(10, 6) for _ in range(2)]
    out = reduce(data, ndims=3)
    assert isinstance(out, list) and out[0].shape == (10, 3)


def test_reduce_shim_is_same_function():
    from hypertools.reduce.reduce import reduce as new_reduce
    from hypertools.tools.reduce import reduce as old_reduce
    assert new_reduce is old_reduce


def test_reduce_registry_models_still_exposed_via_tools():
    # core.model._build_registry imports `from ..tools.reduce import models`
    from hypertools.tools.reduce import models
    assert 'PCA' in models and 'IncrementalPCA' in models


def test_reduce_accepts_dataframe():
    df = pd.DataFrame(np.random.RandomState(1).rand(12, 5))
    out = reduce_df = __import__('hypertools.reduce.reduce', fromlist=['reduce']).reduce(df, ndims=2)
    assert np.asarray(out).shape == (12, 2)


def test_describe_new_path():
    from hypertools.reduce.describe import describe
    data = np.random.RandomState(2).rand(20, 8)
    result = describe(data, max_dims=4, show=False)
    assert 'average' in result and 'individual' in result
```

- [ ] **Step 2: Run — expect failure** (`hypertools.reduce.reduce` missing).

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/reduce/test_reduce_module.py -q -p no:cacheprovider`

- [ ] **Step 3: Move the files with git (preserve history)**

```bash
cd /Users/jmanning/hypertools
git mv hypertools/tools/reduce.py hypertools/reduce/reduce.py
git mv hypertools/tools/describe.py hypertools/reduce/describe.py
```

- [ ] **Step 4: Fix relative imports in the moved files** (they moved one level deeper: `tools/` → `reduce/`, so `.` and `..` references shift).

In `hypertools/reduce/reduce.py`, change the import block:
```python
from .._shared.helpers import *
from .format_data import format_data as formatter
```
to:
```python
from .._shared.helpers import *
from ..tools.format_data import format_data as formatter
```
(`.._shared.helpers` is unchanged — still one level up from the package root; `format_data` now lives in `..tools`.)

In `hypertools/reduce/describe.py`, change:
```python
from .reduce import reduce as reducer
from .format_data import format_data as formatter
```
to:
```python
from .reduce import reduce as reducer
from ..tools.format_data import format_data as formatter
```
(`.reduce` now correctly refers to `hypertools/reduce/reduce.py`.)

- [ ] **Step 5: Create `hypertools/reduce/__init__.py`:**

```python
from .reduce import reduce
from .describe import describe
```

- [ ] **Step 6: Recreate `hypertools/tools/reduce.py` as a shim** (re-export EVERYTHING the codebase imports from it — `reduce`, `models`, `reduce_list`, `_resolve_model`):

```python
# Moved to hypertools.reduce.reduce (HyperTools 2.0). Shim preserves the old path
# (core.model._build_registry imports `models` from here).
from ..reduce.reduce import *  # noqa: F401,F403
from ..reduce.reduce import reduce, models, reduce_list, _resolve_model  # noqa: F401
```

- [ ] **Step 7: Recreate `hypertools/tools/describe.py` as a shim:**

```python
# Moved to hypertools.reduce.describe (HyperTools 2.0). Shim preserves the old path.
from ..reduce.describe import *  # noqa: F401,F403
from ..reduce.describe import describe, get_corr, get_cdist  # noqa: F401
```

- [ ] **Step 8: Run tests + import + focused regression**

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/reduce/test_reduce_module.py tests/test_reduce.py tests/test_describe.py tests/core/test_model.py tests/test_apply_model.py -q -p no:cacheprovider
.venv/bin/python -c "import hypertools; from hypertools.tools.reduce import models; print('OK', len(models))"
```
Expected: all pass (existing `test_reduce.py`/`test_describe.py` green via shim; registry intact).

- [ ] **Step 9: Commit**

```bash
git add hypertools/reduce hypertools/tools/reduce.py hypertools/tools/describe.py tests/reduce
git commit -m "$(printf 'refactor(reduce): re-home reduce+describe into hypertools.reduce; shim tools\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: align/common.py — Aligner base + pad/trim_and_pad

**Files:**
- Create: `hypertools/align/__init__.py` (minimal for now), `hypertools/align/common.py`
- Test: `tests/align/__init__.py`, `tests/align/test_align_base.py`

**Interfaces:**
- Produces: `hypertools.align.common.Aligner(BaseEstimator)` (kwargs-triple: `data`/`fitter`/`transformer`/`required`; `fit` unstacks+trim_and_pads `self.data`, calls `fitter`, sets returned dict as attrs; `transform` re-unstacks+trim_and_pads and calls `transformer`; `fit_transform`), `pad(x, c=None)`, `trim_and_pad(data)`. Consumed by every align child (Tasks 3–5) and the dispatcher (Task 6).

- [ ] **Step 1: Write the failing test** — create `tests/align/__init__.py` (empty) and `tests/align/test_align_base.py`:

```python
import numpy as np
import pandas as pd
from hypertools.align.common import Aligner, pad, trim_and_pad


def test_pad_widens_to_c_columns():
    df = pd.DataFrame(np.ones((3, 2)))
    out = pad(df, c=5)
    assert out.shape == (3, 5)
    assert np.allclose(out.iloc[:, :2].to_numpy(), 1.0)
    assert np.allclose(out.iloc[:, 2:].to_numpy(), 0.0)


def test_trim_and_pad_aligns_shapes():
    a = pd.DataFrame(np.random.RandomState(0).rand(5, 3))
    b = pd.DataFrame(np.random.RandomState(1).rand(4, 2))
    out = trim_and_pad([a, b])
    assert out[0].shape[1] == out[1].shape[1] == 3
    assert out[0].shape[0] == out[1].shape[0] == 4  # common rows


def test_aligner_null_fit_transform_returns_data():
    # a null aligner (no fitter/transformer) returns its (trim_and_padded) data
    a = pd.DataFrame(np.random.RandomState(0).rand(6, 3))
    m = Aligner(fitter=lambda data, **k: {}, transformer=lambda data, **k: data,
                required=[], data=None)
    out = m.fit_transform([a, a])
    assert isinstance(out, list) and out[0].shape == (6, 3)
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/align/test_align_base.py -q -p no:cacheprovider`

- [ ] **Step 3: Create `hypertools/align/common.py`** — port `pad`/`trim_and_pad`/`Aligner` from `jeremy/master:hypertools/align/common.py` verbatim (it is dw-based and correct). Full content:

```python
"""Base class + helpers for hypertools aligners (scikit-learn compatible).

An Aligner wraps a (fitter, transformer, required-params) triple operating on
a *list* of DataFrames: `fit` unstacks the stored data into that list, trims to
common rows and pads to common columns, runs the fitter, and stores the returned
dict as attributes; `transform` re-derives the list and runs the transformer with
those params. Child classes (HyperAlign, Procrustes, SharedResponseModel, ...)
supply the three pieces plus their defaults.
"""
import datawrangler as dw
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


def pad(x, c=None):
    """Horizontally zero-pad a DataFrame (or list of DataFrames) to `c` columns."""
    if type(x) is list:
        if c is None:
            c = np.max([d.shape[1] for d in x])
        return [pad(d, c) for d in x]
    if c is None:
        return x
    y = np.zeros([x.shape[0], c])
    n = np.min([c, x.shape[1]])
    y[:, :n] = x.iloc[:, :n]
    return pd.DataFrame(data=y, index=x.index.copy())


def trim_and_pad(data):
    """Select the common rows across a list of DataFrames and pad to common columns."""
    if len(data) == 0:
        return data
    if type(data) is not list:
        data = [data]
    rows = set(data[0].index.values)
    for d in data[1:]:
        rows = rows.intersection(set(d.index.values))
    c = np.max([x.shape[1] for x in data])
    rows = list(rows)
    return [pad(d.loc[rows], c) for d in data]


class Aligner(BaseEstimator):
    def __init__(self, **kwargs):
        self.data = kwargs.pop('data', None)
        self.fitter = kwargs.pop('fitter', None)
        self.transformer = kwargs.pop('transformer', None)
        self.required = kwargs.pop('required', [])
        self.kwargs = kwargs

    def fit(self, data):
        assert data is not None, ValueError('cannot align empty dataset')
        self.data = data
        if self.fitter is None:
            return
        data = trim_and_pad(dw.unstack(self.data))
        params = self.fitter(data, **self.kwargs)
        assert type(params) is dict, ValueError('fit function must return a dictionary')
        assert all([r in params.keys() for r in self.required]), \
            ValueError('one or more required fields not returned')
        for k, v in params.items():
            setattr(self, k, v)

    def transform(self, *_):
        if self.data is None:
            raise NotFittedError('must fit aligner before transforming data')
        for r in self.required:
            if not hasattr(self, r):
                raise NotFittedError(f'missing fitted attribute: {r}')
        if self.transformer is None:
            return self.data
        data = trim_and_pad(dw.unstack(self.data))
        required_params = {r: getattr(self, r) for r in self.required}
        return self.transformer(data, **dw.core.update_dict(required_params, self.kwargs))

    def fit_transform(self, data):
        self.fit(data)
        return self.transform()
```

- [ ] **Step 4: Create `hypertools/align/__init__.py`** (minimal; children added in later tasks):

```python
from .common import Aligner, pad, trim_and_pad
```

- [ ] **Step 5: Run the base tests — expect pass. Commit:**

```bash
git add hypertools/align/__init__.py hypertools/align/common.py tests/align
git commit -m "$(printf 'feat(align): add Aligner base + pad/trim_and_pad helpers\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 3: align/procrustes.py + align/null.py (Procrustes child + NullAlign)

**Files:**
- Create: `hypertools/align/procrustes.py`, `hypertools/align/null.py`
- Modify: `hypertools/align/__init__.py`, `hypertools/tools/procrustes.py` (→ shim)
- Test: `tests/align/test_procrustes_child.py`

**Interfaces:**
- Produces: `hypertools.align.procrustes.procrustes(source, target, scaling=True, reflection=True, reduction=False, oblique=False, oblique_rcond=-1, format_data=True)` (the dev-2.0 function, numpy-in/out — the classic API), `hypertools.align.procrustes.Procrustes(Aligner)` (DataFrame-list child, `target=`/`scaling=`/… kwargs, `required=['proj','index']`), `hypertools.align.null.NullAlign(Aligner)`. `tools.procrustes` re-exports the `procrustes` function (keeps `tests/test_procrustes.py` green).

- [ ] **Step 1: Write the failing test** — create `tests/align/test_procrustes_child.py`:

```python
import numpy as np
import pandas as pd
from hypertools.align.procrustes import procrustes, Procrustes
from hypertools.align.null import NullAlign


def test_procrustes_function_recovers_rotation():
    rng = np.random.RandomState(0)
    target = rng.rand(20, 3)
    rot, _ = np.linalg.qr(rng.rand(3, 3))
    source = target @ rot
    out = procrustes(source, target)
    assert np.allclose(out, target, atol=1e-6)


def test_procrustes_child_aligns_list_of_dataframes():
    rng = np.random.RandomState(1)
    target = rng.rand(15, 3)
    rot, _ = np.linalg.qr(rng.rand(3, 3))
    a = pd.DataFrame(target)
    b = pd.DataFrame(target @ rot)
    out = Procrustes().fit_transform([a, b])
    assert np.allclose(np.asarray(out[0]), np.asarray(out[1]), atol=1e-6)


def test_null_align_returns_input_rows_cols():
    a = pd.DataFrame(np.random.RandomState(2).rand(8, 4))
    out = NullAlign().fit_transform([a, a])
    assert out[0].shape == (8, 4)
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/align/test_procrustes_child.py -q -p no:cacheprovider`

- [ ] **Step 3: Create `hypertools/align/procrustes.py`.** Two parts:

(a) The **classic `procrustes` function** — copy `jeremy/master`-equivalent = the dev-2.0 function verbatim from the current `hypertools/tools/procrustes.py` (the `procrustes(source, target, scaling=..., ...)` with nested `fit`/`transform`), changing only its import `from .format_data import format_data as formatter` → `from ..tools.format_data import format_data as formatter`.

(b) A **`Procrustes(Aligner)` child** whose fitter/transformer operate on a list of DataFrames, ported from `jeremy/master:hypertools/align/procrustes.py` (`align`/`xform`/`fitter`/`transformer`/`Procrustes`), with these ADAPTATIONS (our `core` has no `eval_dict`/per-class config):
  - Drop `from ..core import get_default_options, eval_dict`. Give `Procrustes.__init__` inline defaults instead:
    ```python
    class Procrustes(Aligner):
        def __init__(self, target=None, scaling=True, reflection=True,
                     reduction=False, oblique=False, oblique_rcond=-1, index=0, **kwargs):
            required = ['proj', 'index']
            super().__init__(required=required, fitter=fitter, transformer=transformer,
                             data=None, target=target, scaling=scaling, reflection=reflection,
                             reduction=reduction, oblique=oblique, oblique_rcond=oblique_rcond,
                             index=index, **kwargs)
    ```
    (`self.kwargs` then carries target/scaling/…; `fitter`/`transformer` read them via `**kwargs`, and `required=['proj','index']` — the fork's `align`/`xform`/`fitter`/`transformer` functions are copied as-is except `np.asmatrix` is deprecated: replace `d = np.asmatrix(data); res = (d * proj).A` in `xform` with `res = np.asarray(data) @ np.asarray(proj)`.)
  - Keep `xform` returning `pd.DataFrame(data=res, index=data.index)`.

- [ ] **Step 4: Create `hypertools/align/null.py`** — port `jeremy/master:hypertools/align/null.py`, dropping the `eval_dict`/config read:

```python
from .common import Aligner


def fitter(*args, **kwargs):
    return {}


def transformer(data, **kwargs):
    return data


class NullAlign(Aligner):
    """Returns the (trimmed + padded) dataset unchanged."""
    def __init__(self, **kwargs):
        super().__init__(required=[], fitter=fitter, transformer=transformer,
                         data=None, **kwargs)
```

- [ ] **Step 5: Update `hypertools/align/__init__.py`:**

```python
from .common import Aligner, pad, trim_and_pad
from .procrustes import procrustes, Procrustes
from .null import NullAlign
```

- [ ] **Step 6: Recreate `hypertools/tools/procrustes.py` as a shim:**

```python
# Moved to hypertools.align.procrustes (HyperTools 2.0). Shim preserves the old path.
from ..align.procrustes import procrustes  # noqa: F401
```

- [ ] **Step 7: Run — expect pass** (fix ports until green; do not weaken tolerances):

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/align/test_procrustes_child.py tests/test_procrustes.py -q -p no:cacheprovider
```

- [ ] **Step 8: Commit**

```bash
git add hypertools/align tests/align hypertools/tools/procrustes.py
git commit -m "$(printf 'feat(align): add procrustes fn + Procrustes/NullAlign children; shim tools/procrustes\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 4: align/hyperalign.py — HyperAlign (dev-2.0 rescale algorithm)

**Files:**
- Create: `hypertools/align/hyperalign.py`
- Modify: `hypertools/align/__init__.py`
- Test: `tests/align/test_hyperalign.py`

**Interfaces:**
- Produces: `hypertools.align.hyperalign.HyperAlign(Aligner)` (`n_iter=10` kwarg; `required=['proj']`; fitter runs dev-2.0's rescaled hyperalignment passes and stores per-dataset fitted `Procrustes` projectors; transformer applies them). Consumed by the dispatcher (Task 6). **Acceptance: `tests/test_align.py::test_hyper` (rtol=1) and, through the Task 6 shim, `test_procrustes`/`test_align_geo` (tight `allclose`).**

**Why this is not a blind fork port:** the fork's `HyperAlign.fitter` omits per-pass rescaling; dev-2.0's `align` (`hypertools/tools/align.py`, `_hyperalign_pass` + the `n_iter` rescale loop) fixes procrustes' scaling collapse. Port **dev-2.0's algorithm** into the fitter.

- [ ] **Step 1: Write the failing test** — create `tests/align/test_hyperalign.py`:

```python
import numpy as np
import pandas as pd
from hypertools.align.hyperalign import HyperAlign


def test_hyperalign_recovers_rotation_of_two_datasets():
    rng = np.random.RandomState(0)
    base = rng.rand(20, 4)
    rot, _ = np.linalg.qr(rng.rand(4, 4))
    a = pd.DataFrame(base)
    b = pd.DataFrame(base @ rot)
    out = HyperAlign(n_iter=10).fit_transform([a, b])
    # aligned datasets should be close (hyperalignment of a pure rotation)
    assert np.corrcoef(np.asarray(out[0]).ravel(),
                       np.asarray(out[1]).ravel())[0, 1] > 0.95


def test_hyperalign_preserves_scale_across_iterations():
    rng = np.random.RandomState(1)
    data = [pd.DataFrame(rng.rand(15, 3)) for _ in range(3)]
    out = HyperAlign(n_iter=10).fit_transform(data)
    norms = [np.linalg.norm(np.asarray(o)) for o in out]
    # rescaling keeps magnitudes on the original order (not collapsed to ~0)
    assert min(norms) > 1e-3
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/align/test_hyperalign.py -q -p no:cacheprovider`

- [ ] **Step 3: Create `hypertools/align/hyperalign.py`.** Structure it as an `Aligner` child whose fitter runs dev-2.0's rescaled passes (adapted from `hypertools/tools/align.py`'s `_hyperalign_pass` + the `n_iter` loop, lines 110–128 & 143–174) using the `Procrustes` **child** (which stores projections), then stores the final per-dataset projectors so `transform` can re-apply them:

```python
import numpy as np

from .common import Aligner
from .procrustes import Procrustes


def _rescale(t, mean_norm):
    norm = np.linalg.norm(t)
    return t * (mean_norm / norm) if norm > 0 else t


def _one_pass(m):
    """One full classic hyperalignment pass (Haxby et al., 2011) with per-pass
    rescale to the datasets' mean Frobenius norm (prevents geometric collapse)."""
    mean_norm = np.mean([np.linalg.norm(np.asarray(x)) for x in m])
    m = [np.asarray(x, dtype=float) for x in m]

    # STEP 1: initial (sequential) template
    template = np.copy(m[0])
    for x in range(1, len(m)):
        template += _procrustes_np(m[x], template / (x + 1))
    template = _rescale(template / len(m), mean_norm)

    # STEP 2: refined template
    template2 = np.zeros_like(template)
    for x in range(len(m)):
        template2 += _procrustes_np(m[x], template)
    template2 = _rescale(template2 / len(m), mean_norm)

    # STEP 3: align every dataset to the refined template
    return [_procrustes_np(m[x], template2) for x in range(len(m))], template2


def _procrustes_np(source, target):
    """Numpy Procrustes projection of `source` onto `target` (scaling+reflection),
    reusing the align() primitive so behavior matches the Procrustes child."""
    from .procrustes import align as _proc_align
    proj = _proc_align(np.asarray(source), np.asarray(target))
    return np.asarray(source) @ np.asarray(proj)


def fitter(data, n_iter=10, **kwargs):
    assert type(data) is list, 'data must be a list'
    n = len(data)
    if n <= 1 or n_iter == 0:
        # identity projections
        return {'proj': [np.eye(d.shape[1]) for d in data]}

    m = [np.asarray(d, dtype=float) for d in data]
    orig_norm = np.mean([np.linalg.norm(x) for x in m])
    aligned = m
    template2 = None
    for _ in range(max(1, int(n_iter))):
        aligned, template2 = _one_pass(aligned)
        cur_norm = np.mean([np.linalg.norm(np.asarray(a)) for a in aligned])
        if cur_norm > 0:
            aligned = [np.asarray(a) * (orig_norm / cur_norm) for a in aligned]

    # final projections: map each ORIGINAL dataset onto the converged template
    from .procrustes import align as _proc_align
    proj = [_proc_align(np.asarray(d), np.asarray(template2)) for d in data]
    return {'proj': proj}


def transformer(data, **kwargs):
    proj = kwargs['proj']
    import pandas as pd
    return [pd.DataFrame(np.asarray(d) @ np.asarray(p), index=d.index)
            for d, p in zip(data, proj)]


class HyperAlign(Aligner):
    """Hyperalignment (Haxby et al., 2011) with per-pass rescaling."""
    def __init__(self, n_iter=10, **kwargs):
        assert n_iter >= 0, 'n_iter must be non-negative'
        super().__init__(required=['proj'], fitter=fitter, transformer=transformer,
                         data=None, n_iter=n_iter, **kwargs)
```

> **NOTE for the executor:** the two tests above are a floor, not the real gate. The REAL acceptance gate is `tests/test_align.py` passing through the Task 6 shim (`test_procrustes` uses default align→HyperAlign and asserts tight `np.allclose`). If the projection-based `transform` cannot hit `test_procrustes`'s tolerance, fall back to the dev-2.0 structure where the fitter stores the converged **aligned data + template** and the transformer returns the stored aligned data directly for the fit set (dev-2.0's `align` returns aligned data, not re-projected data). Document whichever structure you land on in the task report. Do NOT weaken `test_align.py`.

- [ ] **Step 4: Update `hypertools/align/__init__.py`:**

```python
from .common import Aligner, pad, trim_and_pad
from .procrustes import procrustes, Procrustes
from .hyperalign import HyperAlign
from .null import NullAlign
```

- [ ] **Step 5: Run — expect pass:**

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/align/test_hyperalign.py -q -p no:cacheprovider
```

- [ ] **Step 6: Commit**

```bash
git add hypertools/align/hyperalign.py hypertools/align/__init__.py tests/align/test_hyperalign.py
git commit -m "$(printf 'feat(align): add HyperAlign (dev-2.0 rescaled hyperalignment algorithm)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 5: align/srm.py — SharedResponseModel + DeterministicSharedResponseModel

**Files:**
- Create: `hypertools/align/srm.py`
- Modify: `hypertools/align/__init__.py`
- Test: `tests/align/test_srm.py`

**Interfaces:**
- Produces: `hypertools.align.srm.SharedResponseModel(Aligner)`, `hypertools.align.srm.DeterministicSharedResponseModel(Aligner)` — adapters over `hypertools.external.brainiak.{SRM, DetSRM}`. `features=` kwarg (default: min columns). **RSRM is NOT carried** (external.brainiak has no RSRM).

- [ ] **Step 1: Write the failing test** — create `tests/align/test_srm.py`:

```python
import numpy as np
import pandas as pd
from hypertools.align.srm import SharedResponseModel, DeterministicSharedResponseModel


def _rotated_pair(seed=0, k=4):
    rng = np.random.RandomState(seed)
    base = rng.rand(30, k)
    rot, _ = np.linalg.qr(rng.rand(k, k))
    return [pd.DataFrame(base), pd.DataFrame(base @ rot)]


def test_srm_aligns_to_shared_space():
    out = SharedResponseModel(features=3).fit_transform(_rotated_pair())
    assert isinstance(out, list) and len(out) == 2
    # shared responses should be correlated across the two views
    assert np.corrcoef(np.asarray(out[0]).ravel(),
                       np.asarray(out[1]).ravel())[0, 1] > 0.5


def test_detsrm_runs_and_shapes():
    out = DeterministicSharedResponseModel(features=3).fit_transform(_rotated_pair(1))
    assert len(out) == 2 and np.asarray(out[0]).shape[1] == 3


def test_rsrm_not_exported():
    import hypertools.align.srm as srm
    assert not hasattr(srm, 'RobustSharedResponseModel')
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/align/test_srm.py -q -p no:cacheprovider`

- [ ] **Step 3: Create `hypertools/align/srm.py`** — port `jeremy/master:hypertools/align/srm.py`'s `fitter`/`transformer`/`srm_fitter`/`detsrm_fitter` and the `SharedResponseModel`/`DeterministicSharedResponseModel` classes, with these ADAPTATIONS:
  - Import only `from ..external.brainiak import SRM, DetSRM` (NO `RSRM`). Delete `rsrm_fitter` and the `RobustSharedResponseModel` class entirely.
  - Drop `from ..core import get_default_options, eval_dict`; give each class inline defaults:
    ```python
    class SharedResponseModel(Aligner):
        def __init__(self, features=None, **kwargs):
            super().__init__(required=['model', 'features', 'indices'],
                             fitter=srm_fitter, transformer=transformer,
                             data=None, features=features, **kwargs)
    ```
    (and `DeterministicSharedResponseModel` identically with `detsrm_fitter`.)
  - Keep the fitter/transformer bodies as-is (they call `align_type(features=features)`, `.fit([d.values.T ...])`, and rebuild DataFrames via the stored `indices`). Guard `features is None` → `np.min([d.shape[1] for d in data])`.

- [ ] **Step 4: Update `hypertools/align/__init__.py`:**

```python
from .common import Aligner, pad, trim_and_pad
from .procrustes import procrustes, Procrustes
from .hyperalign import HyperAlign
from .srm import SharedResponseModel, DeterministicSharedResponseModel
from .null import NullAlign
```

- [ ] **Step 5: Run — expect pass:**

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/align/test_srm.py -q -p no:cacheprovider
```

- [ ] **Step 6: Commit**

```bash
git add hypertools/align/srm.py hypertools/align/__init__.py tests/align/test_srm.py
git commit -m "$(printf 'feat(align): add SRM + DetSRM adapters over external.brainiak (RSRM not carried)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 6: align/align.py — dispatcher + tools/align.py compat shim

**Files:**
- Create: `hypertools/align/align.py`
- Modify: `hypertools/align/__init__.py`, `hypertools/tools/align.py` (→ compat shim)
- Test: `tests/align/test_align_dispatcher.py`

**Interfaces:**
- Produces: `hypertools.align.align.align(data, model='HyperAlign', **kwargs)` (`@dw.decorate.funnel`; resolves via `unpack_model` against `[HyperAlign, SharedResponseModel, DeterministicSharedResponseModel, Procrustes, NullAlign]`; applies the resolved `Aligner` via `fit_transform` directly; returns DataFrame/list). `tools.align.align(data, align='hyper', n_iter=10, format_data=True)` is a **compat shim** mapping the classic API onto the dispatcher (see Step 4). **Acceptance: `tests/test_align.py` (all cases) green.**

- [ ] **Step 1: Write the failing test** — create `tests/align/test_align_dispatcher.py`:

```python
import numpy as np
import pandas as pd
from hypertools.align.align import align


def test_dispatcher_hyperalign_by_name():
    rng = np.random.RandomState(0)
    base = rng.rand(20, 3)
    rot, _ = np.linalg.qr(rng.rand(3, 3))
    out = align([pd.DataFrame(base), pd.DataFrame(base @ rot)], model='HyperAlign')
    assert isinstance(out, list) and len(out) == 2


def test_dispatcher_null_by_name():
    a = pd.DataFrame(np.random.RandomState(1).rand(10, 4))
    out = align([a, a], model='NullAlign')
    assert out[0].shape == (10, 4)


def test_dispatcher_accepts_arrays():
    rng = np.random.RandomState(2)
    base = rng.rand(12, 3)
    out = align([base, base.copy()], model='Procrustes')
    assert np.allclose(np.asarray(out[0]), np.asarray(out[1]), atol=1e-6)
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/align/test_align_dispatcher.py -q -p no:cacheprovider`

- [ ] **Step 3: Create `hypertools/align/align.py`** (dispatcher — applies the resolved Aligner directly, mirroring `manip.py`; NOT via `core.apply_model`):

```python
"""hyp.align dispatcher: resolve an aligner spec and fit_transform it.

Wrapped by datawrangler's funnel so any input (array / DataFrame / list / text /
polars) arrives as DataFrame(s); the resolved Aligner (list-based, sklearn-
compatible) is applied directly. NOT routed through core.apply_model, whose
stack-and-fit-once recipe is wrong for aligning a *list* to a shared template.
"""
import datawrangler as dw

from .common import Aligner
from .hyperalign import HyperAlign
from .procrustes import Procrustes
from .srm import SharedResponseModel, DeterministicSharedResponseModel
from .null import NullAlign
from ..core.shared import unpack_model


ALIGNERS = [HyperAlign, SharedResponseModel, DeterministicSharedResponseModel,
            Procrustes, NullAlign]


@dw.decorate.funnel
def align(data, model='HyperAlign', **kwargs):
    resolved = unpack_model(model, valid=ALIGNERS, parent_class=Aligner)
    if isinstance(resolved, type):
        resolved = resolved(**kwargs)
    elif isinstance(resolved, dict):
        cls = resolved['model']
        resolved = cls(*resolved.get('args', []), **resolved.get('kwargs', {}))
    return resolved.fit_transform(data)
```

- [ ] **Step 4: Recreate `hypertools/tools/align.py` as a COMPAT shim** mapping the classic array/mode API onto the new dispatcher AND preserving dev-2.0's exact `test_align.py` behavior. Because `test_align.py` asserts tight `allclose` on the DEFAULT path and array output, the shim must (a) translate `align='hyper'`→`'HyperAlign'`, `'SRM'`→`'SharedResponseModel'`, (b) thread `n_iter`, (c) honor `align=None`→return data, `align=True`→ValueError, dict form, and (d) return a list of numpy arrays. Keep it thin — delegate the algorithm to the new dispatcher:

```python
# Classic array/mode align API (HyperTools <2.0). Thin compat wrapper over the
# new class-based dispatcher in hypertools.align.align.
import numpy as np
import warnings

from ..align.align import align as _align_dispatch

_ALIAS = {'hyper': 'HyperAlign', 'HyperAlign': 'HyperAlign',
          'SRM': 'SharedResponseModel',
          'SharedResponseModel': 'SharedResponseModel',
          'DetSRM': 'DeterministicSharedResponseModel',
          'Procrustes': 'Procrustes', 'NullAlign': 'NullAlign'}


def align(data, align='hyper', n_iter=10, format_data=True):
    if align is None:
        return data
    if align is True:
        raise ValueError("align=True was removed in hypertools 2.0; specify the "
                         "algorithm instead, e.g. align='hyper' or align='SRM'.")
    if isinstance(align, dict):
        params = dict(align.get('params', {}))
        model = align['model']
        if model is None:
            return data
        n_iter = params.get('n_iter', n_iter)
    else:
        model, params = align, {}
    model = _ALIAS.get(model, model)
    if model == 'HyperAlign':
        params.setdefault('n_iter', n_iter)
    out = _align_dispatch(data, model=model, **params)
    if not isinstance(out, list):
        out = [out]
    return [np.asarray(o) for o in out]
```

- [ ] **Step 5: Update `hypertools/align/__init__.py`** to export the dispatcher:

```python
from .common import Aligner, pad, trim_and_pad
from .procrustes import procrustes, Procrustes
from .hyperalign import HyperAlign
from .srm import SharedResponseModel, DeterministicSharedResponseModel
from .null import NullAlign
from .align import align
```

- [ ] **Step 6: Run the dispatcher tests + the classic acceptance gate:**

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/align/ tests/test_align.py tests/test_procrustes.py -q -p no:cacheprovider
```
Expected: all green. **If `test_align.py::test_procrustes`/`test_align_geo` (tight `allclose`) fail, fix `HyperAlign` (Task 4) — do NOT weaken the tests.** `test_align_geo` passes a geo object; the funnel + format_data path must accept it (geo still exists until Plan 7). If the funnel rejects a geo, the compat shim may call `formatter(data, ppca=True)` first (import `from .format_data import format_data as formatter`) before delegating — mirror dev-2.0's `format_data=True` behavior.

- [ ] **Step 7: Commit**

```bash
git add hypertools/align/align.py hypertools/align/__init__.py hypertools/tools/align.py tests/align/test_align_dispatcher.py
git commit -m "$(printf 'feat(align): add align dispatcher + classic tools/align compat shim\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 7: cluster/ package (move cluster; shim tools/) + wire public API + plan-close regression

**Files:**
- Create: `hypertools/cluster/__init__.py`, `hypertools/cluster/cluster.py`
- Modify: `hypertools/tools/cluster.py` (→ shim), `hypertools/__init__.py`
- Test: `tests/cluster/__init__.py`, `tests/cluster/test_cluster_module.py`

**Interfaces:**
- Produces: `hypertools.cluster.cluster.cluster(x, cluster='KMeans', n_clusters=3, format_data=True)` + `models`/`mixture_models` dicts (dev-2.0 logic: hard labels list; soft proportions for GaussianMixture/BayesianGaussianMixture/LDA/NMF). `tools.cluster` re-exports these (registry `from ..tools.cluster import models, mixture_models` keeps resolving). `hypertools.__init__` continues to export classic `reduce`/`align`/`cluster`/`describe`/`normalize` and additionally surfaces the new module packages.

- [ ] **Step 1: Write the failing test** — create `tests/cluster/__init__.py` (empty) and `tests/cluster/test_cluster_module.py`:

```python
import numpy as np


def test_cluster_new_path_hard_labels():
    from hypertools.cluster.cluster import cluster
    rng = np.random.RandomState(0)
    data = np.vstack([rng.randn(50, 3), rng.randn(50, 3) + 100])
    labels = cluster(data, n_clusters=2)
    assert type(labels) is list and len(set(labels)) == 2


def test_cluster_shim_is_same_function():
    from hypertools.cluster.cluster import cluster as new_c
    from hypertools.tools.cluster import cluster as old_c
    assert new_c is old_c


def test_cluster_registry_dicts_via_tools():
    from hypertools.tools.cluster import models, mixture_models
    assert 'KMeans' in models and 'GaussianMixture' in mixture_models


def test_cluster_soft_mixture_proportions():
    from hypertools.cluster.cluster import cluster
    rng = np.random.RandomState(1)
    data = np.vstack([rng.randn(40, 3), rng.randn(40, 3) + 50])
    props = cluster(data, cluster='GaussianMixture', n_clusters=2)
    assert props.shape == (80, 2) and np.allclose(props.sum(axis=1), 1)
```

- [ ] **Step 2: Run — expect failure.**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/cluster/test_cluster_module.py -q -p no:cacheprovider`

- [ ] **Step 3: Move the file + fix imports**

```bash
cd /Users/jmanning/hypertools
git mv hypertools/tools/cluster.py hypertools/cluster/cluster.py
```
In `hypertools/cluster/cluster.py`, change:
```python
from .._shared.helpers import *
from .format_data import format_data as formatter
```
to:
```python
from .._shared.helpers import *
from ..tools.format_data import format_data as formatter
```

- [ ] **Step 4: Create `hypertools/cluster/__init__.py`:**

```python
from .cluster import cluster, models, mixture_models
```

- [ ] **Step 5: Recreate `hypertools/tools/cluster.py` as a shim** (registry imports `models`, `mixture_models` from here):

```python
# Moved to hypertools.cluster.cluster (HyperTools 2.0). Shim preserves the old path
# (core.model._build_registry imports models/mixture_models from here).
from ..cluster.cluster import *  # noqa: F401,F403
from ..cluster.cluster import cluster, models, mixture_models  # noqa: F401
```

- [ ] **Step 6: Surface the new module packages in `hypertools/__init__.py`.** Inspect the current file first (`grep -n 'reduce\|align\|cluster\|describe\|normalize\|manip' hypertools/__init__.py`). The classic names must remain exported unchanged. Add the new package submodules as attributes so `hyp.reduce`/`hyp.align`/`hyp.cluster` (module) and the classic callables coexist per the current pattern used for `manip` in Plan 3. Do NOT remove any existing export. Concretely, ensure these remain importable and unchanged: `hyp.reduce` (callable), `hyp.align` (callable), `hyp.cluster` (callable), `hyp.describe`, `hyp.normalize`, `hyp.manip`. (The classic callables already resolve through the `tools/` shims, so no export change is strictly required — verify with the API test below and only edit if something regressed.)

- [ ] **Step 7: Run the module tests + classic acceptance gate + public-API smoke:**

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/cluster/ tests/test_cluster.py -q -p no:cacheprovider
.venv/bin/python -c "import hypertools as hyp; import numpy as np; \
d=[np.random.rand(10,5) for _ in range(2)]; \
print('reduce', np.asarray(hyp.reduce(d, ndims=2)[0]).shape); \
print('cluster', type(hyp.cluster(np.random.rand(20,3), n_clusters=2)).__name__); \
print('align', len(hyp.align(d)))"
```

- [ ] **Step 8: PLAN-CLOSE full-suite regression** (run in background; ~13 min):

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest -q -p no:cacheprovider
```
Expected: **293 + new (reduce/align/cluster module tests) passed**, no regressions, exit 0.

- [ ] **Step 9: Commit**

```bash
git add hypertools/cluster tests/cluster hypertools/tools/cluster.py hypertools/__init__.py
git commit -m "$(printf 'refactor(cluster): re-home cluster into hypertools.cluster; shim tools; close Plan 4\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Self-Review

**1. Spec coverage:** reduce (Reducer/apply_model registry, describe fidelity, lazy UMAP already in `_resolve_model`, PPCA-fill via `format_data(ppca=True)`) → Task 1; align (Aligner base + pad/trim_and_pad → T2; Procrustes + NullAlign → T3; HyperAlign w/ dev-2.0 rescale → T4; SRM/DetSRM adapters → T5; dispatcher + classic compat → T6) → Tasks 2–6; cluster (hard labels + soft mixtures, `cluster={'model','n_clusters'}`) → Task 7. Classic `hyp.reduce/align/cluster/describe/normalize` preserved via `tools/` shims (strangler). RSRM explicitly documented as not-carried (external.brainiak lacks it) — a tracked gap, not a silent drop. Reduce/cluster deliberately omit a hypertools base class (no custom children; sklearn estimators resolve eval-free via `core.apply_model`) — matches the fork and honors "single source of truth / YAGNI"; flagged as a reviewable deviation from the spec's literal §3 layout.

**2. Placeholder scan:** every step has concrete code or an exact port instruction + specific adaptations (drop `eval_dict`/config reads → inline defaults; drop `RSRM`; `np.asmatrix`→`np.asarray @`); no TBD/"handle errors".

**3. Type consistency:** `Aligner(fitter, transformer, required, data)` triple consistent across T2–T6; `unpack_model(model, valid=[...], parent_class=Aligner)` matches `core.shared.unpack_model`; dispatcher `align(data, model=...)` signature consistent with `manip.py`; `models`/`mixture_models` re-exported so `core.model._build_registry` imports keep resolving.

## Execution Handoff

After Plan 4, Plan 5 (io: load/sources/streaming/save) proceeds; the reduce/align/cluster module surface feeds Plan 6 (plot + colors) where the weights-trajectory recipe (gaussian smooth → SRM n_iter=20 → smooth → UMAP) exercises `align.SharedResponseModel` + the owed gaussian `Smooth` mode.
