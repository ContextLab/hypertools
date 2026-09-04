# Plan 4, Task 1 "Native palette-from-image" — interception audit

**Date:** 2026-08-01
**Plan:** `docs/superpowers/plans/2026-07-28-hypertools-1.1-examples-and-tutorials.md`, Task 1 (starts line 169)
**Base commit:** `065c841e` (dev-1.0, clean)
**Worktree used:** `/tmp/palette_audit` (disposable, removed after the audit)
**Python:** `/Users/jmanning/hypertools/.venv/bin/python` throughout

Every number in this document came from a command that was actually run. Nothing
is projected.

---

## 1. Verdict: "ONE interception point" is **FALSE**

The plan states (line ~183):

> Because `_continuous_palette` (`colors.py:269`) delegates to `_get_palette`, and
> `mat2colors` (`colors.py:24`), `get_palette_colors` and `continuous_colormap` all
> route through those two functions, **one interception makes an image palette work
> on every path** — categorical hue, continuous hue, matrix hue, the matplotlib
> colorbar and the plotly colorbar — with no per-call-site change.

That claim is false, and the failure is **broader than the review reported**. The
review found categorical hue broken. In fact **every `hyp.plot(..., palette='image:…')`
call is broken on the matplotlib backend — including continuous hue, matrix hue, and
no hue at all** — because `sns.set_palette` runs unconditionally on every matplotlib
plot call and is handed the raw palette string.

### 1a. The paths that DO route through `_get_palette` (the plan is right about these)

| Consumer | Location | Route |
|-|-|-|
| `mat2colors` categorical | `colors.py:106` | `_get_palette` |
| `mat2colors` continuous | `colors.py:118` | `_continuous_palette` → `_get_palette` |
| `mat2colors` matrix | `colors.py:158` | `_get_palette` |
| `get_palette_colors` | `colors.py:246` | `_get_palette` |
| `continuous_colormap` | `colors.py:259` | `_continuous_palette` → `_get_palette` |
| MultiIndex colors | `multiindex.py:151` | `get_palette_colors` |
| plotly colorbar | `plotly_backend.py:2468` | `continuous_colormap` |
| mpl colorbar | `plot.py:5383` | `continuous_colormap` |
| partially-labeled hue | `plot.py:4033` | `get_palette_colors` |
| legend colors | `plot.py:5317` | `get_palette_colors` |

### 1b. The SECOND path the plan misses — `_seaborn_palette_arg`, and five raw seaborn call sites

`_seaborn_palette_arg` is defined at **`hypertools/plot/plot.py:113`**. Its entire
body for a string palette is `return palette` — the string goes to seaborn
**untouched**, never reaching `colors._get_palette`:

```python
# hypertools/plot/plot.py:113-124  (current, unpatched)
def _seaborn_palette_arg(palette, n_colors):
    """`palette` in a form seaborn's `color_palette`/`set_palette` accept.
    ...
    """
    from matplotlib.colors import Colormap
    if isinstance(palette, Colormap):
        return [tuple(c) for c in get_palette_colors(palette, n_colors)]
    return palette          # <-- 'image:<path>' leaves here unchanged
```

All five of its call sites hand the result straight to seaborn:

| # | Call site (true current line) | Seaborn call | When it runs |
|-|-|-|-|
| 1 | `plot.py:208-209` | `sns.color_palette(...)` | categorical hue on a **line** plot (`_categorical_color_label_maps`) |
| 2 | `plot.py:4118-4119` | `sns.color_palette(...)` | nested-list multilevel styling |
| 3 | `plot.py:4657-4658` | `sns.color_palette(...)` | `_resolve_dataset_colors` (surface=/density=/morph) |
| 4 | `plot.py:4767-4768` | `sns_local.color_palette(...)` | **plotly** backend per-dataset colors |
| 5 | **`plot.py:4825-4826`** | **`sns.set_palette(...)`** | **EVERY matplotlib plot call, unconditionally** |

Call site 5 is the fatal one. It sits in the `else` branch of the backend switch and
is not guarded by any hue condition, so **no matplotlib plot of any kind survives an
`image:` palette** under the plan as written.

**Measured evidence** — native traceback from the plan's own prescribed test:

```
File "/tmp/palette_audit/hypertools/plot/plot.py", line 4825, in plot
File ".../seaborn/palettes.py", line 237, in color_palette
    raise ValueError(f"{palette!r} is not a valid palette name")
ValueError: 'image:/private/var/.../painting.png' is not a valid palette name
```

### 1c. Paths checked and cleared

- `hypertools/_shared/helpers.py:116` — `vals2colors` calls `sns.color_palette(cmap, res)`
  directly, but it is a **legacy re-export with zero internal callers**
  (`grep -rn vals2colors hypertools/` → only its own definition and the re-export at
  `colors.py:341`). Not reachable from `plot(palette=…)`. No change needed.
- `morph.py`, `trails.py`, `density.py`, `surface.py`, `animate.py`,
  `matplotlib_backend.py`, `backend.py` — no palette resolution (only two unrelated
  comment mentions of "palettegen" and "palette cycle").

**Conclusion: there are exactly TWO interception points required** —
`colors._get_palette` (`colors.py:287`) and `plot._seaborn_palette_arg`
(`plot.py:113`). Patching either alone leaves real user-facing calls broken.

### 1d. Drifted line citations in the plan

Every line number in Plan 4 Task 1 has drifted. Corrections:

| Plan says | Truth |
|-|-|
| `_get_palette` string branch at `colors.py:305-306` | correct (305-306) |
| `_continuous_palette` at `colors.py:269` | correct |
| `mat2colors` at `colors.py:24` | correct |
| `get_palette_colors` at `colors.py:227` | correct |
| `continuous_colormap` at `colors.py:250` | correct |
| `continuous_colormap` "ends at `colors.py:260`" | correct |
| short-list blending at `colors.py:323-331` | correct |
| **`plot()`'s `palette` docstring at `plot.py:807-820`** | **actually `plot.py:1066-1078`** |
| *(not mentioned at all)* | `_seaborn_palette_arg` at `plot.py:113`; `sns.set_palette` at `plot.py:4825` |

---

## 2. Reproduction of the failure (plan applied verbatim)

Method: the plan's Task 1 code blocks were extracted **programmatically and verbatim**
from the markdown (4 python blocks: the test file, the implementation, and the
before/after pair for the `_get_palette` string branch), then applied to
`/tmp/palette_audit` at `065c841e`. Import shadowing verified before running:

```
hypertools: /tmp/palette_audit/hypertools/__init__.py
colors:     /tmp/palette_audit/hypertools/plot/colors.py
has IMAGE_PALETTE_N: True
```

### 2a. The plan's prescribed test file

```
2 failed, 14 passed in 2.28s
```

(The plan's Step 5 predicts "**16 passed**".)

Both failures are seaborn rejecting the `image:` string, with this exact text:

```
ValueError: 'image:/private/var/folders/tp/qtzc39jx5w556wl5w3dj21wr0000gn/T/
pytest-of-jmanning/pytest-779/test_palette_string_colours_a_0/painting.png'
is not a valid palette name
```

raised at `seaborn/palettes.py:237`, reached from `hypertools/plot/plot.py:4825`.

| Failing test | Reached seaborn via |
|-|-|
| `test_palette_string_colours_a_categorical_hue` | `plot.py:4825` `sns.set_palette` |
| `test_plotly_backend_accepts_an_image_palette` | `plot.py:4825` `sns.set_palette` |

### 2b. The plan's own tests understate the damage — the real 6-scenario result is 0/6

The plan's suite only catches 2 failures because its "continuous hue" test calls
`continuous_colormap()` directly rather than `hyp.plot()`, and its "missing file"
test asserts an error anyway (`mat2colors` raises `FileNotFoundError` before
execution reaches line 4825). Running the maintainer's six scenarios as real
`hyp.plot` calls against the plan-as-written:

```
1. categorical hue / matplotlib  FAIL  ValueError: 'image:...' is not a valid palette name
2. categorical hue / plotly      FAIL  ValueError: 'image:...' is not a valid palette name
3. continuous hue                FAIL  ValueError: 'image:...' is not a valid palette name
4. matrix hue                    FAIL  ValueError: 'image:...' is not a valid palette name
5. direct palette= (no hue)      FAIL  ValueError: 'image:...' is not a valid palette name
6. MORE THAN SIX categories (9)  FAIL  ValueError: 'image:...' is not a valid palette name

0/6 scenarios pass
```

### 2c. A THIRD, independent defect in the plan's test file

`test_palette_string_colours_a_categorical_hue` can **never** pass as written, even
against a perfect implementation. It harvests colors from `ax.collections`:

```python
drawn = np.vstack([np.atleast_2d(c.get_facecolor())[:, :3]
                   for c in _ax(fig).collections
                   if len(np.atleast_2d(c.get_facecolor()))])
```

But a `fmt='.'` plot draws `Line2D` artists into `ax.lines`. The only collections on
a 3-D axes are pane/grid artists whose facecolor array is **empty**. Measured:

```
collections and their facecolor shapes:
   Line3DCollection (0, 4)      <- x6, all empty
LINE colors (the actual data):
   [0.863 0.078 0.078]          <- VIVID, correct
   [0.784 0.769 0.737]          <- BEIGE, correct
```

So the filter removes all six collections, the list is empty, and
`np.vstack([])` raises `ValueError: need at least one array to concatenate`. The
implementation was producing exactly the right colors; the assertion was looking in
the wrong place.

---

## 3. The fix

Two interception points, plus a dynamic color count.

### 3a. Policy for "image yields fewer distinct colors than requested"

**Chosen: deterministic interpolation via `sns.blend_palette`, with an explicit error
for the single-color image.** Rejected: cycling.

Justification:

1. **Cycling is a silent correctness failure.** Repeating anchors would give two
   different categories the *same* color, making the plot and its legend ambiguous.
   That is precisely the ambiguity `_get_palette` already refuses to permit — it
   raises for a short categorical list (`colors.py:332-335`) rather than reuse a color.
2. **Interpolation is already this module's established answer to "too few anchors."**
   `colors.py:323-331` blends short lists with `blend_palette` for the continuous
   path, documented in `_continuous_palette`'s own docstring as "seaborn
   ``blend_palette`` semantics; F02-006/F24-017". Reusing it keeps one mechanism in
   the module instead of introducing a second.
3. **Why not just raise, as a short user list does?** A user-supplied short list is
   the user's own choice and they can simply pass more colors. A caller *cannot* add
   colors to an image. Raising would make `palette='image:…'` fail on any two-tone
   image with an error the user has no way to act on.
4. **It is deterministic.** `image_palette` is seeded (`random_state=0`) and
   `blend_palette` is a pure function of the anchor list. Verified bit-identical
   across 5 repeated calls.
5. **The most salient anchor stays first**, so the "vivid subject leads" contract that
   justifies the whole feature survives interpolation. Verified at n = 2, 5, 9, 12.
6. **The one case interpolation cannot serve raises.** A solid-color image genuinely
   cannot yield 9 distinct colors; there is no honest interpolation. The error names
   the image and gives three fixes, matching the module's existing error style.

Measured:

```
2-color image,  2 requested ->  2 returned,  2 DISTINCT, first=[0.863 0.078 0.078]
2-color image,  5 requested ->  5 returned,  5 DISTINCT, first=[0.863 0.078 0.078]
2-color image,  9 requested ->  9 returned,  9 DISTINCT, first=[0.863 0.078 0.078]
2-color image, 12 requested -> 12 returned, 12 DISTINCT, first=[0.863 0.078 0.078]

single-color image, 5 requested ->
  ValueError: palette='image:.../one_color.png' yielded 1 color but 5 are
  required (one per category/component); that image has a single dominant
  color, so pass a more colorful image, an explicit list of colors, or a
  palette name
single-color image, 1 requested -> works: [0.784 0.784 0.784]

determinism (5 repeats, n=9): bit-identical -> True
```

### 3b. How many colors are extracted (the `IMAGE_PALETTE_N = 6` fix)

`_get_palette` already carries a `continuous` flag, so the resolver can pick the right
count with no signature change anywhere:

- **categorical / matrix** → extract exactly `n_colors`. k-means with k = the number
  of categories is the best k-color summary of that image, and it removes the 6-category
  cap entirely.
- **continuous** → extract `IMAGE_PALETTE_N` (6) anchors and let the existing
  short-list blending build the gradient. This preserves the plan's own (correct)
  reasoning: a continuous mapping asks for `n_bins` = 100 colors, and clustering an
  image into 100 groups is both slow and meaningless.

`IMAGE_PALETTE_N` is therefore **kept**, but demoted from "the count" to "the
continuous-anchor count". Cost measured at ~10-15 ms/call (200 px thumbnail), so no
caching is warranted:

```
k= 1: 10.1 ms/call    k= 6: 13.0 ms/call
k= 2: 10.8 ms/call    k= 9: 15.3 ms/call
                      k=20: 15.5 ms/call
```

### 3c. The patch

Everything in the plan's `image_palette` / `_image_pixels` / `IMAGE_PALETTE_PREFIX` /
`_ACHROMATIC_EPS` block is **kept unchanged** — all 12 extraction tests passed against
it. Only the interception changes.

#### Patch 1 of 3 — `hypertools/plot/colors.py`: new resolver helper

Insert immediately **before** `def _get_palette` (`colors.py:287`):

```python
def _image_palette_list(source, n_colors, sns, continuous):
    """Colors for a `palette='image:<path>'` string, as a list `_get_palette`
    can then handle exactly like any other color list.

    How many colors are EXTRACTED depends on the mapping. A categorical or
    matrix mapping needs one color per category, so exactly `n_colors`
    anchors are pulled: k-means with k = the number of categories is the
    best k-color summary of that image, and extracting a FIXED count would
    instead cap every plot at that many categories. A CONTINUOUS mapping
    asks for `n_bins` (100) colors, and clustering an image into 100 groups
    is both slow and meaningless, so it takes `IMAGE_PALETTE_N` anchors and
    lets the short-list blending below build the gradient from them.

    An image can hold FEWER distinct colors than there are categories (a
    two-tone image, nine groups). Unlike a user-supplied short list -- which
    raises, because the user can simply pass more colors -- a caller cannot
    add colors to an image, so the anchors are interpolated up to `n_colors`
    with the same ``blend_palette`` semantics the continuous path already
    uses (F02-006/F24-017). Interpolating keeps every category a DIFFERENT
    color and leaves the most salient anchor first; cycling the anchors
    would silently give two categories the same color, which is the
    ambiguity the short-list error exists to prevent. A single-color image
    is the one case interpolation cannot serve, and it raises."""
    colors = [tuple(c) for c in image_palette(
        source, n_colors=IMAGE_PALETTE_N if continuous else max(n_colors, 1))]
    if continuous or len(colors) >= n_colors:
        return colors
    if len(colors) == 1:
        raise ValueError(
            f"palette='{IMAGE_PALETTE_PREFIX}{source}' yielded 1 color but "
            f"{n_colors} are required (one per category/component); that "
            "image has a single dominant color, so pass a more colorful "
            "image, an explicit list of colors, or a palette name")
    return [tuple(np.asarray(c)[:3])
            for c in sns.blend_palette(colors, n_colors)]
```

#### Patch 2 of 3 — `hypertools/plot/colors.py`: the `_get_palette` string branch

Replaces `colors.py:305-306`.

```python
    if isinstance(palette, str):
        if palette.startswith(IMAGE_PALETTE_PREFIX):
            # resolve to a color LIST and fall through to the list handling
            # below, so a continuous mapping blends the extracted anchors
            # into its gradient exactly as it would any short list
            palette = _image_palette_list(
                palette[len(IMAGE_PALETTE_PREFIX):].strip(),
                n_colors, sns, continuous)
        else:
            return sns.color_palette(palette, n_colors)
```

> **Why the fall-through matters.** Returning `_image_palette_list(...)` directly from
> this branch instead of falling through breaks continuous hue with
> `IndexError: index 10 is out of bounds for axis 0 with size 6` — the 6 anchors never
> get blended to `n_bins`. This was caught by running scenario 3, not by inspection.

#### Patch 3 of 3 — `hypertools/plot/plot.py`: `_seaborn_palette_arg` (the second interception)

Replaces `plot.py:113-124`.

```python
def _seaborn_palette_arg(palette, n_colors):
    """`palette` in a form seaborn's `color_palette`/`set_palette` accept.

    plot() documents palette= as a name, a list of colors, or a matplotlib
    `Colormap` (F02-011); seaborn handles the first two natively but not a
    Colormap INSTANCE, so that one is pre-sampled to `n_colors` RGB tuples
    via `get_palette_colors` (the same resolution `mat2colors`/the colorbar
    use, keeping every path's colors identical).

    An ``'image:<path>'`` string is the same kind of case: seaborn has no
    idea what it means and raises "is not a valid palette name", so it is
    pre-resolved through `get_palette_colors` too. This is the SECOND
    interception `palette=` needs -- every call site below hands its palette
    straight to seaborn without going through `colors._get_palette`, so
    intercepting only there would leave `sns.set_palette` (and so EVERY
    matplotlib plot call) raising on an image palette."""
    from matplotlib.colors import Colormap

    from .colors import IMAGE_PALETTE_PREFIX
    if isinstance(palette, Colormap) or (
            isinstance(palette, str)
            and palette.startswith(IMAGE_PALETTE_PREFIX)):
        return [tuple(c) for c in get_palette_colors(palette, n_colors)]
    return palette
```

This single edit covers all five raw-seaborn call sites (208, 4118, 4657, 4767, 4825)
because every one of them already routes through this function.

---

## 4. Results — real execution, all six scenarios

Image used: a deterministic 100x100 PNG written to the scratch dir — 90% muted beige
canvas `(200,196,188)`, a 10% vivid red band `(220,20,20)`, plus 8 small vivid patches
so more than six distinct colors genuinely exist.
`image_palette(n_colors=6)` → 6 colors, first `[0.854 0.083 0.124]` (the red, not the beige).

| Scenario | Before fix | After fix |
|-|-|-|
| categorical hue / matplotlib | FAIL — `ValueError: 'image:…' is not a valid palette name` | **PASS** — 2 distinct colors for 2 categories |
| categorical hue / plotly | FAIL — same `ValueError` | **PASS** — 3 traces, 2 distinct marker colors |
| continuous hue | FAIL — same `ValueError` | **PASS** — 19 distinct gradient colors over 20 values |
| matrix hue | FAIL — same `ValueError` | **PASS** — 11 distinct blended colors from a (12,3) hue matrix |
| direct `palette=` (no hue) | FAIL — same `ValueError` | **PASS** — 3 distinct colors for 3 datasets |
| **> six categories (9)** | FAIL — same `ValueError` | **PASS** — 9 distinct colors for 9 categories |
| | **0/6** | **6/6** |

Note the ">6 categories" row fails *twice* under the plan as written: first on the
seaborn error, and then — once that is fixed but `IMAGE_PALETTE_N` stays hardcoded —
on `ValueError: palette= supplies 6 color(s) but 9 are required`. Both are fixed.

### Non-degeneracy checks (rendered pixels, not just objects)

A 9-category figure was rendered to a real PNG and its pixels counted:

```
saved render_9cat.png
64 distinct vivid RGB values present in the rendered PNG
9 distinct artist colors for 9 categories
```

Visual inspection of the PNG confirms nine clearly distinguishable colors (red, green,
blue, orange, purple, cyan, light green, magenta, grey). Continuous colorbar:
`ListedColormap N=100, 100 distinct colors`.

### Test results

| Suite | Before fix | After fix |
|-|-|-|
| Plan's prescribed `tests/plot/test_image_palette.py` | 2 failed, 14 passed | **16 passed** (with the §2c assertion corrected) |
| …plus the 3 new tests from §5 Change D2 | n/a | **19 passed** |
| Maintainer's 6 scenarios | 0/6 | **6/6** |
| Full suite (`pytest tests`) | n/a | **2788 passed, 13 skipped, 0 failed** |

Full-suite detail, including one failure hit and fixed during the audit, is in §6.

### Not tested, and why

Nothing in the six scenarios was blocked. `plotly` is installed in this venv, so the
plotly rows are real runs, not skips. No optional dependency was missing.

---

## 5. Exact changes Plan 4's Task 1 needs

Four concrete edits. **The plan file itself was not modified by this audit.**

### Change A — Step 4's rationale paragraph is wrong; replace it

**BEFORE** (plan lines ~183-186, in "API design, and why", item 2):

```
2. **`palette='image:<path>'`**, intercepted at the single string branch of
   `_get_palette` (`colors.py:305-306`). Because `_continuous_palette`
   (`colors.py:269`) delegates to `_get_palette`, and `mat2colors`
   (`colors.py:24`), `get_palette_colors` and `continuous_colormap` all route
   through those two functions, **one interception makes an image palette work
   on every path** — categorical hue, continuous hue, matrix hue, the
   matplotlib colorbar and the plotly colorbar — with no per-call-site change.
```

**AFTER:**

```
2. **`palette='image:<path>'`**, intercepted in **two** places, because
   `palette=` reaches seaborn by two independent routes.
   (a) `_get_palette`'s string branch (`colors.py:305-306`) serves everything
   that resolves colors through `colors.py`: `mat2colors`' categorical,
   continuous and matrix paths, `get_palette_colors`, `continuous_colormap`,
   the MultiIndex colors (`multiindex.py:151`), and both colorbars.
   (b) `_seaborn_palette_arg` (**`plot.py:113`**) serves the five call sites
   that hand `palette=` to seaborn RAW, never touching `colors.py`:
   `plot.py:208-209`, `4118-4119`, `4657-4658`, `4767-4768`, and — the fatal
   one — **`plot.py:4825-4826`'s `sns.set_palette`, which runs on EVERY
   matplotlib plot call regardless of hue**. Measured red state with only (a)
   patched: all six of categorical/plotly/continuous/matrix/direct/9-category
   raise `ValueError: 'image:…' is not a valid palette name` from
   `seaborn/palettes.py:237` via `plot.py:4825`.
```

### Change B — Step 4 gains the `_seaborn_palette_arg` edit and a dynamic count

Replace Step 4's single code block with **Patch 1, Patch 2 and Patch 3 from §3c above**,
and retitle the step:

**BEFORE:** `- [ ] **Step 4: Intercept the `'image:<path>'` spelling in one place**`
**AFTER:**  `- [ ] **Step 4: Intercept the `'image:<path>'` spelling at BOTH resolvers**`

Also replace Step 4's closing sentence:

**BEFORE:**

```
Nothing else in `colors.py` changes: `_continuous_palette` already delegates
here for every non-cyclic palette, and `'image:...'` is not in `_CYCLIC_PALETTES`.
```

**AFTER:**

```
Nothing else in `colors.py` changes: `_continuous_palette` already delegates
here for every non-cyclic palette, and `'image:...'` is not in `_CYCLIC_PALETTES`.
`plot.py` changes in exactly one function -- `_seaborn_palette_arg` -- which all
five raw-seaborn call sites already route through.
```

### Change C — the `IMAGE_PALETTE_N` constant comment

**BEFORE** (Step 3's constant block):

```python
#: `palette='image:<path>'` extracts this many anchor colors. A CONTINUOUS
#: mapping asks `_get_palette` for `n_bins` (100) colors, and clustering an
#: image into 100 groups is both slow and meaningless -- so the string form
#: always extracts this few and lets the existing short-list blending
#: (colors.py:323-331) build the gradient. Callers who want more pass an
#: explicit list: `palette=image_palette(path, n_colors=12)`.
IMAGE_PALETTE_N = 6
```

**AFTER:**

```python
#: How many anchor colors `palette='image:<path>'` extracts for a CONTINUOUS
#: mapping, which asks `_get_palette` for `n_bins` (100) colors -- clustering
#: an image into 100 groups is both slow and meaningless, so it takes this
#: few and lets the short-list blending (colors.py:323-331) build the
#: gradient. A CATEGORICAL or matrix mapping instead extracts exactly as many
#: colors as it has categories, so the number of groups is NOT capped at this
#: value; see `_image_palette_list`.
IMAGE_PALETTE_N = 6
```

### Change D — two test-file corrections

**D1.** `test_palette_string_colours_a_categorical_hue` reads the wrong artists and can
never pass. Add `from matplotlib.colors import to_rgb` to the imports, then:

**BEFORE:**

```python
    drawn = np.vstack([np.atleast_2d(c.get_facecolor())[:, :3]
                       for c in _ax(fig).collections
                       if len(np.atleast_2d(c.get_facecolor()))])
    assert any(np.allclose(c, VIVID, atol=0.02) for c in drawn)
```

**AFTER:**

```python
    # a fmt='.' plot draws Line2D artists; the only collections on a 3-D
    # axes are the pane/grid Line3DCollections, whose facecolor is (0, 4)
    drawn = [to_rgb(ln.get_color()) for ln in _ax(fig).lines]
    assert any(np.allclose(c, VIVID, atol=0.02) for c in drawn)
```

**D2.** The suite has no test for the two defects this audit found. Add these three,
and update Step 5's expected count from **16 passed** to **19 passed** (and the
suite arithmetic in the plan's verification task from +16 to +19):

```python
def test_more_than_six_categories_each_get_their_own_colour(tmp_path):
    """IMAGE_PALETTE_N must not cap the number of hue categories: a
    CATEGORICAL mapping extracts one anchor per category, not six."""
    path = six_png(tmp_path)
    n = 9
    rng = np.random.default_rng(0)
    ds = [rng.normal(size=(n * 4, 4))]
    hue = [chr(97 + i) for i in range(n) for _ in range(4)]
    fig = hyp.plot(ds, '.', hue=hue, palette=f'image:{path}', show=False)
    drawn = {tuple(np.round(to_rgb(ln.get_color()), 4))
             for ln in _ax(fig).lines}
    assert len(drawn) == n


def test_fewer_image_colours_than_categories_interpolates(tmp_path):
    """A two-tone image cannot supply nine anchors. The anchors are blended
    up rather than cycled, so no two categories share a colour, and the most
    salient anchor stays first."""
    path = painting_png(tmp_path)          # exactly 2 distinct colours
    pal = get_palette_colors(f'image:{path}', 9)
    assert len(pal) == 9
    assert len(np.unique(np.round(pal, 4), axis=0)) == 9
    assert pal[0] == pytest.approx(VIVID, abs=0.02)


def test_a_single_colour_image_cannot_serve_many_categories(tmp_path):
    """The one case interpolation cannot serve raises, naming the fixes."""
    arr = np.full((100, 100, 3), 200, np.uint8)
    path = _png(tmp_path, arr, 'solid.png')
    with pytest.raises(ValueError, match='single dominant color'):
        get_palette_colors(f'image:{path}', 5)
    assert len(get_palette_colors(f'image:{path}', 1)) == 1
```

### Change E — the `IMAGE_PALETTE_PREFIX` comment states a false fact

The plan's Step 3 justifies the prefix with a claim that is simply untrue:

**BEFORE:**

```python
#: Prefix that marks a `palette=` string as "extract this from an image".
#: Seaborn/matplotlib palette names never contain a colon, so there is no
#: collision; an unmatched name still reaches seaborn and raises its own
#: "is not a valid palette name" error.
IMAGE_PALETTE_PREFIX = 'image:'
```

Seaborn palette specs **routinely** contain a colon — `ch:`, `light:`, `dark:` and
`blend:` are all documented spellings. Verified they all still resolve correctly
(they do not *start with* `image:`, which is what the guard actually tests):

```
ch:2,r=.3,l=.8       -> OK, 4 colors, first=[0.729 0.817 0.905]
light:b              -> OK, 4 colors, first=[0.943 0.943 0.952]
dark:salmon_r        -> OK, 4 colors, first=[0.98  0.502 0.447]
blend:#7AB,#EDA      -> OK, 4 colors, first=[0.467 0.667 0.733]
```

**AFTER:**

```python
#: Prefix that marks a `palette=` string as "extract this from an image".
#: Seaborn has its own colon-prefixed spellings ('ch:', 'light:', 'dark:',
#: 'blend:'), but none is 'image:' and the guard tests the PREFIX, so there
#: is no collision; any unmatched name still reaches seaborn and raises its
#: own "is not a valid palette name" error.
IMAGE_PALETTE_PREFIX = 'image:'
```

### Also worth correcting while editing Task 1

- Step 6 cites the `palette` docstring at **`plot.py:807-820`**; it is actually
  **`plot.py:1066-1078`**.
- Step 2's predicted red state ("collection FAILS with `ImportError`") is right, but
  the *stubbed* prediction — "the four `palette='image:...'` tests fail with
  `ValueError: … is not a valid palette name`" — is wrong in detail: with the plan's
  implementation applied, only **two** fail that way; `test_palette_string_...
  _continuous_hue` passes (it never calls `hyp.plot`) and
  `test_palette_string_with_a_missing_file_names_the_file` passes for an unrelated
  reason (`mat2colors` raises `FileNotFoundError` before line 4825 is reached).

---

## 6. Full-suite regression result

See the appended result below. Non-`image:` palette handling is byte-identical: the
`else: return sns.color_palette(palette, n_colors)` branch is unchanged, and
`_seaborn_palette_arg` returns `palette` untouched for every string that does not
start with `image:`.

```
2788 passed, 13 skipped, 2 deselected, 1 warning in 587.45s (0:09:47)
```

**Zero failures.** 2788 = 2769 pre-existing + the 19 tests of this task (the 16 from
the plan, corrected per §5 Change D1, plus the 3 new ones from Change D2 — verified
independently as `19 passed in 2.36s`). The one pre-existing warning
(`Animation was deleted without rendering anything` in
`test_plot_animation_audit_fixes.py`) is unrelated to color resolution and is present
without these changes.

The suites the plan's Step 7 flags as most at risk — `tests/test_colors.py`,
`tests/plot/test_colors_module.py`, `tests/test_colorbar.py` — all pass, confirming
the non-`image:` string branch is unchanged.

Colon-prefixed seaborn palette spellings were spot-checked directly and still resolve
(see §5 Change E).

### One failure was hit and fixed during the audit

A first run with `-x` gave `1 failed, 2107 passed`:

```
FAILED tests/test_packaging_artifacts.py::test_sdist_contains_only_tracked_files_plus_allowlist
AssertionError: 1 untracked file(s) leaked into the sdist (first 10):
  ['tests/plot/test_image_palette.py']
```

This is a real guard doing its job, not a false positive: the new test file was
untracked in the audit worktree. `git add tests/plot/test_image_palette.py` — which
Task 1's own Step 9 already prescribes — fixes it, and the re-run is fully green.
**Implication for the plan:** if Task 1 is implemented without staging the new test
file before running the suite, Step 7 will report this failure. Worth a one-line note
in Step 7.

---

## 7. Reproduction commands

```bash
git worktree add /tmp/palette_audit 065c841e
# apply the plan verbatim, then:
cd <scratchdir>
PYTHONPATH=/tmp/palette_audit MPLBACKEND=Agg \
  /Users/jmanning/hypertools/.venv/bin/python -m pytest \
  /tmp/palette_audit/tests/plot/test_image_palette.py \
  -c /tmp/palette_audit/pyproject.toml --rootdir=/tmp/palette_audit -q
PYTHONPATH=/tmp/palette_audit MPLBACKEND=Agg \
  /Users/jmanning/hypertools/.venv/bin/python scenarios.py
git worktree remove /tmp/palette_audit --force
```

`PYTHONPATH` shadowing was verified (`hypertools.__file__` →
`/tmp/palette_audit/...`) so the editable install of the main checkout was never
exercised. **`hypertools/` in the main worktree was not modified.**
