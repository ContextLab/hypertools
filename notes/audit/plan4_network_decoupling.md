# Plan 4, Task 8: decoupling the example gate from the network

**Date:** 2026-08-01
**Repo:** `/Users/jmanning/hypertools`, branch `dev-1.0`, HEAD `065c841e` (clean)
**Plan under audit:** `docs/superpowers/plans/2026-07-28-hypertools-1.1-examples-and-tutorials.md` ("Plan 4")
**Worktree used for the proof:** `/tmp/netsplit_audit` @ `065c841e` (removed at the end of this audit)
**Python:** `/Users/jmanning/hypertools/.venv/bin/python` (3.12.10, matplotlib 3.10.8, sphinx-gallery 0.21.0)

> The plan file was NOT edited. `examples/` and `tests/` in the main worktree were NOT modified.

---

## 0. The contradiction, stated exactly

Plan 4 asserts, verbatim, in the docstring of the test module it prescribes
(`tests/test_examples_are_native.py`, plan line 2229):

> `No network, no mocks: it reads the committed files.`

and in **Contract 4** (plan line 92):

> **Network fetches live in examples, wrapped in a fallback, never in a library test.** Every fetch
> follows the shape the current examples already use (`animate_market_forecast.py:70-97`,
> `animate_weather_decades.py:74-95`): a `try/except Exception: return None` fetcher, a deterministic
> synthetic substitute, and a `print(...)` naming which source was used. Task 1's tests write real
> image files to `tmp_path` and touch no network. `image_palette()` deliberately does **not** accept
> a URL, so the library never fetches.

The same module then prescribes (plan lines 2351-2372):

```python
@pytest.mark.parametrize('stem', sorted(STATED_ARTIFACT))
def test_examples_produce_their_stated_artifact(stem):
    """Executable semantics, not source-shape. Each example is RUN and the
    object it advertises is inspected."""
    import runpy
    import matplotlib
    matplotlib.use('Agg')
    want = STATED_ARTIFACT[stem]
    ns = runpy.run_path(f'examples/{stem}.py')
    ...
```

`runpy.run_path` executes the example's entire module body. Every one of the five examples fetches
from its module body (§1). So the module's own docstring is false the moment this test runs, and
Contract 4's "never in a library test" is violated by the test the same plan writes.

**This is not the only instance.** Plan 4 Task 5 Step 2a (plan lines 1580-1610) prescribes a second
*committed* test, `tests/plot/test_recency_fade.py`, with a module-scoped fixture:

```python
@pytest.fixture(scope='module')
def example():
    ns = runpy.run_path('examples/animate_conversation.py')
```

whose own docstring concedes the cost — *"The example is executed once per module -- it builds
sentence embeddings and a UMAP reduction, so this is not cheap"* — and which is preceded by
`pytest.importorskip('sentence_transformers')`, i.e. it is *enabled* precisely on the machines where
the ~90 MB model download will be attempted.

The plan's *manual* `runpy` verification steps (plan lines 943, 1147, 1383, 1747, 1861) are inside
`bash` heredocs a human runs by hand. Those are fine — a maintainer running a verification step may
fetch. Only the two **committed pytest modules** are in scope here.

---

## 1. Premise verification, per example

### Method (no mocks)

Network coupling was measured with a real CPython **audit hook** (`sys.addaudithook`), which observes
`socket.getaddrinfo` / `socket.connect` / `socket.create_connection` / `urllib.Request` events raised
by the interpreter itself. Nothing is replaced, patched or faked — the hook watches the genuine call
and refuses it, so the run happens under truly unavailable network. Each example ran with a **cold
cache**: `TMPDIR` pointed at a fresh directory (defeating
`tempfile.gettempdir()/hypertools_gallery_cache`) and `HOME` pointed at a fresh directory (defeating
`~/hypertools_data`).

Sentinel script: `netsentinel.py` (scratchpad; disposable).
Command shape:

```bash
TMPDIR=$COLD/tmp HOME=$COLD/home MPLBACKEND=Agg \
  .venv/bin/python netsentinel.py examples/<stem>.py
```

### Result — all five are network-coupled, in three different ways

| example | events blocked | host | call site | offline outcome |
|-|-|-|-|-|
| `animate_weather_decades.py` | **6** | `archive-api.open-meteo.com` | `examples/animate_weather_decades.py:84` in `fetch_city_months` | degrades: `weather: 6 cities (synthetic (offline fallback))`, exit 0 |
| `animate_market_forecast.py` | **1** | `fred.stlouisfed.org` | `examples/animate_market_forecast.py:83` in `fetch_fred` | degrades: `market data: 1000 days x 5 series (synthetic basket (offline fallback))`, exit 0 |
| `animate_painting_embeddings.py` | **7** | `commons.wikimedia.org` ×5, `huggingface.co` ×2 | `examples/animate_painting_embeddings.py:131` in `canvas_color`; `:105` `SentenceTransformer('all-MiniLM-L6-v2')` | degrades: hand-picked colours + TF-IDF, exit 0 |
| `animate_conversation.py` | **2** | `huggingface.co` | `examples/animate_conversation.py:93` `SentenceTransformer('all-MiniLM-L6-v2')` | degrades: TF-IDF fallback, exit 0 |
| `animate_morph_zoo.py` | **4** | `www.dropbox.com` | `examples/animate_morph_zoo.py:67` `hyp.load(name)` → `hypertools/io/load.py:734` `_download_example_data` | **HARD FAILS**: `HypertoolsIOError: Failed to download 'bunny' dataset`, exit 17 |

**Does merely IMPORTING trigger it?** Yes, for all five. None of the five has an
`if __name__ == '__main__'` guard today; every fetch is on the module body's straight-line path
(`animate_weather_decades.py:116`, `animate_market_forecast.py:113`,
`animate_painting_embeddings.py:158-159`, `animate_conversation.py:131`,
`animate_morph_zoo.py:74`). `runpy.run_path` and `import` are identical here.

### Three distinct severities — do not treat the five alike

1. **`animate_morph_zoo.py` is a hard CI failure**, not a slow test. `hyp.load()` is not wrapped in
   `try/except`, so on a cold cache with no network the gate crashes rather than degrading. The
   example's own docstring claims *"so this example is fully offline and deterministic after the
   first run"* — true only **after** the first run; the gate's first run on a fresh CI runner is
   exactly the failing case. This one also does **not** match Contract 4's prescribed shape
   ("a `try/except Exception: return None` fetcher, a deterministic synthetic substitute"), because
   the fetch is inside the library, not the example.
2. **Weather / market / paintings / conversation are nondeterminism, not failure.** They always
   render, but *what they render depends on whether the network was up*, so the gate silently tests
   two different artifacts. That is worse than a crash for a semantic gate: `axes >= 2` passes on
   both paths, so the gate cannot tell you which one it checked.
3. **Model downloads (conversation, paintings) are the slow half.** `pytest.importorskip
   ('sentence_transformers')` in Task 5's `test_recency_fade.py` makes the download *more* likely on
   a well-provisioned runner, not less.

### Corollary defect found while verifying: the plan's assertion cannot run at all

The plan reads `ns['ani']`. Only **three** of the five examples bind `ani` at module level
(`market_forecast`, `painting_embeddings`, `morph_zoo` — all `fig, ani = hyp.plot(...)`).
`animate_weather_decades.py:186` and `animate_conversation.py:163` bind **`anim`** (a
`HyperAnimation`) and `fig`; there is no `ani`. Run verbatim:

```
$ MPLBACKEND=Agg .venv/bin/python -c "<the plan's test body, on animate_weather_decades>"
weather: 6 cities (open-meteo archive)
PLAN TEST FAILED: AssertionError: no animation was produced
names bound at top level: ['anim', 'fig']
```

So `test_examples_produce_their_stated_artifact` fails 2 of its 5 parametrisations on day one, for a
reason unrelated to what it is trying to gate. The plan's expected count of **5** passing IDs
(plan line 2447) is wrong; it would be 3 at best.

---

## 2. The `construct_artifact` split

### The boundary

```
load_<domain>()  ->  data          # the ONLY function that can touch the network
construct_artifact(data)  ->  HyperAnimation   # pure given its input; no I/O
if __name__ == '__main__':          # 3-line driver; the guard is what buys import-safety
    ...
```

**Signature, per example.** One argument, as prescribed. `data` is a small `typing.NamedTuple`
declared in the example, so `construct_artifact` reads `data.monthly`, not `data[0]`:

| example | loader | payload (`NamedTuple` fields) | fixture that drives the test |
|-|-|-|-|
| `animate_weather_decades` | `load_weather(cities=CITIES)` | `Weather(monthly, daily, hemispheres, source)` | **synthetic, 0 bytes committed** — the example's own `synthetic_city_months` / `synthetic_city_daily` (seeded `default_rng`) |
| `animate_market_forecast` | `load_market(ids=FRED_IDS, ...)` | `Market(dates, prices, source)` | **synthetic, 0 bytes** — the example's own `synthetic_basket()` (seeded) |
| `animate_conversation` | `embed_turns(TURNS)` | `Conversation(vectors, speakers, spans, source)` | **synthetic, 0 bytes** — the TF-IDF branch is already the deterministic offline path and is a real `sklearn` fit, not a stand-in |
| `animate_painting_embeddings` | `load_paintings(PAINTINGS)` | `Paintings(vectors, owners, colors, source)` | **one committed fixture, ~3 KB** — see below |
| `animate_morph_zoo` | `load_shapes(SHAPES, n=N)` | `Shapes(clouds, titles)` | **synthetic, 0 bytes** — deterministic parametric clouds; see the caveat below |

### Fixture sizing, and why four of five need none

The maintainer's brief prefers synthetic deterministic inputs "where they exercise the same code
path". For weather, market and conversation they exercise *exactly* the same path — these examples
already ship a synthetic branch that produces the same dtype, the same units and the same shape as
the real source, and everything downstream of `load_*` is common. Committing a real-data fixture
there would add bytes and buy nothing. Concretely: a real weather fixture would be 6 cities × 420
months × 4 features (float32, ~40 KB npz) **plus** 6 × ~12,800 daily temperatures (~300 KB raw,
~120 KB npz) — a quarter of a megabyte to reproduce numbers the seeded generator already produces
for free.

The one place a committed fixture *is* warranted is **paintings**: `canvas_color` runs k-means over
real JPEG pixels, and synthetic pixels would not exercise the decode-and-cluster path. Commit one
64×64 JPEG (**≈ 2–4 KB**, measured range for a 64×64 quality-80 JPEG) under
`tests/fixtures/paintings/`, and drive `canvas_color`/`image_palette` from that local path. Note
Plan 4 Task 1 already establishes exactly this pattern for `image_palette()` — *"every image is
written to `tmp_path` and read back"* (plan line 206) — so this is consistent with, not new to, the
plan.

**Caveat for `animate_morph_zoo`.** Its input is `hyp.load('bunny')` etc. — *library* example data,
not example-owned data. A synthetic cloud tests the morph but not `hyp.load`. Two honest options:
(a) drive `construct_artifact` with deterministic parametric clouds (sphere/cube/torus generated in
the test — this is what the morph actually consumes, since the example immediately `normalize()`s
and subsamples to 2000 points), and let the opt-in smoke test cover `hyp.load`; or (b) commit one
2000×3 float32 cloud (**24 KB**) as a shape stand-in. **(a) is recommended** — `hyp.load`'s download
path already has its own tests in `tests/` and does not need re-testing through an example.

### Readability: before / after, `animate_weather_decades.py`

The concern is real and was checked, not assumed. **These examples contain zero sphinx-gallery
narration blocks** — measured: `grep -c "^# %%\|^####"` returns `0` for all five. sphinx-gallery
therefore already renders each as *one* prose docstring followed by *one* code block, so the split
cannot fragment interleaved narration, because there is none to fragment.

**BEFORE** (top-level structure, `examples/animate_weather_decades.py` @ `065c841e`):

```
"""...docstring..."""
imports; CACHE; START/END; CITIES; FEATS                      # :44-71
def fetch_city_months(name, lat, lon)                         # :74     <- NETWORK
def synthetic_city_months(hemi, ...)                          # :98
mats, hemis, offline = [], [], False                          # :114    <- runs the fetch
for ...: m = fetch_city_months(...) or synthetic_city_months(...)
print('weather: ...')                                         # :122
min_len = ...; city_loops = hyp.reduce(...)                   # :131-136
Nmean_loop / Smean_loop / colormaps / enc()                   # :139-162
anim = hyp.plot(...); fig = anim.figure; ax = ...             # :186-193
def fetch_city_daily_temp(name)                               # :214    <- NETWORK (cache read)
def synthetic_city_daily(hemi, n_days, ...)                   # :229
daily = [...]  (second fetch loop)                            # :239-247
ax_t = fig.add_axes(...); temp_line(); colorbars; title       # :254-309
def decorate(ctx)                                             # :312
anim.on_frame(decorate)                                       # :336
```

**AFTER** (`/tmp/netsplit_audit/examples/animate_weather_decades.py`):

```
"""...same docstring + one new paragraph, "Shape of this file"..."""
imports; CACHE; START/END; CITIES; FEATS                      # unchanged
class Weather(NamedTuple)                                     # monthly / daily / hemispheres / source
# --- the data half: the ONLY code here that reaches the network ---
def fetch_city_months(name, lat, lon)                         # unchanged
def fetch_city_daily_temp(name)                               # unchanged, moved UP beside its sibling
def synthetic_city_months(hemi, ...)                          # unchanged
def synthetic_city_daily(hemi, n_days, ...)                   # unchanged
def load_weather(cities=CITIES) -> Weather                    # the two fetch loops, now named
# --- the figure half: no network, no I/O, deterministic given its input ---
def construct_artifact(data) -> HyperAnimation                # everything else, verbatim, indented
if __name__ == '__main__':
    weather = load_weather()
    print(f'weather: {len(CITIES)} cities ({weather.source})')
    anim = construct_artifact(weather)
    fig = anim.figure
```

**Readability verdict: preserved, and arguably improved.** The narrative order is unchanged; the
reader now gets two labelled halves instead of a 336-line straight line with the two fetch loops
150 lines apart (`:114` and `:239`). The per-frame `decorate` closure now visibly closes over the
figure's own state instead of module globals. What it costs:

- **+15 code lines** (195 → 210 logical statements; 336 → 381 raw). Measured with an AST-based
  counter that ignores comments, blanks and docstrings. The overhead is the `NamedTuple` (6), the
  two `def` lines, `load_weather`'s own scaffolding, and the 4-line guard.
- One level of indentation on the figure half.

**Flag for the plan author:** Task 8 budgets `('examples/animate_weather_decades.py', 62)` code lines
(plan line 2243). A ~15-line split overhead is ~24% of that budget. Either raise the budget by 15 in
the plan (Contract 6 says budgets are renegotiated *in the plan*, never weakened in the test), or
drop the `NamedTuple` and return a plain tuple — `construct_artifact(monthly, daily, hemis)` — which
saves ~6 lines at the cost of the self-documenting field names. The `NamedTuple` is recommended;
the budget bump is the honest fix.

**sphinx-gallery compatibility: verified, not assumed.** sphinx-gallery 0.21.0 executes each example
inside a *fake `__main__` module* — `sphinx_gallery/gen_rst.py:1271-1280`:

```python
    # Examples may contain if __name__ == '__main__' guards
    # for in example scikit-learn if the example uses multiprocessing.
    # Here we create a new __main__ module, and temporarily change
    # sys.modules when running our example
    fake_main = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("__main__", None)
    )
    example_globals = fake_main.__dict__
```

so the guarded driver **does** run at docs build. Confirmed end to end:

```
$ MPLBACKEND=Agg .venv/bin/python /tmp/netsplit_audit/examples/animate_weather_decades.py
weather: 6 cities (open-meteo archive)
EXIT=0
```

(Separately noted, unchanged by this split: `hyp.plot(..., show=False)` does not register its figure
with pyplot — measured `plt.get_fignums() == []` after a `show=False` call, `[1]` with the default —
so sphinx-gallery's `'matplotlib'` scraper (`docs/conf.py:355`) sees nothing for these five either
way. That is consistent with Plan 4's own finding that none of the five has a gallery thumbnail, and
is Task 8 Step 6's business, not this audit's.)

---

## 3. Replacing `ani._save_count`

### What it actually is

`_save_count` is **not** a hypertools attribute. `HyperAnimation`
(`hypertools/plot/hyper_animation.py:45`) defines exactly two properties, `figure` (`:67`) and
`animation` (`:72`) — there is no `_save_count` and no frame-count accessor. Measured:

```
type(out) = HyperAnimation
type(ani) = FuncAnimation
hasattr(out, '_save_count') = False        <-- the HyperAnimation has none
hasattr(ani, '_save_count') = True
ani._save_count = 40                        (duration=4 * frame_rate=10)
len(list(ani.new_frame_seq())) = 40
hasattr(ani, 'save_count') = False          <-- no public matplotlib accessor
```

It is matplotlib's private `FuncAnimation.__init__` field (`self._save_count = frames` when `frames`
is an int, matplotlib 3.10.8). `hyp.plot` always passes an int —
`max(1, int(round(frame_rate * duration)))` for parallel/serial/spin
(`hypertools/plot/matplotlib_backend.py:1991, 2013, 2024`) and `sum(frame_counts)` for a morph
(`:2039`) — so the value is always known and always equals `frame_rate × duration`.

### Two defects in the plan's use of it

1. **It reaches for a private matplotlib field from a test** — which Plan 4's own Contract 3 forbids
   for examples (*"After this plan, no example or notebook contains `ani._func`, `ani._args`,
   `hypertools._shared` ... or any other name not documented in `plot()`'s docstring"*, plan line 78)
   and whose spirit plainly extends to the gate that enforces it. The plan even lists `ani._func`
   and `ani._args` in `DEFECT_MARKERS` (plan lines 2262-2263) while itself using `ani._save_count`
   ten lines further down.
2. **The assertion is a tautology.** `assert ns['ani']._save_count >= 1` can never fail: the value is
   `max(1, ...)` by construction, so `>= 1` holds even for a zero-length, zero-dataset animation.
   It gates nothing.

### Recommendation: a library-owned `HyperAnimation.n_frames`

There is no existing supported property to point at (verified above), so the library should own one.
Encapsulating matplotlib's private field inside hypertools is the right home for the reach — the
library already makes exactly this reach at `hypertools/plot/animate.py:174-175`:

```python
    total = getattr(line_ani, '_save_count', None) or \
        getattr(line_ani, 'save_count', None) or 100
```

Exact code, to be inserted in `hypertools/plot/hyper_animation.py` immediately after the `animation`
property (`:77`):

```python
    @property
    def n_frames(self):
        """How many frames this animation plays.

        The supported way to ask an animation its length. ``hyp.plot`` always
        builds the animation with an explicit integer frame count --
        ``max(1, round(frame_rate * duration))`` for the parallel/serial/spin
        styles and ``sum(frame_counts)`` for a morph
        (``matplotlib_backend.py``) -- so the count is always known.
        matplotlib stores it on the private ``FuncAnimation._save_count`` and
        offers no public accessor, so the library reads that private
        attribute HERE, once, instead of every caller reaching for it (the
        same reach ``animate.py``'s ``_save_animation`` already makes).

        Raises ``TypeError`` if the wrapped animation has no knowable frame
        count -- an unbounded ``frames=`` generator, which ``hyp.plot`` never
        constructs.
        """
        for attr in ('_save_count', 'save_count'):
            count = getattr(self.animation, attr, None)
            if count is not None:
                return int(count)
        raise TypeError(
            'this animation has no knowable frame count (its frame sequence '
            'is unbounded); every animation hyp.plot builds is finite, so '
            'this HyperAnimation was not constructed by hyp.plot')
```

Why not `len(list(ani.new_frame_seq()))`, which is public matplotlib and returns the same 40?
Because `new_frame_seq()` returns a bare `range_iterator` with **no `__len__`** (measured), so
counting it means materialising the sequence — and if the animation were ever built with
`frames=None`, matplotlib's `itertools.count()` would make that hang forever. `_save_count` is the
cheap, safe read, and it belongs behind the library's own property.

Verified against the real class in the worktree:

```
tests/...::test_hyperanimation_n_frames_matches_duration_times_frame_rate PASSED
tests/...::test_n_frames_raises_when_the_frame_count_is_unknowable       PASSED
```

with `artifact.n_frames == 36` for `duration=3, frame_rate=12`, and
`artifact.n_frames == len(list(artifact.animation.new_frame_seq()))` — i.e. the property agrees with
the public matplotlib iterator, and the `TypeError` branch is exercised.

Note this also **fixes the `ns['ani']` breakage for free**: `n_frames` lives on the `HyperAnimation`,
which every example has (whether it binds it as `anim` or unpacks it as `fig, ani`), so the gate no
longer depends on which name an example happens to use.

---

## 4. Proof

Implemented in the disposable worktree `/tmp/netsplit_audit` @ `065c841e`:

| file | change |
|-|-|
| `examples/animate_weather_decades.py` | split into `load_weather()` / `construct_artifact(data)` + guarded driver |
| `hypertools/plot/hyper_animation.py` | `+ HyperAnimation.n_frames` |
| `tests/test_examples_produce_their_stated_artifact.py` | new — the fixture-driven replacement gate |
| `tests/test_examples_smoke.py` | new — the opt-in whole-example smoke test |

### How "no network" is proven, twice

**(a) In-test, committed, enforced.** The test module installs a real CPython audit hook and arms it
for the duration of each test via a `no_network` fixture. A fetch on the gate's code path raises
`AssertionError` naming the URL. **This is not a mock** — nothing is patched or substituted; the
interpreter's own auditing subsystem observes the genuine call. The guard carries its own **negative
control** (`test_the_no_network_guard_actually_catches_a_fetch`), which deliberately dials
`archive-api.open-meteo.com` and requires the guard to fire — without it, "no network" would be an
untested claim.

**(b) Out-of-test, at process level, cold cache.** The whole pytest process was additionally run
under the external audit sentinel with `TMPDIR` and `HOME` pointed at fresh directories, so no cache
could satisfy a request silently.

### Real pytest output

Ordinary run:

```
$ cd /tmp/netsplit_audit && MPLBACKEND=Agg .venv/bin/python -m pytest \
      tests/test_examples_produce_their_stated_artifact.py -v -p no:randomly

platform darwin -- Python 3.12.10, pytest-9.0.2, pluggy-1.6.0
rootdir: /private/tmp/netsplit_audit
configfile: pyproject.toml
collected 7 items

tests/...::test_the_no_network_guard_actually_catches_a_fetch          PASSED [ 14%]
tests/...::test_importing_the_weather_example_fetches_nothing          PASSED [ 28%]
tests/...::test_weather_example_produces_its_stated_artifact           PASSED [ 42%]
tests/...::test_weather_artifact_is_deterministic_given_its_input      PASSED [ 57%]
tests/...::test_weather_per_frame_hook_runs_and_moves_the_now_cursor   PASSED [ 71%]
tests/...::test_hyperanimation_n_frames_matches_duration_times_frame_rate PASSED [ 85%]
tests/...::test_n_frames_raises_when_the_frame_count_is_unknowable     PASSED [100%]

============================== 7 passed in 2.57s ===============================
```

Under the external audit sentinel, cold `TMPDIR` + cold `HOME`, negative control deselected (it
exists only to dial out, and would trip the outer sentinel first — hooks stack in registration
order):

```
$ TMPDIR=$COLD/tmp HOME=$COLD/home MPLBACKEND=Agg \
    .venv/bin/python netsentinel.py .venv/bin/pytest \
    tests/test_examples_produce_their_stated_artifact.py -v -p no:randomly \
    --deselect ...::test_the_no_network_guard_actually_catches_a_fetch

collected 7 items / 1 deselected / 6 selected
...::test_importing_the_weather_example_fetches_nothing                PASSED [ 16%]
...::test_weather_example_produces_its_stated_artifact                 PASSED [ 33%]
...::test_weather_artifact_is_deterministic_given_its_input            PASSED [ 50%]
...::test_weather_per_frame_hook_runs_and_moves_the_now_cursor         PASSED [ 66%]
...::test_hyperanimation_n_frames_matches_duration_times_frame_rate    PASSED [ 83%]
...::test_n_frames_raises_when_the_frame_count_is_unknowable           PASSED [100%]

======================= 6 passed, 1 deselected in 16.26s =======================
SystemExit: 0

[sentinel] 0 blocked network event(s)
```

**`0 blocked network event(s)`** on a cold cache is the strongest available proof: the gate's code
path did not merely *fail* to reach the network, it never *attempted* to.

For contrast, when the negative control is included, the sentinel reports exactly one attempt, and
attributes it to the control:

```
[sentinel] 1 blocked network event(s)
[sentinel]   urllib.Request https://archive-api.open-meteo.com/v1/archive
[sentinel]       .../tests/test_examples_produce_their_stated_artifact.py:111
                     in test_the_no_network_guard_actually_catches_a_fetch
```

### The opt-in smoke test, verified both ways

```
$ MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_examples_smoke.py -q -p no:randomly
sssss                                                                    [100%]
5 skipped in 0.23s

$ HYPERTOOLS_EXAMPLE_SMOKE=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
    "tests/test_examples_smoke.py::test_example_runs_end_to_end[animate_weather_decades]" -q
.                                                                        [100%]
1 passed in 2.85s
```

### Cost comparison

| | plan's `runpy` gate | this gate |
|-|-|-|
| network on the CI path | yes, 1–7 requests per example | **none** (enforced) |
| behaviour on cold-cache CI | `animate_morph_zoo` **crashes**; other four silently test a different artifact | identical every run |
| wall clock, weather | ~17 s cold (network round trips) + full data | **2.6 s** |
| passes today | **no** — `KeyError`/`AssertionError` on 2 of 5 (`ani` unbound) | yes, 7/7 |
| private API reached | `ani._save_count`, `ani._func`, `ani._args` | none |

---

## 5. Exact before/after blocks for Plan 4

### 5a. Contract 4 (plan line 92) — corrected wording

**BEFORE:**

> 4. **Network fetches live in examples, wrapped in a fallback, never in a library test.** Every
>    fetch follows the shape the current examples already use (`animate_market_forecast.py:70-97`,
>    `animate_weather_decades.py:74-95`): a `try/except Exception: return None` fetcher, a
>    deterministic synthetic substitute, and a `print(...)` naming which source was used. Task 1's
>    tests write real image files to `tmp_path` and touch no network. `image_palette()` deliberately
>    does **not** accept a URL, so the library never fetches.

**AFTER:**

> 4. **Network fetches live in an example's loader, behind a guarded driver, and never on any
>    test's code path.** Every fetch follows the shape the current examples already use
>    (`animate_market_forecast.py:70-97`, `animate_weather_decades.py:74-95`): a
>    `try/except Exception: return None` fetcher, a deterministic synthetic substitute, and a
>    `print(...)` naming which source was used. **`animate_morph_zoo.py` is the exception that must
>    be fixed: its data comes from `hyp.load()` (`:67` → `hypertools/io/load.py:734`), which is not
>    wrapped and raises `HypertoolsIOError: Failed to download 'bunny' dataset` on a cold cache with
>    no network — measured 2026-08-01. Give it the same fallback shape as its four siblings.**
>
>    Each example is structured as `load_<domain>() → data`, `construct_artifact(data) → artifact`,
>    and a driver under `if __name__ == '__main__':`. **Importing an example must fetch nothing**;
>    sphinx-gallery runs examples inside a fake `__main__` module
>    (`sphinx_gallery/gen_rst.py:1271-1280`), so the guard does not change the gallery build.
>
>    **No committed test may execute an example's driver.** Tests import the example and call
>    `construct_artifact()` with the example's own seeded synthetic input, or with a small committed
>    fixture (paintings: one 64×64 JPEG, ~3 KB). The prohibition is *enforced*, not documented: the
>    gate arms a CPython audit hook (`sys.addaudithook`) that raises on `socket.getaddrinfo` /
>    `socket.connect` / `urllib.Request` while a test runs, and carries a negative control proving
>    the guard still fires. An audit hook is not a mock — nothing is patched; the interpreter
>    observes the real call.
>
>    The one place an example's driver *is* executed is `tests/test_examples_smoke.py`, which is
>    skipped unless `HYPERTOOLS_EXAMPLE_SMOKE=1` and belongs on a nightly/release schedule, never on
>    the PR path.
>
>    Task 1's tests write real image files to `tmp_path` and touch no network. `image_palette()`
>    deliberately does **not** accept a URL, so the library never fetches.

### 5b. `test_examples_are_native.py` docstring (plan line 2229)

**BEFORE:**

```
No network, no mocks: it reads the committed files.
```

**AFTER:**

```
No network, no mocks. This module only reads committed files; the executable
semantic gate lives in tests/test_examples_produce_their_stated_artifact.py,
which imports each example (never runs its driver) and enforces the offline
property with an armed CPython audit hook rather than by convention.
```

### 5c. The replacement gate (plan lines 2338-2372)

**BEFORE** — delete `STATED_ARTIFACT` and `test_examples_produce_their_stated_artifact` from
`tests/test_examples_are_native.py` entirely (plan lines 2338-2372, the `import runpy` /
`ns['ani']._save_count` block quoted in §0).

**AFTER** — a new module, `tests/test_examples_produce_their_stated_artifact.py`. Full working source
is in the worktree; the load-bearing pieces are:

```python
_NET_EVENTS = ('socket.getaddrinfo', 'socket.connect', 'socket.gethostbyname',
               'socket.create_connection', 'urllib.Request')
#: one-slot switch the permanently-installed audit hook reads. An audit hook
#: can never be uninstalled (CPython, by design), so it is installed once and
#: armed only for the duration of a test that claims to be offline.
_ARMED = []


def _audit(event, args):
    if _ARMED and event in _NET_EVENTS:
        raise AssertionError(
            f'this test performed network access: {event} {args[:1]!r} -- '
            'the example-artifact gate must run offline; move the fetch '
            'behind the example driver or into the opt-in smoke test')


sys.addaudithook(_audit)


@pytest.fixture
def no_network():
    """Make any real network call inside the test raise immediately."""
    _ARMED.append(True)
    try:
        yield
    finally:
        _ARMED.clear()


def load_example(stem):
    """Import ``examples/<stem>.py`` as a module.

    The example's network-touching driver is behind ``if __name__ ==
    '__main__'``, and this loads it under its own name, so importing it
    fetches nothing.
    """
    path = os.path.join(REPO, 'examples', f'{stem}.py')
    spec = importlib.util.spec_from_file_location(f'_gallery_{stem}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_no_network_guard_actually_catches_a_fetch(no_network):
    """Negative control. Without this, "no network" is an untested claim."""
    import urllib.request
    with pytest.raises(AssertionError, match='performed network access'):
        urllib.request.urlopen('https://archive-api.open-meteo.com/v1/archive',
                               timeout=5)


def test_weather_example_produces_its_stated_artifact(no_network,
                                                      closed_figures):
    example = load_example('animate_weather_decades')
    artifact = example.construct_artifact(synthetic_weather(example))

    assert isinstance(artifact, HyperAnimation)
    assert artifact.n_frames == 160, 'duration=8 * frame_rate=20'
    fig = artifact.figure
    assert len([a for a in fig.axes if hasattr(a, 'zaxis')]) == 1
    assert len(fig.axes) >= 2, 'the second panel was dropped'
    assert any(a.get_title() == 'every city, every day' for a in fig.axes)
```

Two things changed in the *assertions*, not just the plumbing:

- `assert artifact.n_frames == 160` — an **exact** count derived from the example's own
  `duration=8, fps=20`, replacing the tautological `_save_count >= 1`. This one can actually fail.
- `isinstance(artifact, HyperAnimation)` replaces `ns.get('ani') is not None`, so the gate no longer
  depends on whether an example binds `anim` or `ani`.

Per-example assertions replacing the `STATED_ARTIFACT` dict (one focused test each, rather than a
parametrised bag of optional flags — a `dict(animated=True, axes=2)` entry silently checks nothing
when a key is absent):

| example | the artifact it advertises | assertion |
|-|-|-|
| `animate_market_forecast` | animated + a forecast overlay | `n_frames == 160`; `any(ln.get_linestyle() in ('--', ':') for ln in ax.lines)` |
| `animate_weather_decades` | animated + a second panel | `n_frames == 160`; `len(fig.axes) >= 2`; the panel's title |
| `animate_painting_embeddings` | spin + one canvas colour per painting | `n_frames == 240`; `len(colors) == 5`, each a 3-tuple in `[0, 1]` |
| `animate_conversation` | serial reveal + a per-frame hook | `n_frames == 192`; the hook fires and alphas change between frames |
| `animate_morph_zoo` | a morph with per-segment titles | `n_frames == 240`; titles blank on transitions, named on holds |

### 5d. The opt-in smoke test (new file)

```python
pytestmark = pytest.mark.skipif(
    os.environ.get('HYPERTOOLS_EXAMPLE_SMOKE') != '1',
    reason='network + model downloads; set HYPERTOOLS_EXAMPLE_SMOKE=1 to run')


@pytest.mark.parametrize('stem', EXAMPLES)
def test_example_runs_end_to_end(stem):
    """Run the example in a SUBPROCESS, exactly as a user or the gallery
    build would (`python examples/<stem>.py`), and require a clean exit."""
    path = os.path.join(REPO, 'examples', f'{stem}.py')
    env = dict(os.environ, MPLBACKEND='Agg')
    result = subprocess.run([sys.executable, path], env=env, cwd=REPO,
                            capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, (
        f'{stem} exited {result.returncode}:\n{result.stderr[-4000:]}')
    assert 'Traceback' not in result.stderr, result.stderr[-4000:]
```

**How it is enabled:** `HYPERTOOLS_EXAMPLE_SMOKE=1`. In CI, a scheduled (`cron`) workflow only —
never the PR path. A subprocess rather than `runpy` because these scripts leave global matplotlib
and RNG state behind, and one crashing must not take the suite with it.

### 5e. Task 5's `tests/plot/test_recency_fade.py` (plan lines 1580-1610)

**BEFORE:**

```python
@pytest.fixture(scope='module')
def example():
    ns = runpy.run_path('examples/animate_conversation.py')
    yield ns
    plt.close('all')
```

**AFTER:**

```python
@pytest.fixture(scope='module')
def example():
    """Import the example (its driver is guarded, so this fetches nothing)
    and build the artifact from the deterministic TF-IDF embedding path."""
    module = load_example('animate_conversation')
    artifact = module.construct_artifact(module.embed_turns(module.TURNS,
                                                            vectorizer='tfidf'))
    yield module, artifact
    plt.close('all')
```

and **drop** `pytest.importorskip('sentence_transformers')` — that line does not skip the download,
it *selects for* it. The recency-fade logic under test is identical on both embedding paths.

### 5f. Test-count arithmetic (plan lines 2438-2452, 2508, 2609)

Removing `test_examples_produce_their_stated_artifact` (5 IDs) from `test_examples_are_native.py`
takes it from **106** to **101**. The new
`tests/test_examples_produce_their_stated_artifact.py` contributes **11** for the full five-example
build-out (1 negative control + 1 import-safety + 5 artifact + 2 `n_frames` + 2 determinism), plus
**5 skipped** from `test_examples_smoke.py`. Net Task 8 delta: **101 + 11 = 112** passing, 5 skipped.
The plan's `+134` total (line 2508) becomes **+140 passing, +5 skipped** — re-derive it in the plan
rather than patching the number in the test file.

---

## 6. Summary of defects found

| # | severity | finding |
|-|-|-|
| 1 | **Fatal** | `test_examples_produce_their_stated_artifact` executes all five examples via `runpy`, fetching from 5 hosts — contradicting its own docstring (plan 2229) and Contract 4 (plan 92). |
| 2 | **Fatal** | The same test fails on day one for an unrelated reason: `ns['ani']` is unbound in `animate_weather_decades.py` and `animate_conversation.py` (they bind `anim`). Verified: `AssertionError: no animation was produced`. Expected 5 passing IDs is at best 3. |
| 3 | **High** | `animate_morph_zoo.py` has **no** offline fallback: `hyp.load()` raises `HypertoolsIOError` on a cold cache with no network (exit 17). Cold-cache CI crashes, it does not degrade. Contract 4 asserts all fetches follow the fallback shape; this one does not. |
| 4 | **High** | `tests/plot/test_recency_fade.py` (Task 5) is a second committed `runpy` test, and its `pytest.importorskip('sentence_transformers')` *selects for* the ~90 MB model download. |
| 5 | **Medium** | `assert ns['ani']._save_count >= 1` is a tautology — the value is `max(1, ...)` by construction and can never be `< 1`. The gate gates nothing. |
| 6 | **Medium** | The gate reaches a private matplotlib attribute while the same module lists `ani._func`/`ani._args` as `DEFECT_MARKERS` (plan 2262-2263). No `HyperAnimation` frame-count property exists; one must be added. |
| 7 | **Low** | Weather's 62-line budget (plan 2243) does not account for the ~15-line split overhead. Renegotiate in the plan per Contract 6. |
| 8 | **Low** | Task 8's expected counts (106 at plan 2438-2452; +134 at 2508; +126 at 2609) all shift; re-derive. |

---

## 7. Cleanup

Worktree `/tmp/netsplit_audit` removed (`git worktree remove --force`) after the runs above; verified
with `git worktree list`. No files in `examples/` or `tests/` of the main worktree were modified, and
the plan file was not edited.
