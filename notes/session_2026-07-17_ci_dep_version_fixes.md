# Session 2026-07-17: CI run 29582796739 dependency-version failures (dev-1.0-refactor @ 24dcd740)

Task: local suite green (2333 passed) but CI failed all 12 jobs on newer deps
(mpl 3.11.0, plotly 6.9.0, skl 1.9.0, np 2.4.x, pd 3.0.3, no `build`).
Reproduced with a scratch venv at
/private/tmp/claude-501/.../scratchpad/ci-venv (python3.12, exact CI pins).
Main .venv treated as READ-ONLY-RUN-ONLY (never pip-modified).

## Cluster 1 — test_round3.py::test_animated_svg_plotly "hang" (CI job-killer)
- NOT a semantic hang: kaleido 1.x launches a fresh headless-Chrome per
  `to_image` call (~3s/call here, far worse on 2-core CI). 60 frames -> 183s
  locally, >1200s on CI -> pytest-timeout thread-abort killed the job.
  Same per-call cost on plotly 6.8 AND 6.9 (measured) — CI was slower runner,
  not a plotly regression.
- FIX (hypertools/plot/plotly_backend.py):
  - `_shared_kaleido_session()` context manager: kaleido.start_sync_server/
    stop_sync_server around the per-frame export loops (svg + png/gif/video).
    Reuses (never stops) an already-running server; no-op fallback for
    kaleido without the sync-server API. Suppresses (narrowly) plotly's
    per-call "kopts argument is ignored if using a server" UserWarning.
  - Hoisted the frameless `base = go.Figure(fig)` copy out of
    frame_snapshots() (was O(n_frames^2): re-copied all embedded frames per
    frame, then threw them away).
- Result: 183s -> 47s locally in BOTH venvs; CI cold-starts eliminated.

## Cluster 2 — test_animation_margins.py (10F) + round17 pixel test (macOS job)
- NOT margins: matplotlib 3.11 `plt.close()` now DETACHES the figure's real
  canvas (swaps in bare FigureCanvasBase; draw() no-op, no buffer_rgba).
  plot.py's show=False branch closes the figure (GH #148/F09-003) then
  returns it -> returned fig unrenderable on mpl >= 3.11.
- FIX (hypertools/plot/plot.py, show=False close branch): capture
  `_live_canvas = fig.canvas` before plt.close(fig); re-attach via
  fig.set_canvas(_live_canvas) if matplotlib swapped it. Restores mpl<=3.10
  close semantics (deregistered but renderable). No-op on 3.10.

## Cluster 3 — test_format_data_f08_fixes typo-vectorizer/semantic tests
- CI has no text extra -> datawrangler raises ModuleNotFoundError (an
  ImportError, NOT OSError) before any network call -> rewrap missed it.
- FIX (hypertools/tools/format_data.py): `except (OSError, ImportError)`.
- BONUS BUG FOUND by new guarded test: text2mat `_resolve_registry_name`
  permanently inserts unknown names into the live vectorizer_models/texts
  registries, so the rewrap only fired on a name's FIRST use per process.
  FIX: frozen `_SKLEARN_VECTORIZER_NAMES`/`_SKLEARN_SEMANTIC_NAMES`
  (text2mat.py, captured at import) now used for the membership test in
  format_data.py.
- TESTS: original two tests unchanged (now env-independent). Added
  `test_typo_vectorizer_chains_the_real_hf_network_error` (skipif no
  sentence_transformers; real HF call, asserts OSError cause) and
  `test_typo_names_without_hf_tier_raise_same_clear_valueerror` (subprocess
  with REAL sys.meta_path blocker for sentence_transformers, mirrors
  test_gensim_text.py pattern; no mocks).

## Cluster 4 — test_packaging_artifacts.py 10 ERRORs (no `build` on CI)
- FIX: pyproject.toml dev extra += "build>=1.2" (comment cites run
  29582796739); module-level `pytest.importorskip('build')` so a build-less
  env SKIPs cleanly instead of 10 fixture ERRORs.

## Also fixed en route
- tests/test_round3.py: 4x unclosed `open(out).read()` -> with-blocks
  (pre-existing ResourceWarning, surfaced under -W error; assertions
  untouched).
- NOTE: tests/test_io_audit_final.py has an unrelated pre-existing
  working-tree modification (not from this session).

## Verification results
- scratch venv (CI pins): margins 19/19 + round17 pixel PASS;
  format_data+round17-full+backend_state 69P/1S (HF-skip); packaging 11/11;
  test_round3.py FULL 9/9 in 51s WITH -W error::ResourceWarning (formerly
  hanging svg test now 47s); plot/plot_save/morph sweep 132/132.
- main venv (mpl 3.10.8/plotly 6.8.0): packaging 11/11; margins/round17-full
  green; format_data 28/28 incl. real-HF-call tests (registry-pollution fix
  verified against real HF network errors, repeat-use case); svg test 47s
  (was ~183s); plot/plot_save/morph/round3-full sweep 141/141.
ALL AFFECTED FILES GREEN IN BOTH VENVS. No git commits made (task forbade
git write commands); working tree holds the fixes for the coordinator.
