# Morph Animation + Tight Hulls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Continues PR #272 work (Plan 10). Branch dev-1.0-refactor; NEVER push master; NEVER merge the PR.

**Goal (Jeremy, 2026-07-06):** (1) hulls must be TIGHTER — hug the observations; current surface-morph frames show surfaces exceeding the axes bounds and obvious slack. (2) New `animate='morph'` style — Hungarian-matched morphs BETWEEN datasets-as-point-clouds (like the shapes demo, but a first-class library animation). (3) `rotations` accepts a per-segment list for morph. (4) Shape demos switch to `animate='morph'`; docs/tutorials updated. Evidence in PR #272 with screenshots for BOTH backends.

## Global Constraints
- Same as Plan 10 (`docs/superpowers/plans/2026-07-05-seven-features.md`): venv python, MPLBACKEND=Agg, TDD no mocks, both backends similar quality, 6 kaleido deselects locally, commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`, evidence PNGs in docs/images/v1.0-seven-features/.
- Suite baseline: 681 passed @ eaf7fcaa.

### Task M1: Tight hulls
**Files:** hypertools/plot/meshutil.py, hypertools/plot/surface.py, tests/test_meshutil.py, tests/test_surface.py.
Evidence of the problem: /private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-066a-4ce2-b1f6-7c07c5e87b15/scratchpad/morph_inspect/grid.png (frames f20/f60/f100/f140 show box-escape; f100 shows a mid-morph EXPLOSION — surface swallows the whole axes box).
- Diagnose the f100 explosion: reproduce the demo's frame-100 interpolated cloud and inspect what smooth_hull_3d + the grow-only containment rescale produce. Suspect the rescale over-grows (ratio computed against nearest-vertex distances can blow up) — fix the rescale to be mathematically bounded (e.g. per-ray scale capped; scale = max over hull verts of (r_hull / r_mesh_along_ray) computed robustly via support function, cap ≤1.10).
- Tighten: default `pre_inflate` 1.15 → 1.0 (no blanket padding). After interleaved subdivide+Taubin, apply a MINIMAL uniform grow about the centroid so ≥99% of ORIGINAL INPUT POINTS (not just hull verts) are inside (tolerance 0.5% of cloud extent); assert final slack: mean distance from each original hull vertex to the mesh surface ≤2% of cloud extent (add a `hull_slack()` test helper measuring it). Verify visually: single blob + cube + bunny cloud renders show the surface kissing the extreme points.
- Cube-scale interaction: with tight hulls, `surface_cube_scale` margins shrink accordingly; the axes box should rarely need to grow beyond 1.02.
- Tests: tightness metric (slack ≤2%), containment ≥99%, explosion regression (the exact f100-style interpolated cloud → mesh max|vert| ≤ 1.05 * max|point|), all sizes n=5..2000. Update any tests pinned to pre_inflate=1.15 defaults.
- Re-render + VIEW: 8-frame grid of the CURRENT animate_surface_morph example (unchanged code) — no box escapes, snug holds. Commit.

### Task M2: `animate='morph'` (both backends) + `rotations` list
**Files:** hypertools/plot/plot.py, matplotlib_backend.py, plotly_backend.py, new hypertools/plot/morph.py, tests/test_morph_animation.py.
**Spec:**
- `animate='morph'`: treats the input datasets (post analyze/reduce/align pipeline, in plot order) as point clouds; morphs ds1→ds2→...→dsN with hold segments: 2N-1 segments [hold1, morph1→2, hold2, ..., holdN]. Requires ≥2 morphing datasets else ValueError.
- Matching (hypertools/plot/morph.py): sample min-count points per morphing dataset (seeded rng, no replacement; expose `morph_samples` kwarg default None=min count, cap 1000 for the Hungarian cost), chain `scipy.optimize.linear_sum_assignment` on `cdist` between consecutive clouds (exactly the shapes-demo algorithm). Smoothstep easing within morph segments. Provide `morph_schedule(n_datasets, n_steps)` + `morph_frame(clouds, frame)` helpers reusable by both backends.
- `animate` may also be a LIST (len = n datasets) with entries 'morph' or None/False: 'morph'-tagged datasets join the morph sequence (in order); untagged render as STATIC clouds/lines in the background. Any other mode string inside a list → ValueError. Scalar 'morph' = all datasets tagged.
- Drawing: one point artist (mpl) / one scatter3d trace (plotly) for the morphing cloud; marker size ~ shapes demo; color = each dataset's palette color, LINEARLY INTERPOLATED (RGB) across morph segments; holds use the dataset color. Camera spins per rotations.
- `rotations=` (new kwarg or extend existing rotation handling — CHECK what exists: grep rotations/azim in backends): scalar (default 2) = total rotations spread uniformly over all frames (existing spin behavior parity); for morph a LIST of len 2N-1 maps rotations per segment ([hold1, trans1, hold2, ...]); list with any other animate mode → ValueError; wrong length → ValueError naming expected 2N-1.
- Camera azimuth accumulates across segments (continuous rotation, no jumps); per-segment rotation r_k spread over that segment's frames.
- `surface=True` + morph: per-frame hull recompute of the morphing cloud (reuse existing per-frame surface machinery); surface color follows the interpolated color. Static (untagged) datasets keep their static surfaces.
- trails flags with morph → warn + ignore (same pattern as spin/serial, naming datasets).
- duration/frame-rate: reuse existing animation kwargs (`duration`, `framerate`... CHECK existing names) to derive n_steps per segment: total_frames = duration*framerate split evenly across 2N-1 segments.
- plotly: go.Frames with scatter coords + camera eye rotating (like existing plotly spin); surface Mesh3d per frame if surface on.
- Tests: schedule math (segments/frames), Hungarian matching deterministic + optimal on a known toy (2 clouds where optimal assignment is known), frame content at hold vs mid-morph (exact interpolation values), color interpolation endpoints, rotations list honored (azim at segment boundaries matches cumulative sums), ValueErrors (list length, <2 morph datasets, rotations list with animate=True, mixed list modes), plotly frame counts + camera eyes, surface+morph smoke (mesh differs across frames), trails warning.
- Evidence: 6-frame grids (holds + mid-morphs) BOTH backends, 3 toy blobs with different colors → docs/images/v1.0-seven-features/morph_anim_{mpl,plotly}.png. VIEW: point clouds morph, colors blend, camera rotates per rotations=[1,0.25,2,0.25,1], no box escape. Commit.

### Task M3: Demos + docs migration
**Files:** examples/plot_shape_morph.py, examples/animate_surface_morph.py, plot() docstring, README, docs regeneration (controller runs the build).
- plot_shape_morph.py → `hyp.plot(clouds, animate='morph', rotations=[...], ...)` — the manual Hungarian/FuncAnimation code collapses to a few lines (keep the shapes-zoo loading/normalizing prose). Keep gallery-tractable frame counts.
- animate_surface_morph.py → same + `surface=True` (+`keep_points` styling); manual per-frame mesh code removed.
- Both examples must still produce captured videos (figure registration — verify via the runpy fignums check; hyp.plot animate path keeps figures registered already).
- Docstring: full animate='morph' + rotations documentation. README bullet.
- Verify each example runs headless; extract ≥4 frames per example + VIEW (tight hulls, no escapes).
- Tests: suite green. Commit.

### Task M4: Visual review + evidence + PR
- Fresh visual reviewer agent: morph animation frames both backends (holds/mid-morphs/rotations list behavior), tight-hull static configs re-check (blob/cube/5-pt), surface+morph. Fix findings.
- Controller: docs rebuild (gallery re-executes both morph examples), full suite, push, CI 12/12, PR #272 comment (before/after tightness frames + morph grids both backends + numbers: slack %, containment %, explosion regression) + body §6 update.
