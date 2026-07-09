## Two more fixes: surface hue-per-vertex coloring + story-trajectories gif thumbnail

Follow-up to the release-hardening pass, addressing two reported issues. Same method:
reproduce → verify (screenshots + numeric) → fix → **independent red-team subagent** → re-verify.

---

### 1. Surface coloring: each hull vertex now takes a distance-weighted blend of its points' colors (`8e6ed682`)

**Before:** a `hue=`'d `surface=True` hull was painted **one flat color** — the *mean* of all its
points' hue colors. For a rainbow hue that mean is ~gray, so the surface ignored *where* each color
was (left image below).

**After:** each mesh **vertex** is colored by an inverse-distance-weighted (Shepard / IDW) blend of
the enclosed data points' colors — the nearest coordinates dominate, so the surface matches the hue
of the points it wraps (right image).

| Before (flat mean → gray) | After (per-vertex, distance-weighted) |
|-|-|
| ![before](https://raw.githubusercontent.com/ContextLab/hypertools/fix/qc-notes-2026-07/notes/fix-qc-notes-2026-07/evidence/surface/before_mpl.png) | ![after](https://raw.githubusercontent.com/ContextLab/hypertools/fix/qc-notes-2026-07/notes/fix-qc-notes-2026-07/evidence/surface/after_mpl.png) |

Implementation:
- `meshutil.vertex_colors_from_points(verts, points, point_colors, power=2)` — `(V,3)` IDW per-vertex
  RGB (weight `1/dist²`, exact color on a coincident point); `face_colors_from_vertex_colors` averages
  a triangle's 3 vertex colors for the matplotlib per-face path.
- `_blinn_phong_shade` already broadcasts a per-element base color, so per-vertex (plotly `vertexcolor`)
  / per-face (matplotlib) colors flow through the existing lighting unchanged; `_blend_toward_white` is
  now array-aware.
- `plot.py` bundles each surfaced dataset's `(points, per-point colors)` and threads it to both backends'
  static 3-D draw. No hue (or no surface) → the prior flat color, so non-hue surfaces are unchanged.

Numeric verification: matplotlib per-face colors track position; plotly `Mesh3d.vertexcolor` now holds
**8740 distinct colors** (was 1). Both backends agree spatially. **+16 tests** (IDW unit tests + spatial
correctness for both backends).

The independent red-team returned **SOLID** (IDW math correct; both backends agree spatially at
corr 0.61/0.66 for a `hue=z` gradient; matrix/categorical/continuous hue all work; `line_colors[i]` is
length-aligned to points by construction; no-hue surfaces unchanged; single-point / 2-D / animation all
fall back cleanly; 2000 points render in 0.13s/0.28s). Its one note — an explicit `surface={'color': ...}`
was silently overridden by hue — is fixed in **`afd7c07c`**: an explicit color now wins; hue only colors
surfaces that inherit their color.

---

### 2. Story-trajectories gallery thumbnail now animates (`b84f8276`)

**Root cause:** the story example already ships an 80-frame animated gif thumbnail
(`docs/_static/thumbnails/sphx_glr_plot_story_trajectories_thumb.gif`) and sets
`sphinx_gallery_thumbnail_path` to it, and the tutorial (`docs/tutorials.rst`, in the toctree) embeds
it — but `docs/post_build.py`'s `GIF_REPLACEMENTS` map, which swaps each animated example's static
`.png` thumbnail for its `.gif` in the built gallery HTML (a Read-the-Docs `post_build` job), **never
listed the story example**. So its gallery card showed a frozen still frame.

**Fix:** one entry added to `GIF_REPLACEMENTS`. Verified end-to-end against a real
`sphinx-build` + `post_build.py` run: `auto_examples/index.html` now references the story `.gif`
(**0** remaining `.png` refs), the gif is copied into `_images/`, and `tutorials.html` embeds it. The
example script, tutorial, toctree, and source gif were already correct; RTD regenerates
`auto_examples/` from source, so no committed-build change is needed.

The independent red-team returned **SOLID** for the story fix, and its completeness audit surfaced a
**second frozen thumbnail** (see below).

---

### 3. A second frozen thumbnail: `animate_surface_morph` (`5972fb4e`)

The story red-team audited *every* animated gallery example and found `animate_surface_morph` also
showed a frozen card: it ends with a static-figure tweak (an alpha-fade of the point layer), so
sphinx-gallery thumbnailed *that* still frame — and, unlike story, it shipped no gif at all. Fixed by
shipping a right-sized animated gif thumbnail (90 frames, 524 KB, subsampled from the 360-frame render),
pointing `sphinx_gallery_thumbnail_path` at it, and registering the swap. Verified end-to-end (index.html
now references the gif, gif copied to `_images/`).

Added **`tests/test_docs_thumbnails.py`** to keep `post_build.GIF_REPLACEMENTS` and the shipped
`_static/thumbnails/*.gif` set in lockstep — a shipped-but-unregistered (or registered-but-missing) gif,
the exact defect behind *both* story and surface_morph, now fails CI instead of silently freezing a card.

---

### Tests
Full suite on the final HEAD, all commits included (kaleido plotly-export tests excluded — pre-existing
headless deadlock, not a regression): **1466 passed, 4 skipped, 0 failed**.

**Branch is for review only — do not merge; base `dev-1.0-refactor`, `master` untouched.**
