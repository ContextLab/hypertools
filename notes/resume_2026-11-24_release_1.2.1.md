# Resume notes: HyperTools 1.2.1 (PR #300), written 2026-11-24 about 05:45 EST

Read this first in a fresh session. Details per review round are in
`notes/session_2026-11-22_feat295-animated-panels.md`.

## Where things stand

- 1.2.0 is released (PyPI, GitHub, Read the Docs, conda-forge). `master` ==
  `v1.2.0` == 8086b5b8. Nothing for 1.2.1 is tagged, uploaded or on master.
- PR #300 (draft), branch `post-1.2.0`, is the 1.2.1 work.
- Pushed head: 54b11252. Local full suite on it: 9239 passed, 24 skipped
  (matplotlib 3.10). The related label, legend and layout files also pass
  on matplotlib 3.11.2 (2496 passed on each version). CI on 54b11252 was
  running when this was written; CI on the previous head 7a5f997e was
  green (16 test and gate jobs; Read the Docs red as expected).
- A commit after 54b11252 holds only these notes and a path fix in the
  session notes. See "Unpushed" at the end.

## What PR #300 contains

- GH #299: trails and chemtrails with marker formats; `head_marker=`,
  `tail_linewidth=`, `trail_alpha=`, `hold=`, `caption_kwargs=`,
  `suptitle=` / `suptitle_kwargs=`.
- GH #297: animated cube and its axis labels inside the canvas and off the
  legend on every frame; crowded `labels=` moved off each other and off
  legends, colorbars, titles, captions; continuous `hue=` on animated
  markers; half `fillstyle=` and `markerfacecoloralt=` on hue markers;
  named 2-D legend `loc` inside the frame; example edits.
- Animated datapoint labels planned over all frames (no wobble).
- `labels=` on streaming plots (causal placement).
- `legend_colors=` pair keys take their dataset's line styling (a defect
  that 1.1.0 and 1.2.0 share).
- Version 1.2.1, CHANGELOG, README image links `v1.2.1`.

## The owner's decisions, in his words (2026-11-23)

- Labels in animations: "for axis labels, they should move with the
  rotations so they remain aligned with the axes. same for labels of
  individual datapoints (e.g., in every frame the labels should have
  correct positioning). for static text like legends, titles, and other
  annotations: compute placement once based on the *worst case* rotations
  of the axes, and then place once so that the static text (and other
  static content) remains stable across frames."
- Wobble: "slight shifts between frames are OK but not if they are
  wobbling back and forth so that it looks glitchy. movement needs to be
  in a consistent direction". Also: "let's see how it looks".
- Streams: asked "What about streaming animations?"; of three options he
  chose 2, add streaming labels to 1.2.1 (relayed by his coordinating
  session as "hypertools: 2").
- `hold=` shorter than one frame: "Warn, add no frame".
- `markerfacecoloralt=` on hue markers: "Keep it".
- Argument names (`head_marker`, `tail_linewidth`, `trail_alpha`, `hold`,
  `caption_kwargs`, `suptitle`): "Names are fine".
- All of these except the streams question reached this session through
  his coordinating session's question dialogs.

## Waiting on the owner

Update 2026-11-24 about 06:20 EST: he looked at the review page (version
2, built from 54b11252) and wrote, in this session: "The new figures and
animations look great!" Item 1 below is answered. That is approval of the
pictures, not the go for release. He then answered three questions in a
dialog in this session:
- Cube floor: "Keep 55% (Recommended)". Item 2 is settled.
- Streams: "Add the three now", meaning `legend=`, `caption=` and
  `suptitle=` on streaming plots go into 1.2.1. Work started 06:30 EST in
  the worktree of branch `worktree-agent-a054066ab320401bb`; see "Status
  at wind-down" for how far it got. `head_marker=`, `tail_linewidth=` and
  `trail_alpha=` on streams were not asked for.
- Release: "Prepare, then ask me", meaning after the stream arguments are
  in and green: final gallery from the last commit, publish
  `docs-gallery-v1.2.1` and `docs-notebooks/v1.2.1/`, record commit, dist,
  draft GitHub release, mark PR #300 ready; then STOP and ask him before
  the master fast-forward, the tag and PyPI.

1. His look at the review page (private to him):
   https://claude.ai/artifact/AYEuVQguA8jJEB9jq2yWC6
   It was built from 7a5f997e. Things he may object to: long leader lines
   in animated labels; 30 crowded labels still overlap on about half the
   frames; stream labels that end up close together.
2. Cube floor: with a very tall corner legend the animated cube may now
   shrink to 55% (was 70%); needed on matplotlib 3.11, where text lines
   are taller. An existing test pinned 70% and was changed to 55%. His
   call whether to keep 70% and accept the overlap.
3. Whether streams should also take `legend=`, `caption=`, `suptitle=`
   (small), `head_marker=` (medium), `tail_linewidth=` / `trail_alpha=`
   (large). Not built.
4. A tutorial sentence about LSL discovery of two outlets on one macOS
   machine (his to decide; none written).
5. Go for release. Nothing is released without it.

## Known limits (reported to him)

- 30 crowded animated labels: 1 to 3 overlapping pairs on about half the
  frames (0 for 8 and 12 labels); a little worse on matplotlib 3.11.
- Grid whose panels have unequal label counts: up to 3 overlapping pairs
  on most frames under 3.11 (up to 2 on a few frames under 3.10). Not yet
  improved.
- Stream labels are causal: they react, they cannot move early. They
  cannot be dragged and ignore `font=`.
- `plot()` with animated labels plans up front: about 0.5 s for 20 labels,
  1.8 s for 60.
- The LSL test that failed in full-suite runs before the 2026-11-23 crash:
  cause only partly known (see the session notes).

## Exact next steps

1. Check CI on the pushed head (`gh pr checks 300`).
2. Gallery: a local rebuild from 54b11252 and a comparison with 1.2.0 is
   in `~/hypertools-review-1.2.1/` (`after/`, `before/`, `pairs/`,
   `comparison.json`; the 7a5f997e build is under `build_7a5f997e/`).
   The status of that rebuild is in the last section of this file. Update
   the review page from it (page source: `~/hypertools-review-1.2.1/page/`,
   republish to the same artifact URL after reading it).
3. Apply the owner's feedback on the page; each code change means: related
   tests on matplotlib 3.10 AND 3.11, full suite, push, CI.
4. Release only on his explicit go, per `RELEASE_CHECKLIST.md` ("The
   1.2.1 release: read this first"): rebuild the gallery from the FINAL
   commit, publish `docs-gallery-v1.2.1` and `docs-notebooks/v1.2.1/`,
   record commit in `docs/prebuilt_gallery.json`, build dist, draft
   release, fast-forward master, tag, PyPI, conda-forge. Read the Docs on
   the PR stays red until the gallery is published.
5. On merge: comment on and close GH #297 and #299 with factual
   descriptions only.

## Traps

- The local venv has matplotlib 3.10; CI uses 3.11 on Python 3.11+. Test
  both before pushing. The 3.11 copy used on 2026-11-23 lived in a
  temporary folder; rebuild it with
  `uv pip install --target <dir> "matplotlib==3.11.*"` if it is gone.
- `scripts/add_colab_install_cell.py` must never be run off master, not
  even with `--help`.
- Temporary folders do not survive a reboot. Keep review material under
  `~/hypertools-review-1.2.1/`.
- Two agent worktrees exist under `.claude/worktrees/` with branches
  `worktree-agent-a054066ab320401bb` and
  `worktree-agent-abf3e80dedb9e2214`; both are fully merged into
  `post-1.2.0` and can be removed.
- The fetch for the comparison added the local ref
  `origin/docs-gallery-v1.2.0`.

## Status at wind-down

Written 2026-11-24 06:10 EST.

- Gallery rebuilt locally from 54b11252: 52 of 52 examples, 0 warnings.
  Against 1.2.0: 116 of 141 files byte-identical, 25 files (10 figures)
  changed, none worse than 1.2.0. `plot_missing_data_001`, `_002` and
  `plot_multiindex_001` are byte-identical to 1.2.0 again. The market
  legend keys match their paths.
- The review page (version 2) shows that build.
- Not explained: `plot_labels_001` differs between the 7a5f997e build and
  the 54b11252 build in 0.0013% of pixels (none by more than 8/255;
  labels in the same places). Probably the static cube zoom fix of
  54b11252, which touches every static 3-D figure; not verified. Check it
  (render the example on both commits and diff) before release.
- Still showing a few label boxes on data points:
  `animate_painting_embeddings`.
- No gallery was published; no `docs-gallery-v1.2.1`, no
  `docs-notebooks/v1.2.1/`.
- CI on 54b11252: see `gh pr checks 300`. If it was still running at
  suspend, nobody was watching it.

## Unpushed

The commit that adds this file (and changes one path in the session
notes) may be local only. `git status -sb` shows whether `post-1.2.0` is
ahead of `origin/post-1.2.0`. It changes no code; pushing it starts a new
CI run.
