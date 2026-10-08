# Session 2026-10-08: finishing the 1.1 final review (visual sign-off round)

Branch `fix/1.1-release-review`; PR #286 still OPEN and unmerged. Nothing merged, tagged or published.

## State at session start
- PR head = c433b8f0 (unchanged since 2026-09-12). Local branch had 6 unpushed notes commits
  (incl. c7dea477 by Jeremy's 2026-10-03 session moving notes into .claude/compound lessons).
- RTD: still no build since 2026-07-24 (API v2 confirms). GitHub hook 11883014 -> readthedocs
  legacy URL, no secret; deliveries list empty (retention). Needs RTD admin re-sync. MANUAL.
- CI rerun (attempt 2) of run 34674962598 at c433b8f0 started for dependency drift.

## Done this session
- Headless tour at c433b8f0 with current deps: 241 PASS / 3 SKIP / 0 FAIL; 74 visual cases.
- Built a review artifact (https://claude.ai/artifact/UPKL2S8yGd6VaAp9i5adm5) from the executed
  tour (scratchpad gallery/extract.py; plotly animations -> mp4 via gallery/plotly_anim_mp4.py);
  verdicts stored in its db collection `verdicts` (read with ArtifactData list).
- Jeremy's verdicts: 54 pass / 20 flag. Flags -> actions:
  - plotly animations didn't play in page -> page now plays frame-by-frame mp4s.
  - ANIM-clock one curve, COLOR-luminance shows input not palette, TEXT-* unreadable,
    PANEL-models subplots title "example is wrong" -> tour cases changed (b240258a).
  - ANIM-forecast truth shown from frame 0 -> DECIDED: truth point appears when a forecast reaches it.
  - RED-describe 'average' was pooled stack -> DECIDED: mean of individual curves; pooled kept as
    result['pooled']; integer ticks; describe(show=False) returns fig.
  - PLOT-scale-matplotlib grid -> plotly (no grid) is correct; remove matplotlib grid.
  - GUI-native: no Qt binding locally; remains manual/optional.
- Agent visual pre-review found 2 real plotly bugs (Jeremy had passed both): nested-list leaf
  names 1,2,2,4 and 3-D markers at half size -> fixed a4b146af (full suite 6583/0).
- Decisions: forecast_cluster/hue default colours offset past observed colours (approved);
  raise floors (183de73c); XOM CIK pin + HON note (8d02cb81); describe/t= docs + anim size
  note (approved); axis_scale midrange DEFERRED to 1.2 (open issue); set_autoinstall KEEP.
- _aa_x (plotly 1-D forecast branch) proven unreachable (1-D animation refused at
  plotly_backend.py:1630); 528-call settrace sweep, line 5788 hit 0 times.
- Bluesky clips re-rendered at c433b8f0 (gitignored, notes/bluesky-launch/).

## Pending
- Integrate worktree agents: truth timing + cluster colours; describe + grid + docstrings.
- Regenerate media: market_sectors tutorial mp4 + thumb (XOM data), anything else touched.
- Full verification: pytest, ruff, sphinx -W, doctest, tutorials, headless tour, push, CI, Colab.
- Round-2 visual review of changed cases only (carry forward pass for pixel-identical renders).
- Open GitHub issue for axis_scale midrange (1.2).
