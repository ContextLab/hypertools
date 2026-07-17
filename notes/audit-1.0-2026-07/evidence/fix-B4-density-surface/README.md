# Fix evidence — batch B4-density-surface (unit F07-plot-density-surface)

Branch `audit/release-1.0-2026-07`, 2026-07-17. All screenshots are real
Chromium (Playwright) renders of `write_html` output (plotly) or Agg
`savefig` output (matplotlib); no kaleido.

## F07-001 (major): plotly surface= NaN'd out enclosed points

- `before_f001_default.png` / `before_f001_plotly_surface_default.html`:
  pre-fix `hyp.plot([a, b], '.', surface=True, backend='plotly')` — opaque
  pastel hulls, 299/300 data points NaN'd out (measured non-NaN: [1, 0]),
  mesh opacity 1.0.
- `after_f001_default.png` (alpha=0.6 default) and `after_f001_a02.png`
  (alpha=0.2): post-fix — every point kept (non-NaN [150, 150]), meshes
  genuinely translucent (per-layer opacity 1−sqrt(1−alpha) over the doubled
  winding = 0.3675 / 0.1056), points clearly visible through the hulls.
- `after_f001_plotly_surface_a10.html`: alpha=1.0 keeps the historical
  artifact-free opaque path (non-NaN [1, 0], opacity 1.0) — enclosed points
  would be invisible behind the opaque mesh and drawing them punches WebGL
  holes.
- `mpl_reference_a06.png`: matplotlib reference for the same scene/alpha.
- `design_v1_translucent_a02.png` / `design_v4_translucent_a06.png`:
  design-selection renders (V4 = per-layer 1−sqrt(1−alpha) chosen; it
  matches the requested total alpha exactly and halves the per-layer
  depth-sort speckle contrast vs. naive opacity=alpha).

## F07-005 (minor): animated surfaces dropped per-vertex hue coloring

- `before_f005_mpl_spin_hue_surface_frame0.png`: pre-fix spin frame — hull
  around a rainbow-hue helix renders near-achromatic (max |r−g| across
  faces 0.115; static reference is 0.499).
- `after_f005_mpl_spin_hue_surface_frame0.png`: post-fix — hull tinted per
  vertex by the local hue (max |r−g| 0.499, 13693/13693 unique face colors,
  matching the static plot exactly).
- plotly (numeric, from frame vertexcolor arrays): before max|r−g|/255 =
  0.059 → after 0.518 with 10248 unique vertex colors in frame 1.
