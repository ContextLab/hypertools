# 2026-10-09 -- built-docs-site defect fixes (branch fix/1.1-release-review)

Scope: docs/, examples/, tests/ only (hypertools/ untouched). Not pushed.

Fixed (all verified in a browser on the RTD-parity build, 1400px / 400px / dark):
- Gallery cards: title is now the visible link to the example page (whole card
  clickable, thumbnail still -> Colab). custom.css used to `display:none` the
  only example-page link. Tooltip was rendered twice (ours + sphinx-gallery's);
  now one, theme-coloured, below the card.
- Duplicate videos: `docs/_gallery_scrapers.py` (one scraper = sphinx-gallery's
  matplotlib scraper + show=False HyperAnimations, each once). Before/after
  videos: animate 5->2, animate_trails 5->2, animate_spin / animate_surface_morph
  / animate_trails_mix / plot_story_trajectories / save_movie 2->1.
- Duplicate plotly figure (plot_interactive_backend 2->1): `ignore_repr_types`
  (hyp.plot returns HyperPlotlyFigure, so the regex names that class too).
- Dark theme: card title / tooltip / link-hover colours use furo variables.
- Phone: plotly figures refit to the column width (gallery-fixes.js), CSS
  scroll fallback.
- tutorials/plot.html console: post_build strips plotly's dead ES-module CDN
  import (403) and its MathJax 2.7.5 <script> (clashes with the page's MathJax 4).
- Tooltips: post_build strips RST backticks/roles (15 cards).
- `remove_config_comments`, explore-mode warning filtered for the gallery build
  (reset_modules hook) + a sentence in examples/explore.py saying the image is static.
- API page H1s keep their case (`section:has(> dl.py) > h1`).

Left alone / to decide:
- tutorials.rst shows 200px GIFs with `:width: 400` (rst option, not CSS).
- plot_impute / plot_missing_data "Out:" blocks show real data warnings that
  include the build machine's path to the example.
- Built site has no link to CHANGELOG.md (README.md and pyproject do).
- Incremental rebuild note: sphinx-gallery only re-runs an example when its
  `docs/auto_examples/<name>.py.md5` is stale; delete that file to force it.
