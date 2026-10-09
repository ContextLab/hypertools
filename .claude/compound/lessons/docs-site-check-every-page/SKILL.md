---
name: docs-site-check-every-page
description: Use when asked to build the hypertools Sphinx / Read the Docs site and confirm it looks correct, or when reviewing built docs pages in a browser.
created: 2026-10-09
origin: project hypertools, session f69d921d
---
Check EVERY built page with a scripted browser, never a hand-picked list. Build the RTD
way (from docs/: READTHEDOCS=True READTHEDOCS_GIT_IDENTIFIER=master READTHEDOCS_OUTPUT=$OUT
sphinx -T -b html -d _build/doctrees . $OUT/html, then python docs/post_build.py), serve
$OUT/html with `python -m http.server`, and run check_built_site.py (beside this lesson;
playwright is in .venv; see also scripts/verify_docs_playwright.py). It reads each page's
::before/::after content for ERROR/WARNING, broken images, console errors, leaked RST and
overflow at 1400px and 400px, and saves screenshots to look at.
Why: a sampled review passed the site while docs/hierarchy.html showed furo's red
"ERROR: Adding a table of contents" box from a `.. contents::` directive. That box is CSS
::before text, so sphinx -W (0 warnings), HTML text scans and link checkers cannot see it.
Before trusting 0 findings, plant a defect on a temporary copy of a page and confirm the
checker reports it.
