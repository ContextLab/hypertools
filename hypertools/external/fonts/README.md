# Vendored font

`NotoSans-Regular.ttf` — Noto Sans Regular, Version 2.008, from the Google
Noto project (<https://github.com/googlefonts/noto-fonts>).

Licensed under the **SIL Open Font License, Version 1.1** (see `OFL.txt`),
which permits redistribution as part of this package.

Why it is bundled: matplotlib ships only DejaVu Sans, whose look is dated and
whose coverage is uneven. Bundling one small (~570 KB) face gives hypertools a
consistent, good-looking sans-serif on EVERY platform — no "font not found"
fallbacks and no per-machine variation in rendered output — while remaining a
tiny fraction of the install. Broader scripts (CJK, emoji, Indic) are far too
large to bundle; those still resolve through the per-glyph fallback stack built
in `hypertools/plot/fonts.py` (system pan-Unicode fonts, then DejaVu Sans).
