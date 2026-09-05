# Vendored font

`NotoSans-Regular.ttf` and `NotoSans-Bold.ttf` — Noto Sans Regular and Bold,
both Version 2.008, from the Google Noto project
(<https://github.com/googlefonts/noto-fonts>):

- Regular: <https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf>
  (sha256 `b85c38ecea8a7cfb39c24e395a4007474fa5a4fc864f6ee33309eb4948d232d5`)
- Bold: <https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf>
  (sha256 `c976e4b1b99edc88775377fcc21692ca4bfa46b6d6ca6522bfda505b28ff9d6a`)

Licensed under the **SIL Open Font License, Version 1.1** (see `OFL.txt`),
which permits redistribution as part of this package.

Why it is bundled: matplotlib ships only DejaVu Sans, whose look is dated and
whose coverage is uneven. Bundling these small (~570 KB each) faces gives
hypertools a consistent, good-looking sans-serif on EVERY platform — no "font
not found" fallbacks and no per-machine variation in rendered output — while
remaining a tiny fraction of the install. The Bold face (added GH #285) makes
`fontweight='bold'` resolve to a real bold face instead of silently falling
back to Regular. Broader scripts (CJK, emoji, Indic) are far too large to
bundle; those still resolve through the per-glyph fallback stack built in
`hypertools/plot/fonts.py` (system pan-Unicode fonts, then DejaVu Sans).
