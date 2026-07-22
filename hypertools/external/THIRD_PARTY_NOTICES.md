# Third-party notices

HyperTools is distributed under the MIT License (see the top-level `LICENSE`).
It **bundles** the following third-party components, each retained under its own
license. This file records their provenance, copyright, license, and the
modifications HyperTools made, as required for redistribution.

---

## Apache License 2.0 components

A verbatim copy of the Apache License, Version 2.0 is bundled next to this file
as **`LICENSE-APACHE-2.0.txt`** and applies to both components below.

### 1. BrainIAK — Shared Response Model (`brainiak.py`)

- **Upstream:** https://github.com/brainiak/brainiak
- **Copyright:** Copyright 2016 Intel Corporation
- **License:** Apache License, Version 2.0
- **Modifications (HyperTools, Contextual Dynamics Lab):** the Shared Response
  Model implementation (SRM / DetSRM family) was **extracted** from BrainIAK and
  **adapted** into a single standalone vendored module so HyperTools does not
  depend on the full `brainiak` package (import paths and packaging adjusted).
- **Upstream `NOTICE` file:** none — the BrainIAK repository ships no `NOTICE`
  file (checked 2026-07; `NOTICE`/`NOTICE.txt` return HTTP 404), so there are no
  additional attribution notices to reproduce under Apache §4(d).

### 2. pca-magic — Probabilistic PCA (`ppca.py`)

- **Upstream:** https://github.com/allentran/pca-magic
- **Copyright:** Copyright 2015 Allen Tran
- **License:** Apache License, Version 2.0
- **Modifications (HyperTools, Contextual Dynamics Lab):** the Probabilistic PCA
  implementation was **vendored** and **adapted** as a standalone module,
  replacing the unmaintained `pca-magic` PyPI dependency (import/packaging
  adjustments; behavior unchanged).
- **Upstream `NOTICE` file:** none — the pca-magic repository ships no `NOTICE`
  file (checked 2026-07; returns HTTP 404).

---

## SIL Open Font License 1.1 component

### Noto Sans (`fonts/NotoSans-Regular.ttf`)

- **Upstream:** Google Noto Fonts (https://github.com/notofonts/latin-greek-cyrillic)
- **License:** SIL Open Font License, Version 1.1 — the full text ships as
  `fonts/OFL.txt`; provenance is documented in `fonts/README.md`.
- **Modifications:** none (the font file is bundled unmodified).
