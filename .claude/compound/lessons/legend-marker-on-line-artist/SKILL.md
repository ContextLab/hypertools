---
name: legend-marker-on-line-artist
description: Use when a hypertools draw path puts a line's markers on a separate artist (fmt split, truth= overlay).
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
A line whose markers are drawn on a SEPARATE artist shows a marker-less legend glyph. Keep
marker= on the line artist with markevery=[] so the legend handle carries it, and check
every marker-splitting draw path for this.
