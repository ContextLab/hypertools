---
name: plotly-scene-sized-by-height
description: Use when sizing or calibrating a Plotly 3-D scene or its camera distance in hypertools.
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
Plotly sizes a 3-D scene by its domain HEIGHT only and clips at the sides: the same
267x209 px cube appears in 600x300 and 1200x300 scenes, and a 300x600 scene's cube is 415
tall, cut at 300 wide. At hypertools' default eye the cube is 0.89 x height wide and 0.70
x height tall, and apparent size goes as 1/eye-distance only when backing OFF (an eye
nearer than about 0.8x clips the front). Name a calibration constant by its reference:
SCENE_CUBE_WIDTH_PER_HEIGHT = 1.4 means width per CUBE height, and reading it as width per
SCENE height backs the camera off 1.4x in every square panel cell.
