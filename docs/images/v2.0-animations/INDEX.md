# Animation export evidence (hypertools 2.0)

Sample animations exported via `hyp.plot(..., save_path=...)`; the extension
picks the format (.gif / .png [APNG] / .mp4 / .svg [SMIL-animated vector]).

| case | file | notes |
|-|-|-|
| matplotlib window | matplotlib_window.gif | sliding window + camera rotation |
| matplotlib spin | matplotlib_spin.gif | camera spin |
| plotly window | plotly_window.gif | rotates while window advances (parity) |
| plotly spin | plotly_spin.gif | camera spin |
| hyperaligned weights | weights_hyperaligned.gif | reconstruction of the classic readthedocs hypertools.gif: hyp.plot(weights, normalize='across', align='hyper' [n_iter=10 repeated hyperalignment], animate=True, zoom=3.5, rotations=1, frame_rate=50, linewidth=3) |
| shapes zoo morph (full) | shapes_morph.mp4 | 7 shapes, Hungarian point matching, smooth morphs; 3510 frames @30fps, 117s, 13 rotations (scripts/generate_shape_morph.py) |
| shapes zoo morph (preview) | shapes_morph_preview.gif | first 20s of the full morph |
