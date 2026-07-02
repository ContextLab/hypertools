# Animation export evidence (hypertools 2.0)

Sample animations exported via `hyp.plot(..., save_path=...)`; the extension
picks the format (.gif / .png [APNG] / .mp4 / .svg [SMIL-animated vector]).

| case | file | notes |
|-|-|-|
| matplotlib window | matplotlib_window.gif | sliding window + camera rotation |
| matplotlib spin | matplotlib_spin.gif | camera spin |
| plotly window | plotly_window.gif | rotates while window advances (parity) |
| plotly spin | plotly_spin.gif | camera spin |
| story trajectories (weights) | weights_hyperaligned.gif | the classic readthedocs hypertools.gif pipeline, from the 2020 pieman_trajectory_demo notebook: gaussian temporal smoothing (var=300) -> hyp.align(align='SRM', n_iter=20) -> smooth again -> reduce='UMAP' -> animate (scripts/generate_weights_trajectory.py) |
| shapes zoo morph (full) | shapes_morph.mp4 | 7 shapes, Hungarian point matching, smooth morphs; 3510 frames @30fps, 117s, 13 rotations (scripts/generate_shape_morph.py) |
| shapes zoo morph (preview) | shapes_morph_preview.gif | first 20s of the full morph |
