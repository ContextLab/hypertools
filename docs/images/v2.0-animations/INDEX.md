# Animation export evidence (hypertools 2.0)

Sample animations exported via `hyp.plot(..., save_path=...)`; the extension
picks the format (.gif / .png [APNG] / .mp4 / .svg [SMIL-animated vector]).

| case | file | notes |
|-|-|-|
| matplotlib window | matplotlib_window.gif | sliding window + camera rotation |
| matplotlib spin | matplotlib_spin.gif | camera spin |
| plotly window | plotly_window.gif | rotates while window advances (parity) |
| plotly spin | plotly_spin.gif | camera spin |
| hyperaligned weights | weights_hyperaligned.gif | reconstruction of the classic readthedocs hypertools.gif: hyp.plot(weights, align='hyper', animate=True, zoom=2.5) |
