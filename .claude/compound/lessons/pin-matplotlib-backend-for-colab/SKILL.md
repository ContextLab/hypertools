---
name: pin-matplotlib-backend-for-colab
description: Use when hypertools code or a notebook uses a matplotlib-only return API (fig.axes, .canvas, HyperAnimation .draw_frame/.on_frame/.n_frames, an internal hyp.plot call).
created: 2026-10-03
origin: notes kept in CLAUDE.md
---
Colab and Kaggle auto-select the PLOTLY render backend, so matplotlib-only return APIs
break there while passing locally (plot_stream's head plot, the tour's animation clock,
launch notebooks). Pin backend='matplotlib' at such call sites and test them under
hyp.set_interactive_backend('plotly'), the same preference Colab sets.
