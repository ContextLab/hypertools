# Gallery dual-backend audit

| example | backend | status | error |
|-|-|-|-|
| analyze | matplotlib | PASS |  |
| analyze | plotly | PASS |  |
| animate_MDS | matplotlib | PASS |  |
| animate_MDS | plotly | PASS |  |
| animate | matplotlib | PASS |  |
| animate | plotly | PASS |  |
| animate_plotly | matplotlib | PASS |  |
| animate_plotly | plotly | PASS |  |
| animate_spin | matplotlib | PASS |  |
| animate_spin | plotly | PASS |  |
| chemtrails | matplotlib | PASS |  |
| chemtrails | plotly | PASS |  |
| explore | matplotlib | PASS |  |
| explore | plotly | PASS |  |
| plot_2D | matplotlib | PASS |  |
| plot_2D | plotly | PASS |  |
| plot_PPCA | matplotlib | PASS |  |
| plot_PPCA | plotly | PASS |  |
| plot_TSNE | matplotlib | PASS |  |
| plot_TSNE | plotly | PASS |  |
| plot_UMAP | matplotlib | PASS |  |
| plot_UMAP | plotly | PASS |  |
| plot_align | matplotlib | PASS |  |
| plot_align | plotly | PASS |  |
| plot_apply_model | matplotlib | PASS |  |
| plot_apply_model | plotly | PASS |  |
| plot_basic | matplotlib | PASS |  |
| plot_basic | plotly | PASS |  |
| plot_clusters2 | matplotlib | PASS |  |
| plot_clusters2 | plotly | PASS |  |
| plot_clusters3 | matplotlib | PASS |  |
| plot_clusters3 | plotly | PASS |  |
| plot_clusters | matplotlib | PASS |  |
| plot_clusters | plotly | PASS |  |
| plot_corpus | matplotlib | PASS |  |
| plot_corpus | plotly | PASS |  |
| plot_dataframe | matplotlib | PASS |  |
| plot_dataframe | plotly | PASS |  |
| plot_datasaurus | matplotlib | PASS |  |
| plot_datasaurus | plotly | PASS |  |
| plot_describe | matplotlib | PASS |  |
| plot_describe | plotly | PASS |  |
| plot_digits | matplotlib | PASS |  |
| plot_digits | plotly | PASS |  |
| plot_geo | matplotlib | PASS |  |
| plot_geo | plotly | PASS |  |
| plot_hue | matplotlib | PASS |  |
| plot_hue | plotly | PASS |  |
| plot_interactive_backend | matplotlib | PASS |  |
| plot_interactive_backend | plotly | PASS |  |
| plot_labels | matplotlib | PASS |  |
| plot_labels | plotly | PASS |  |
| plot_legend | matplotlib | PASS |  |
| plot_legend | plotly | PASS |  |
| plot_missing_data | matplotlib | PASS | (rerun after fmt-list interp fix) |
| plot_missing_data | plotly | PASS | (rerun after fmt-list interp fix) |
| plot_mixture_models | matplotlib | PASS |  |
| plot_mixture_models | plotly | PASS |  |
| plot_multicolored_lines | matplotlib | PASS |  |
| plot_multicolored_lines | plotly | PASS |  |
| plot_nested_lists | matplotlib | PASS |  |
| plot_nested_lists | plotly | PASS |  |
| plot_normalize | matplotlib | PASS |  |
| plot_normalize | plotly | PASS |  |
| plot_procrustes | matplotlib | PASS | (rerun after fmt-list interp fix) |
| plot_procrustes | plotly | PASS | (rerun after fmt-list interp fix) |
| plot_shapes_zoo | matplotlib | PASS |  |
| plot_shapes_zoo | plotly | PASS |  |
| plot_sotus | matplotlib | PASS |  |
| plot_sotus | plotly | PASS |  |
| plot_text | matplotlib | PASS |  |
| plot_text | plotly | PASS |  |
| precog | matplotlib | PASS |  |
| precog | plotly | PASS |  |
| save_image | matplotlib | PASS |  |
| save_image | plotly | PASS |  |
| save_movie | matplotlib | PASS |  |
| save_movie | plotly | PASS | completes in 1943s (kaleido renders all 600 mp4 frames; see notes) |

Notes:
- matplotlib PNGs for animation examples show the pre-animation (empty)
  frame -- a harness snapshot artifact, not a rendering failure; animation
  output is verified separately (test_animation_export.py + docs mp4 embeds).
- save_movie/plotly timed out at 900s due to kaleido per-frame export speed
  (600 frames at duration=30); completion verified with a longer timeout: 1943s.
