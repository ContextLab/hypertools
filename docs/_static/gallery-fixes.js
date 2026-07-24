// Gallery thumbnail navigation is handled at build time: docs/post_build.py
// wraps each thumbnail <img> in an anchor that opens the example's notebook
// on Colab (the title text below each thumbnail opens the example page).
// The previous runtime click-handler here targeted sphinx-gallery <= 0.16
// markup (.xref spans) that no longer exists, so clicks silently did
// nothing.
