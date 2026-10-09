// Gallery thumbnail navigation is handled at build time: docs/post_build.py
// wraps each thumbnail <img> in an anchor that opens the example's notebook
// on Colab (the title text below each thumbnail opens the example page).
// The previous runtime click-handler here targeted sphinx-gallery <= 0.16
// markup (.xref spans) that no longer exists, so clicks silently did
// nothing.

// plotly figures are drawn at a fixed pixel size (640x480 by default) inside
// a wrapper <div style="width:640px; height:480px">. On a phone the content
// column is narrower than that; custom.css caps the wrapper at the column
// width and lets the figure scroll inside it, but a drag on a 3D figure
// rotates it rather than scrolling, so half the figure stayed out of reach.
// Redraw each figure at the width it actually has (same aspect ratio), and
// back at its natural size when there is room again. Desktop is untouched:
// the wrapper is as wide as the figure there, so nothing is relaid out.
(function () {
    function fitPlotlyFigures() {
        if (!window.Plotly) { return; }
        document.querySelectorAll('.plotly-graph-div').forEach(function (gd) {
            var wrap = gd.parentElement;
            var full = gd._fullLayout;
            if (!wrap || !full || !full.width || !full.height) { return; }
            if (!gd.dataset.hypNaturalSize) {
                gd.dataset.hypNaturalSize = full.width + 'x' + full.height;
            }
            var natural = gd.dataset.hypNaturalSize.split('x').map(Number);
            var width = Math.min(natural[0], wrap.clientWidth);
            if (width < 50 || Math.abs(width - full.width) < 1) { return; }
            var height = Math.round(natural[1] * width / natural[0]);
            wrap.style.height = height + 'px';
            window.Plotly.relayout(gd, {width: width, height: height});
        });
    }

    var pending = null;
    function schedule() {
        clearTimeout(pending);
        pending = setTimeout(fitPlotlyFigures, 150);
    }
    window.addEventListener('load', function () {
        fitPlotlyFigures();
        // figures whose plotly.js arrives after `load` (slow CDN)
        setTimeout(fitPlotlyFigures, 1000);
        setTimeout(fitPlotlyFigures, 4000);
    });
    window.addEventListener('resize', schedule);
}());
