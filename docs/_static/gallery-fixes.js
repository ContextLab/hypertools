// JavaScript to fix gallery thumbnail linking
document.addEventListener('DOMContentLoaded', function() {
    // Get all thumbnail containers
    const thumbnails = document.querySelectorAll('.sphx-glr-thumbcontainer');
    
    thumbnails.forEach(function(container) {
        // Find the hidden xref span that contains the reference
        const xrefSpan = container.querySelector('.xref');
        if (xrefSpan) {
            // Extract the reference text (e.g., "sphx_glr_auto_examples_plot_basic.py")
            const refText = xrefSpan.textContent.trim();
            
            // Convert to proper HTML filename
            let htmlFile = refText.replace('sphx_glr_auto_examples_', '').replace('.py', '.html');
            
            // Add click handler to the entire container
            container.addEventListener('click', function() {
                window.location.href = htmlFile;
            });
            
            // Also add click handler to the image
            const img = container.querySelector('img');
            if (img) {
                img.addEventListener('click', function(e) {
                    e.preventDefault();
                    window.location.href = htmlFile;
                });
            }
        }
    });
});