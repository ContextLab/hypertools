#!/usr/bin/env python3
"""
Post-build script to copy custom GIF thumbnails and update HTML references.

This script should be run after sphinx-gallery builds the documentation
to replace PNG thumbnails with animated GIF thumbnails for specific examples.
"""

import os
import shutil
import re

# Base paths
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_THUMBS_DIR = os.path.join(DOCS_DIR, "_static", "thumbnails")
BUILD_IMAGES_DIR = os.path.join(DOCS_DIR, "_build", "html", "_images")
GALLERY_HTML = os.path.join(DOCS_DIR, "_build", "html", "auto_examples", "index.html")

# Mapping of PNG to GIF thumbnails that should be replaced
GIF_REPLACEMENTS = {
    "sphx_glr_chemtrails_thumb.png": "sphx_glr_chemtrails_thumb.gif",
    "sphx_glr_animate_MDS_thumb.png": "sphx_glr_animate_MDS_thumb.gif", 
    "sphx_glr_animate_spin_thumb.png": "sphx_glr_animate_spin_thumb.gif",
    "sphx_glr_animate_thumb.png": "sphx_glr_animate_thumb.gif",
    "sphx_glr_precog_thumb.png": "sphx_glr_precog_thumb.gif",
    "sphx_glr_save_movie_thumb.png": "sphx_glr_save_movie_thumb.gif"
}

def copy_gif_thumbnails():
    """Copy GIF thumbnails from _static/thumbnails to _build/html/_images"""
    print("Copying GIF thumbnails...")
    
    if not os.path.exists(BUILD_IMAGES_DIR):
        print(f"Error: Build images directory not found: {BUILD_IMAGES_DIR}")
        return False
        
    if not os.path.exists(STATIC_THUMBS_DIR):
        print(f"Error: Static thumbnails directory not found: {STATIC_THUMBS_DIR}")
        return False
    
    # Copy all GIF files from static to build directory
    gif_files = [f for f in os.listdir(STATIC_THUMBS_DIR) if f.endswith('.gif')]
    
    for gif_file in gif_files:
        src = os.path.join(STATIC_THUMBS_DIR, gif_file)
        dst = os.path.join(BUILD_IMAGES_DIR, gif_file)
        
        shutil.copy2(src, dst)
        print(f"  Copied: {gif_file}")
    
    print(f"Copied {len(gif_files)} GIF thumbnails")
    return True

def update_html_references():
    """Update HTML gallery to reference GIF files instead of PNG"""
    print("Updating HTML references...")
    
    if not os.path.exists(GALLERY_HTML):
        print(f"Error: Gallery HTML not found: {GALLERY_HTML}")
        return False
    
    # Read the HTML file
    with open(GALLERY_HTML, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Replace PNG references with GIF references
    replacements_made = 0
    for png_name, gif_name in GIF_REPLACEMENTS.items():
        if png_name in html_content:
            html_content = html_content.replace(png_name, gif_name)
            replacements_made += 1
            print(f"  Replaced: {png_name} -> {gif_name}")
    
    # Write the updated HTML back
    with open(GALLERY_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Made {replacements_made} HTML replacements")
    return True

def main():
    """Main function to run post-build processing"""
    print("Running post-build script to fix animated thumbnails...")
    
    success = copy_gif_thumbnails()
    if success:
        success = update_html_references()
    
    if success:
        print("✅ Post-build processing completed successfully!")
        print("Animated GIF thumbnails should now be working in the gallery.")
    else:
        print("❌ Post-build processing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())