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

# Auto-detect build directory (Read the Docs vs local)
def find_build_dirs():
    """Find the actual build directory paths"""
    possible_build_dirs = [
        # Local build
        os.path.join(DOCS_DIR, "_build", "html"),
        # Read the Docs build (from docs dir)
        os.path.join(DOCS_DIR, "..", "_readthedocs", "html"),
        # Read the Docs alternative paths
        os.path.join(DOCS_DIR, "..", "..", "_readthedocs", "html"),
        # Additional Read the Docs patterns based on error message
        "/tmp/_readthedocs_build/html",
        os.path.join(os.getcwd(), "..", "_readthedocs", "html"),
        os.path.join(os.getcwd(), "_readthedocs", "html"),
        # Check if we're already in the output directory
        os.path.join(os.getcwd(), "_images", ".."),
    ]
    
    # Also check environment variables that Read the Docs might set
    rtd_output = os.environ.get('READTHEDOCS_OUTPUT', '')
    if rtd_output:
        possible_build_dirs.insert(0, rtd_output)
    
    for build_dir in possible_build_dirs:
        if build_dir and os.path.exists(build_dir):
            images_dir = os.path.join(build_dir, "_images")
            gallery_html = os.path.join(build_dir, "auto_examples", "index.html")
            if os.path.exists(images_dir) and os.path.exists(gallery_html):
                return images_dir, gallery_html
    
    return None, None

BUILD_IMAGES_DIR, GALLERY_HTML = find_build_dirs()

# Mapping of PNG to GIF thumbnails that should be replaced
GIF_REPLACEMENTS = {
    "sphx_glr_chemtrails_thumb.png": "sphx_glr_chemtrails_thumb.gif",
    "sphx_glr_animate_MDS_thumb.png": "sphx_glr_animate_MDS_thumb.gif", 
    "sphx_glr_animate_spin_thumb.png": "sphx_glr_animate_spin_thumb.gif",
    "sphx_glr_animate_thumb.png": "sphx_glr_animate_thumb.gif",
    "sphx_glr_precog_thumb.png": "sphx_glr_precog_thumb.gif",
    "sphx_glr_save_movie_thumb.png": "sphx_glr_save_movie_thumb.gif",
    "sphx_glr_animate_plotly_thumb.png": "sphx_glr_animate_plotly_thumb.gif",
    # QC 2026-07: the story-trajectories example ships an animated gif thumbnail
    # (docs/_static/thumbnails/sphx_glr_plot_story_trajectories_thumb.gif) and
    # sets sphinx_gallery_thumbnail_path to it, but it was never registered here,
    # so post_build never swapped its static png for the gif -- the gallery card
    # showed a frozen frame instead of the animation.
    "sphx_glr_plot_story_trajectories_thumb.png":
        "sphx_glr_plot_story_trajectories_thumb.gif",
}

def copy_gif_thumbnails():
    """Copy GIF thumbnails from _static/thumbnails to _build/html/_images"""
    print("Copying GIF thumbnails...")
    
    # Re-detect directories if needed
    global BUILD_IMAGES_DIR, GALLERY_HTML
    if not BUILD_IMAGES_DIR:
        BUILD_IMAGES_DIR, GALLERY_HTML = find_build_dirs()
    
    if not BUILD_IMAGES_DIR or not os.path.exists(BUILD_IMAGES_DIR):
        print(f"Error: Build images directory not found.")
        print(f"Searched paths:")
        possible_dirs = [
            os.path.join(DOCS_DIR, "_build", "html", "_images"),
            os.path.join(DOCS_DIR, "..", "_readthedocs", "html", "_images"),
            os.path.join(DOCS_DIR, "..", "..", "_readthedocs", "html", "_images"),
        ]
        for d in possible_dirs:
            print(f"  {d} - {'EXISTS' if os.path.exists(d) else 'NOT FOUND'}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"DOCS_DIR: {DOCS_DIR}")
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
    
    # Re-detect directories if needed
    global BUILD_IMAGES_DIR, GALLERY_HTML
    if not GALLERY_HTML:
        BUILD_IMAGES_DIR, GALLERY_HTML = find_build_dirs()
    
    if not GALLERY_HTML or not os.path.exists(GALLERY_HTML):
        print(f"Error: Gallery HTML not found: {GALLERY_HTML if GALLERY_HTML else 'None'}")
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

def wrap_thumbnail_links():
    """Make gallery thumbnails clickable: sphinx-gallery >= 0.17 markup no
    longer wraps the thumbnail <img> in an anchor (only the small title
    text links), so clicking a thumbnail did nothing. Per review, clicking
    a gallery example should open a runnable notebook: the image links to
    the example's notebook on Colab (title text still opens the example
    page, which embeds the rendered output)."""
    if GALLERY_HTML is None:
        print("  Skipping thumbnail links (no build dir)")
        return 0
    import re as _re
    import subprocess
    branch = os.environ.get('READTHEDOCS_GIT_IDENTIFIER', '')
    if not branch:
        try:
            branch = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True, cwd=DOCS_DIR,
                timeout=10).stdout.strip()
        except Exception:
            branch = ''
    branch = branch or 'master'
    base = ('https://colab.research.google.com/github/ContextLab/'
            f'hypertools/blob/{branch}/docs/auto_examples/')

    def _wrap(match):
        img = match.group(0)
        m = _re.search(r'sphx_glr_(.+?)_thumb', img)
        if m is None:
            return img
        url = f'{base}{m.group(1)}.ipynb'
        return (f'<a class="hypertools-thumb-link" href="{url}" '
                f'target="_blank" rel="noopener" '
                f'title="open this example as a notebook in Colab">'
                f'{img}</a>')

    with open(GALLERY_HTML) as f:
        html = f.read()
    if 'hypertools-thumb-link' in html:
        print("  Thumbnail links already present")
        return 0
    html, n = _re.subn(
        r'<img[^>]*sphx_glr_[^>]*_thumb[^>]*/?>', _wrap, html)
    with open(GALLERY_HTML, 'w') as f:
        f.write(html)
    print(f"  Wrapped {n} gallery thumbnails with notebook links "
          f"(branch: {branch})")
    return n


def inject_notebook_badges():
    """Add a prominent 'Open in Colab / download notebook' bar to the top of
    every gallery example page, so clicking through a gallery example leads
    straight to a runnable notebook with the relevant code."""
    if GALLERY_HTML is None:
        print("  Skipping notebook badges (no build dir)")
        return 0
    import subprocess
    branch = os.environ.get('READTHEDOCS_GIT_IDENTIFIER', '')
    if not branch:
        try:
            branch = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True, cwd=DOCS_DIR,
                timeout=10).stdout.strip()
        except Exception:
            branch = ''
    branch = branch or 'master'

    gallery_dir = os.path.dirname(GALLERY_HTML)
    n = 0
    for fname in os.listdir(gallery_dir):
        if not fname.endswith('.html') or fname == 'index.html':
            continue
        stem = fname[:-5]
        path = os.path.join(gallery_dir, fname)
        with open(path) as f:
            html = f.read()
        if 'hypertools-colab-badge' in html:
            continue  # already injected (idempotent re-runs)
        colab = ('https://colab.research.google.com/github/ContextLab/'
                 f'hypertools/blob/{branch}/docs/auto_examples/{stem}.ipynb')
        badge = (
            '<p class="hypertools-colab-badge" style="margin:0.5em 0 1em 0">'
            f'<a href="{colab}" target="_blank" rel="noopener">'
            '<img src="https://colab.research.google.com/assets/'
            'colab-badge.svg" alt="Open in Colab" '
            'style="vertical-align:middle"></a>'
            '&nbsp; <em>run this example as a notebook &mdash; or grab the '
            '<a class="reference download" '
            f'href="{stem}.ipynb" download>.ipynb</a> from the links at the '
            'bottom of the page</em></p>')
        # place right after the page's first <h1>
        marker = '</h1>'
        idx = html.find(marker)
        if idx < 0:
            continue
        html = html[:idx + len(marker)] + badge + html[idx + len(marker):]
        with open(path, 'w') as f:
            f.write(html)
        n += 1
    print(f"  Injected notebook badges into {n} example pages "
          f"(branch: {branch})")
    return n


def main():
    """Main function to run post-build processing"""
    print("Running post-build script to fix animated thumbnails...")
    
    success = copy_gif_thumbnails()
    if success:
        success = update_html_references()

    inject_notebook_badges()
    wrap_thumbnail_links()

    if success:
        print("✅ Post-build processing completed successfully!")
        print("Animated GIF thumbnails should now be working in the gallery.")
    else:
        print("❌ Post-build processing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())