# Animated Thumbnails Fix

## Problem
The gallery thumbnails for animated examples (chemtrails, animate_MDS, animate_spin, animate, precog, save_movie) were not displaying as animated GIFs. Instead, they showed static PNG thumbnails or placeholder images.

## Root Cause
Sphinx-gallery ignores the `sphinx_gallery_thumbnail_path` directive when it points to GIF files and generates PNG thumbnails instead. The GIF files are properly stored in version control but need to be manually copied and HTML references updated after the build.

## Solution
A post-build script (`docs/post_build.py`) has been created that:

1. **Copies GIF thumbnails**: Moves animated GIF files from `docs/_static/thumbnails/` to `docs/_build/html/_images/`
2. **Updates HTML references**: Replaces PNG thumbnail references with GIF references in the gallery HTML

## Usage

### Local Development
After building documentation locally:
```bash
cd docs/
make html
python post_build.py
```

### Read the Docs Integration
For Read the Docs, add this to your `.readthedocs.yaml`:
```yaml
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
  jobs:
    post_create_environment:
      # Install dependencies
      - pip install -r requirements.txt
    post_build:
      # Fix animated thumbnails after build
      - python docs/post_build.py
```

### Manual Build Commands
```bash
# Clean build
make clean
make html
python post_build.py

# Or direct sphinx
python -m sphinx.cmd.build -b html . _build/html
python post_build.py
```

## Files Involved

### Animated Examples with Custom Thumbnails
- `examples/chemtrails.py` → `sphx_glr_chemtrails_thumb.gif`
- `examples/animate_MDS.py` → `sphx_glr_animate_MDS_thumb.gif`  
- `examples/animate_spin.py` → `sphx_glr_animate_spin_thumb.gif`
- `examples/animate.py` → `sphx_glr_animate_thumb.gif`
- `examples/precog.py` → `sphx_glr_precog_thumb.gif`
- `examples/save_movie.py` → `sphx_glr_save_movie_thumb.gif`

### Static Examples with Custom Thumbnails
- `examples/explore.py` → `sphx_glr_explore_thumb.png`
- `examples/save_image.py` → `sphx_glr_save_image_thumb.png`
- `examples/analyze.py` → `sphx_glr_analyze_thumb.png`

### Key Directories
- `docs/_static/thumbnails/` - Version controlled custom thumbnails (source)
- `docs/_build/html/_images/` - Built documentation images (target)
- `docs/post_build.py` - Automation script

## Technical Details

Each example file includes a `sphinx_gallery_thumbnail_path` comment:
```python
# sphinx_gallery_thumbnail_path = '_static/thumbnails/sphx_glr_example_thumb.gif'
```

However, sphinx-gallery generates PNG thumbnails regardless of this directive when pointing to GIF files. The post-build script works around this limitation by:

1. Copying the actual GIF files to the correct location
2. Updating the HTML to reference the GIF files instead of PNG files

## Verification

After running the post-build script:
1. Check that GIF files exist in `docs/_build/html/_images/`
2. Open `docs/_build/html/auto_examples/index.html` in a browser
3. Verify that animated examples show moving thumbnails
4. Verify that static examples (explore, save_image, analyze) show custom static thumbnails

## Maintenance

When adding new animated examples:
1. Create the animated GIF thumbnail (50fps, infinite loop)
2. Add the GIF to `docs/_static/thumbnails/`
3. Add `sphinx_gallery_thumbnail_path` comment to the example
4. Update `GIF_REPLACEMENTS` dictionary in `docs/post_build.py`
5. Commit all changes to version control