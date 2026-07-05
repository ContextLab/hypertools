#!/usr/bin/env python
"""Visual verification of the BUILT HTML docs using a real headless browser.

This is Plan 8 / Task 5 of the HyperTools 1.0 docs migration. It does NOT
rebuild the docs and does NOT stub anything out: it serves the already-built
``docs/_build/html`` tree over a real local HTTP server, drives a real
headless Chromium instance (via Playwright) against it, and makes hard
assertions about what actually rendered in the DOM -- non-blank example
images/thumbnails, real embedded animations (mp4 <video> / plotly animated
divs), and a branch-aware "Open in Colab" affordance.

Usage
-----
    .venv/bin/python scripts/verify_docs_playwright.py

Exits non-zero (and prints the failing assertions) if ANY page fails
verification. Screenshots (full-page + cropped element captures used for the
non-blank pixel-variance check) are written to ``docs/images/v1.0-docs/``.
"""

from __future__ import annotations

import functools
import http.server
import io
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_HTML = REPO_ROOT / "docs" / "_build" / "html"
SCREENSHOT_DIR = REPO_ROOT / "docs" / "images" / "v1.0-docs"
BRANCH = "dev-1.0-refactor"

# Minimum standard deviation of pixel intensities (0-255 scale) for an
# element screenshot to be considered "non-blank". A truly blank/white or
# solid-color render has std ~= 0; real plots/animations have std well into
# the double digits. This is a real pixel-content check, not a stub.
MIN_PIXEL_STD = 2.0

MIN_GALLERY_THUMBS = 20  # built gallery has 40 as of this writing

PAGES = [
    {"path": "auto_examples/index.html", "kind": "gallery_index",
     "shot": "01_gallery_index.png"},
    {"path": "auto_examples/plot_basic.html", "kind": "static_image",
     "shot": "02_plot_basic.png"},
    {"path": "auto_examples/animate_spin.html", "kind": "video",
     "shot": "03_animate_spin.png"},
    {"path": "auto_examples/animate_plotly.html", "kind": "plotly_animated",
     "shot": "04_animate_plotly.png"},
    {"path": "auto_examples/plot_shape_morph.html", "kind": "video",
     "shot": "05_plot_shape_morph.png"},
    {"path": "auto_examples/plot_clusters.html", "kind": "static_image",
     "shot": "06_plot_clusters.png"},
    {"path": "tutorials/plot.html", "kind": "tutorial",
     "shot": "07_tutorial_plot.png"},
    {"path": "tutorials/align.html", "kind": "tutorial",
     "shot": "08_tutorial_align.png"},
]


class VerificationFailure(Exception):
    """Raised when a real assertion about the rendered page fails."""


# --------------------------------------------------------------------------
# Local static server for docs/_build/html
# --------------------------------------------------------------------------

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port),
         "--bind", "127.0.0.1", "--directory", str(DOCS_HTML)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + 15
    last_err = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"docs HTTP server process exited early (code {proc.returncode})"
            )
        try:
            urllib.request.urlopen(url, timeout=1)
            return proc
        except Exception as e:  # noqa: BLE001 - polling until server is up
            last_err = e
            time.sleep(0.2)
    proc.terminate()
    raise RuntimeError(f"docs HTTP server did not start in time: {last_err}")


# --------------------------------------------------------------------------
# Pixel-level non-blank verification (real image analysis, no stubs)
# --------------------------------------------------------------------------

def _save_and_check_nonblank(png_bytes: bytes, out_path: Path) -> float:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img).astype(np.float64)
    std = float(arr.std())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png_bytes)
    if std < MIN_PIXEL_STD:
        raise VerificationFailure(
            f"{out_path.name}: element render looks blank (pixel stddev={std:.4f} "
            f"< {MIN_PIXEL_STD})"
        )
    return std


# --------------------------------------------------------------------------
# Colab badge / branch-aware install checks
# --------------------------------------------------------------------------

def verify_colab_badge(page) -> str:
    """Gallery example pages: real 'Open in Colab' badge + branch-aware href."""
    badge_img = page.locator('img[alt="Open in Colab"]')
    if badge_img.count() < 1:
        raise VerificationFailure("no 'Open in Colab' badge image found on page")
    link = page.locator('a:has(img[alt="Open in Colab"])').first
    href = link.get_attribute("href")
    if not href or "colab.research.google.com" not in href:
        raise VerificationFailure(f"Colab badge link href looks wrong: {href!r}")
    if BRANCH not in href:
        raise VerificationFailure(
            f"Colab badge href is not branch-aware (expected {BRANCH!r} in URL): {href!r}"
        )
    return href


def verify_tutorial_branch_aware_install(page) -> str:
    """Tutorial (nbsphinx) pages: no image badge, but a real branch-aware
    `pip install ... @<branch>` cell must be present as the notebook's
    install-from-source instructions."""
    content = page.content()
    needle = f"@{BRANCH}"
    if needle not in content:
        raise VerificationFailure(
            f"tutorial page has no branch-aware install reference ({needle!r} not found)"
        )
    if "pip install" not in content:
        raise VerificationFailure(
            "tutorial page has no 'pip install' cell alongside the branch reference"
        )
    return needle


# --------------------------------------------------------------------------
# Per-page-kind verification
# --------------------------------------------------------------------------

def verify_gallery_index(page, shot_path: Path) -> dict:
    thumbs = page.locator(".sphx-glr-thumbcontainer")
    count = thumbs.count()
    if count < MIN_GALLERY_THUMBS:
        raise VerificationFailure(
            f"gallery index has only {count} thumbnails (expected >= {MIN_GALLERY_THUMBS})"
        )
    imgs = page.locator(".sphx-glr-thumbcontainer img")
    n_check = min(5, imgs.count())
    stds = []
    for i in range(n_check):
        img = imgs.nth(i)
        src = img.get_attribute("src")
        if not src:
            raise VerificationFailure(f"gallery thumbnail #{i} has an empty src")
        natural = img.evaluate("el => [el.naturalWidth, el.naturalHeight]")
        if natural[0] == 0 or natural[1] == 0:
            raise VerificationFailure(
                f"gallery thumbnail #{i} (src={src}) has zero natural size {natural}"
            )
        png_bytes = img.screenshot()
        stds.append(_save_and_check_nonblank(
            png_bytes, SCREENSHOT_DIR / f"01_gallery_index_thumb_{i}.png"))
    page.screenshot(path=str(shot_path), full_page=True)
    return {"thumbnail_count": count, "checked_thumbnails": n_check,
            "pixel_stds": stds}


def verify_static_image(page, shot_path: Path) -> dict:
    imgs = page.locator('img[src*="sphx_glr"]')
    if imgs.count() < 1:
        raise VerificationFailure("no sphx_glr example <img> found on page")
    img = imgs.first
    img.wait_for(state="visible", timeout=10000)
    natural = img.evaluate("el => [el.naturalWidth, el.naturalHeight]")
    if natural[0] == 0 or natural[1] == 0:
        raise VerificationFailure(f"example image has zero natural size {natural}")
    png_bytes = img.screenshot()
    std = _save_and_check_nonblank(
        png_bytes, shot_path.with_name(shot_path.stem + "_crop.png"))
    href = verify_colab_badge(page)
    page.screenshot(path=str(shot_path), full_page=True)
    return {"natural_size": natural, "pixel_std": std, "colab_href": href}


def verify_video(page, shot_path: Path) -> dict:
    videos = page.locator("video")
    if videos.count() < 1:
        raise VerificationFailure("no <video> element found on animated example page")
    video = videos.first
    source_src = page.locator("video source").first.get_attribute("src")
    if not source_src:
        raise VerificationFailure("video <source> element has an empty src")
    page.wait_for_function(
        "sel => { const v = document.querySelector(sel); return !!v && v.readyState >= 1; }",
        arg="video", timeout=15000,
    )
    dims = video.evaluate("v => [v.videoWidth, v.videoHeight]")
    if dims[0] == 0 or dims[1] == 0:
        raise VerificationFailure(f"video has zero decoded dimensions {dims}")
    # Seek partway into the clip and screenshot an actual decoded frame --
    # a real, non-blank verification of the embedded animation content.
    video.evaluate("v => { v.currentTime = Math.min(1.0, (v.duration || 2) / 2); }")
    page.wait_for_timeout(500)
    png_bytes = video.screenshot()
    std = _save_and_check_nonblank(
        png_bytes, shot_path.with_name(shot_path.stem + "_frame.png"))
    href = verify_colab_badge(page)
    page.screenshot(path=str(shot_path), full_page=True)
    return {"video_dims": dims, "source_src": source_src, "pixel_std": std,
            "colab_href": href}


def verify_plotly_animated(page, shot_path: Path) -> dict:
    divs = page.locator(".plotly-graph-div")
    if divs.count() < 1:
        raise VerificationFailure("no plotly-graph-div found on page")
    div = divs.first
    div.wait_for(state="visible", timeout=15000)
    page.wait_for_function(
        "sel => { const el = document.querySelector(sel); "
        "return !!el && el.querySelectorAll('svg, canvas').length > 0; }",
        arg=".plotly-graph-div", timeout=15000,
    )
    content = page.content()
    if "Plotly.animate(" not in content or "Plotly.addFrames(" not in content:
        raise VerificationFailure(
            "no embedded Plotly animation calls (addFrames/animate) found in page source"
        )
    png_bytes = div.screenshot()
    std = _save_and_check_nonblank(
        png_bytes, shot_path.with_name(shot_path.stem + "_plot.png"))
    href = verify_colab_badge(page)
    page.screenshot(path=str(shot_path), full_page=True)
    return {"pixel_std": std, "colab_href": href}


def verify_tutorial(page, shot_path: Path) -> dict:
    imgs = page.locator('img[src*="tutorials_"]')
    if imgs.count() < 1:
        raise VerificationFailure("no rendered figure <img> found on tutorial page")
    img = imgs.first
    img.wait_for(state="visible", timeout=10000)
    natural = img.evaluate("el => [el.naturalWidth, el.naturalHeight]")
    if natural[0] == 0 or natural[1] == 0:
        raise VerificationFailure(f"tutorial figure has zero natural size {natural}")
    png_bytes = img.screenshot()
    std = _save_and_check_nonblank(
        png_bytes, shot_path.with_name(shot_path.stem + "_crop.png"))
    install_ref = verify_tutorial_branch_aware_install(page)
    page.screenshot(path=str(shot_path), full_page=True)
    return {"natural_size": natural, "pixel_std": std, "install_ref": install_ref}


VERIFIERS = {
    "gallery_index": verify_gallery_index,
    "static_image": verify_static_image,
    "video": verify_video,
    "plotly_animated": verify_plotly_animated,
    "tutorial": verify_tutorial,
}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    if not DOCS_HTML.exists():
        print(f"FATAL: built docs not found at {DOCS_HTML}. Build the docs first "
              f"(PATH=.venv/bin:$PATH make -C docs html).", file=sys.stderr)
        return 1

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    port = find_free_port()
    print(f"Starting local docs server on 127.0.0.1:{port} -> {DOCS_HTML}")
    server_proc = start_server(port)
    base_url = f"http://127.0.0.1:{port}/"

    results = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            for entry in PAGES:
                url = base_url + entry["path"]
                shot_path = SCREENSHOT_DIR / entry["shot"]
                print(f"--- Verifying {entry['path']} ({entry['kind']}) ---")
                try:
                    page.goto(url, wait_until="load", timeout=30000)
                    verifier = VERIFIERS[entry["kind"]]
                    detail = verifier(page, shot_path)
                    print(f"    OK: {detail}")
                    results.append({"page": entry["path"], "kind": entry["kind"],
                                     "ok": True, "detail": detail})
                except Exception as e:  # noqa: BLE001 - collect per-page, report all
                    print(f"    FAIL: {e}", file=sys.stderr)
                    results.append({"page": entry["path"], "kind": entry["kind"],
                                     "ok": False, "error": str(e)})
            browser.close()
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    failures = [r for r in results if not r["ok"]]

    print("\n=== Summary ===")
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  [{status}] {r['page']} ({r['kind']})")

    if failures:
        print(f"\n{len(failures)}/{len(results)} page(s) FAILED verification:",
              file=sys.stderr)
        for r in failures:
            print(f"  - {r['page']}: {r['error']}", file=sys.stderr)
        return 1

    print(f"\nAll {len(results)} pages verified OK. Screenshots written to "
          f"{SCREENSHOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
