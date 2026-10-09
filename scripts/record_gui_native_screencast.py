"""Drive the GUI-native tour case in a real Qt window and record it.

Usage (needs a Qt binding, e.g. in a throwaway env with `pip install -e . PyQt6`):

    python scripts/record_gui_native_screencast.py OUT_DIR
    ffmpeg -framerate 6 -i OUT_DIR/f%04d.png -pix_fmt yuv420p screencast.mp4


Same call as the feature tour's GUI-native case (QtAgg, interactive=True,
explore=True). Input is real Qt events (QTest), so it travels the same path
as a mouse. Each step grabs the window to a PNG
checks go to checks.json.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import hypertools as hyp
from mpl_toolkits.mplot3d import proj3d
from PyQt6.QtCore import Qt, QPoint, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
hyp.set_interactive_backend('QtAgg')
hyp.plot(hyp.load('helix'), backend='matplotlib', interactive=True, explore=True, show=True)
fig = plt.gcf()
ax = fig.axes[0]
canvas = fig.canvas
window = canvas.window()
checks = {'qt_backend': matplotlib.get_backend(), 'window_class': type(window).__name__}
frames = []

def grab(tag, at=None):
    """Save the window; `at` is the pointer position in canvas coordinates."""
    QApplication.processEvents()
    path = OUT / f'f{len(frames):04d}.png'
    window.grab().save(str(path))
    entry = {'tag': tag}
    if at is not None:
        pos = canvas.mapTo(window, at)
        dpr = window.devicePixelRatioF()
        entry['pointer'] = [pos.x() * dpr, pos.y() * dpr]
    frames.append(entry)


def point_to_widget(i):
    data = np.asarray(ax.lines[0].get_data_3d()).T if ax.lines else None
    x, y, z = data[i]
    xs, ys, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    dx, dy = ax.transData.transform((xs, ys))
    dpr = canvas.device_pixel_ratio
    return QPoint(int(dx / dpr), int((canvas.height() * dpr - dy) / dpr))

def annotation_texts():
    return [t.get_text() for t in ax.texts if t.get_visible() and t.get_text()]

steps = []


def step(fn):
    steps.append(fn)
    return fn

@step
def start():
    for _ in range(4):
        grab('opened')

@step
def hover():
    n = len(ax.lines[0].get_data_3d()[0])
    seen = []
    for i in np.linspace(0, n - 1, 12).astype(int):
        where = point_to_widget(i)
        QTest.mouseMove(canvas, where)
        QTest.qWait(120)
        seen.append(annotation_texts())
        for _ in range(3):
            grab('hover', where)
    checks['hover_annotations'] = [s for s in seen if s]
    checks['hover_distinct'] = len({tuple(s) for s in seen if s})

@step
def rotate():
    before = (ax.azim, ax.elev)
    c = QPoint(canvas.width() // 2, canvas.height() // 2)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, c)
    for k in range(1, 25):
        QTest.mouseMove(canvas, c + QPoint(6 * k, 2 * k))
        QTest.qWait(40)
        grab('rotate', c + QPoint(6 * k, 2 * k))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, c + QPoint(144, 48))
    checks['rotate_azim_elev'] = {'before': before, 'after': (ax.azim, ax.elev)}

@step
def zoom():
    before = ax.get_xlim3d()
    c = QPoint(canvas.width() // 2, canvas.height() // 2)
    QTest.mousePress(canvas, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, c)
    for k in range(1, 16):
        QTest.mouseMove(canvas, c + QPoint(0, -5 * k))
        QTest.qWait(40)
        grab('zoom', c + QPoint(0, -5 * k))
    QTest.mouseRelease(canvas, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, c + QPoint(0, -75))
    checks['zoom_xlim'] = {'before': list(before), 'after': list(ax.get_xlim3d())}
    for _ in range(4):
        grab('zoomed')

@step
def close():
    QTest.keyClick(canvas, Qt.Key.Key_Q)   # matplotlib's default close-figure key
    QTest.qWait(300)
    checks['figures_open_after_close'] = plt.get_fignums()
    checks['window_visible_after_close'] = window.isVisible()

def run(i=0):
    if i < len(steps):
        steps[i]()
        QTimer.singleShot(150, lambda: run(i + 1))
    else:
        (OUT / 'checks.json').write_text(json.dumps(checks, indent=1, default=str))
        (OUT / 'frames.json').write_text(json.dumps(frames))
        QApplication.instance().quit()

QTimer.singleShot(800, run)
t0 = time.time()
plt.show(block=True)
checks['event_loop_seconds'] = round(time.time() - t0, 1)
# closing the last window can end the loop before run() reaches its else branch
(OUT / 'frames.json').write_text(json.dumps(frames))
(OUT / 'checks.json').write_text(json.dumps(checks, indent=1, default=str))
print('clean exit')
