import os, warnings, traceback
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
import hypertools as hyp

rng = np.random.default_rng(0)
def traj(n=60, d=3):
    t = np.linspace(0, 4*np.pi, n)
    base = np.stack([np.cos(t), np.sin(t), t/ (4*np.pi)], axis=1)[:, :d]
    return base + 0.02*rng.standard_normal((n, d))

X3 = traj(60,3)
X3b = traj(60,3)+0.5
X2 = traj(60,2)
labels = [f"p{i}" if i%10==0 else None for i in range(60)]

def describe(obj):
    t = type(obj)
    info = f"{t.__module__}.{t.__name__}"
    if isinstance(obj, tuple):
        info += " -> tuple(" + ", ".join(type(o).__name__ for o in obj) + ")"
    return info

styles = [True, "parallel", "spin", "serial", "window", "morph", "chemtrails", "precog", "bullettime"]

print("="*70)
print("ISSUE 1 + style enumeration (3D, show=False)")
print("="*70)
for s in styles:
    data = [X3, X3b] if s=="morph" else X3
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                r = hyp.plot(data, ndims=3, animate=s, duration=1, rotations=1, show=False)
            except Warning as w:
                # rerun ignoring warnings to get object
                warnings.simplefilter("ignore")
                r = hyp.plot(data, ndims=3, animate=s, duration=1, rotations=1, show=False)
        rt = describe(r)
        has_html = hasattr(r, "to_html5_video")
        # if tuple, check element 1
        anim = r[1] if isinstance(r, tuple) else None
        anim_type = type(anim).__name__ if anim is not None else "-"
        print(f"animate={s!r:14} -> {rt:55} html5={has_html} animObj={anim_type}")
    except Exception as e:
        print(f"animate={s!r:14} -> EXC {type(e).__name__}: {e}")

print()
print("="*70)
print("2D, show=False")
print("="*70)
for s in styles:
    data = [X2, traj(60,2)+0.5] if s=="morph" else X2
    try:
        warnings.simplefilter("ignore")
        r = hyp.plot(data, ndims=2, animate=s, duration=1, show=False)
        print(f"animate={s!r:14} -> {describe(r)}")
    except Exception as e:
        print(f"animate={s!r:14} -> EXC {type(e).__name__}: {e}")

print()
print("="*70)
print("ISSUE 1 verbatim: to_html5_video on returned object")
print("="*70)
warnings.simplefilter("ignore")
anim = hyp.plot(X3, ndims=3, animate='spin', duration=4, rotations=1, show=False)
print("returned type:", describe(anim))
try:
    anim.to_html5_video()
    print("to_html5_video OK")
except Exception as e:
    traceback.print_exc()

print()
print("="*70)
print("ISSUE 2: animate='chemtrails' -> what happens?")
print("="*70)
r = hyp.plot([X3], ndims=3, animate='chemtrails', duration=1, show=False)
print("chemtrails returned:", describe(r))
r2 = hyp.plot([X3], ndims=3, animate='window', duration=1, show=False)
print("window returned:", describe(r2))
