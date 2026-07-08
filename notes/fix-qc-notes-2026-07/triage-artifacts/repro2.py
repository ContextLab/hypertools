import os, warnings, traceback
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
import hypertools as hyp
warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
t = np.linspace(0,4*np.pi,60)
X3 = np.stack([np.cos(t), np.sin(t), t/(4*np.pi)],axis=1)+0.02*rng.standard_normal((60,3))
labels = [f"p{i}" if i%15==0 else None for i in range(60)]
OUT="/private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-066a-4ce2-b1f6-7c07c5e87b15/scratchpad/triage"

# ---- Issue 3: label persistence across frames ----
print("ISSUE 3: per-point labels per frame")
fig, anim = hyp.plot(X3, ndims=3, animate='serial', labels=labels, duration=1, frame_rate=5, show=False)
ax = fig.axes[0]
# count annotation/text artists on axes
def count_labels(ax):
    return sum(1 for t_ in ax.texts if t_.get_text() in [l for l in labels if l])
# render frame 0 and a late frame, count visible label texts
anim._draw_frame(0)
n0 = len([t_ for t_ in ax.texts if t_.get_visible() and t_.get_text().startswith('p')])
anim._draw_frame(len(range(anim._save_count))-1 if hasattr(anim,'_save_count') else 5)
nlate = len([t_ for t_ in ax.texts if t_.get_visible() and t_.get_text().startswith('p')])
print(f"  visible point-labels at frame0={n0}, at late frame={nlate} (save_count={getattr(anim,'_save_count','?')})")
print(f"  total 'p' text artists on axes = {len([t_ for t_ in ax.texts if t_.get_text().startswith('p')])}")

# ---- gif save path ----
print("\nSAVE path (gif):")
try:
    fig2 = hyp.plot(X3, ndims=3, animate='spin', duration=1, frame_rate=5, save_path=OUT+'/anim_spin.gif', show=False)
    print("  saved gif OK, type returned:", type(fig2).__name__ if not isinstance(fig2,tuple) else 'tuple', "size=", os.path.getsize(OUT+'/anim_spin.gif'))
except Exception as e:
    traceback.print_exc()

# ---- show=True ----
print("\nshow=True (Agg):")
try:
    r = hyp.plot(X3, ndims=3, animate='spin', duration=1, frame_rate=5, show=True)
    print("  returned:", type(r).__name__ if not isinstance(r,tuple) else 'tuple(%s)'%",".join(type(o).__name__ for o in r))
except Exception as e:
    traceback.print_exc()

# ---- plotly backend ----
print("\nplotly backend animate:")
try:
    hyp.set_interactive_backend('plotly')
    r = hyp.plot(X3, ndims=3, animate='spin', duration=1, show=False)
    print("  returned:", type(r).__module__+"."+type(r).__name__ if not isinstance(r,tuple) else 'tuple')
    print("  has frames:", hasattr(r,'frames') and len(getattr(r,'frames') or []))
except Exception as e:
    traceback.print_exc()
