"""Round17 Task 17 evidence: render the canonical pipeline-order flowchart
(GH #153) as a committed SVG (docs/_static/pipeline_order.svg), used by
docs/pipeline_order.rst. A committed SVG avoids adding a graphviz system
binary dependency to the readthedocs build -- this script is the one-time
generator; the SVG itself is what ships.

Run from the repo root:

    .venv/bin/python scripts/round17_evidence/pipeline_order_diagram.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'docs', '_static', 'pipeline_order.svg')

# main-line stages, in canonical order (GH #153)
STAGES = [
    ('load/format\n(impute)', '#007030'),
    ('manip', '#2e8b57'),
    ('normalize', '#3aa76d'),
    ('reduce', '#4cae4c'),
    ('align', '#5fbf6f'),
    ('cluster\n(hue)', '#72c882'),
    ('plot/\nanimate', '#8fd39e'),
]
PREDICT_LABEL = 'predict\n(overlay)'

fig, ax = plt.subplots(figsize=(13, 3.2))
ax.set_xlim(0, len(STAGES) + 1.6)
ax.set_ylim(-1.6, 1.6)
ax.axis('off')

box_w, box_h = 0.86, 0.9
centers = []
for i, (label, color) in enumerate(STAGES):
    x = i * 1.15 + 0.6
    centers.append(x)
    box = FancyBboxPatch((x - box_w / 2, -box_h / 2), box_w, box_h,
                          boxstyle='round,pad=0.05,rounding_size=0.08',
                          linewidth=1.4, edgecolor='#1b1b1b',
                          facecolor=color, alpha=0.85, zorder=2)
    ax.add_patch(box)
    ax.text(x, 0, label, ha='center', va='center', fontsize=10.5,
             color='white', fontweight='bold', zorder=3)

for i in range(len(centers) - 1):
    x0 = centers[i] + box_w / 2
    x1 = centers[i + 1] - box_w / 2
    arrow = FancyArrowPatch((x0, 0), (x1, 0), arrowstyle='-|>',
                             mutation_scale=16, linewidth=1.4,
                             color='#1b1b1b', zorder=1)
    ax.add_patch(arrow)

# predict overlays branch off the final (plot/animate) stage
px = centers[-1] + 1.15
pbox = FancyBboxPatch((px - box_w / 2, -box_h / 2), box_w, box_h,
                       boxstyle='round,pad=0.05,rounding_size=0.08',
                       linewidth=1.4, edgecolor='#1b1b1b', linestyle='--',
                       facecolor='#d9d9d9', alpha=0.9, zorder=2)
ax.add_patch(pbox)
ax.text(px, 0, PREDICT_LABEL, ha='center', va='center', fontsize=10.5,
         color='#1b1b1b', fontweight='bold', zorder=3)
arrow = FancyArrowPatch((centers[-1] + box_w / 2, 0), (px - box_w / 2, 0),
                         arrowstyle='-|>', mutation_scale=16, linewidth=1.4,
                         linestyle='--', color='#1b1b1b', zorder=1)
ax.add_patch(arrow)

ax.text((centers[0] + centers[-1]) / 2, 1.25,
         'Canonical hypertools pipeline order (GH #153)',
         ha='center', va='center', fontsize=13, fontweight='bold')
ax.text((centers[0] + centers[-1]) / 2, -1.3,
         'A manip LIST may interleave any stage explicitly and overrides this order; '
         'standalone stage kwargs (manip=, normalize=, reduce=, align=, cluster=) always follow it.',
         ha='center', va='center', fontsize=8.5, style='italic', color='#444444')

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, format='svg', bbox_inches='tight')
print(f'wrote {OUT}')
