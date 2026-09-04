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

# The two HIERARCHY-only operations (1.1). They are drawn as a side branch,
# off the main line, because they run only for a MultiIndex DataFrame and are
# not ordinary stages: expansion feeds leaves INTO the chain (so every leaf
# gets the identical canonical pipeline), and mean-trace construction hangs
# off the end of it (means are built in the PLOTTED space, which is why they
# reach trace_data and never xform_data). See docs/hierarchy.rst.
HIERARCHY_COLOR = '#f0f0f0'
EXPANSION_LABEL = 'hierarchy\nexpansion'
MEANS_LABEL = 'hierarchy: mean traces\n+ hue propagation'

fig, ax = plt.subplots(figsize=(13, 4.6))
# xlim must clear the predict box's RIGHT edge: it is centred 1.15 past the
# last stage and is box_w wide, so it ends at len(STAGES) + 1.72. The old
# limit of len(STAGES) + 1.6 cut its dashed border off mid-box (visible in
# the SVG this script has been shipping); +2.3 leaves a margin instead.
ax.set_xlim(0, len(STAGES) + 2.3)
ax.set_ylim(-2.7, 2.5)
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


def side_branch(label, x, y, source, target, width=1.75):
    """Draw one hierarchy-only box at (x, y), wired to two main-line stages.

    `source`/`target` are main-line x-centers: the branch is ENTERED from
    `source` and REJOINS at `target`, both with dashed arrows, so the eye
    reads it as a detour off the linear chain rather than a stage in it.
    Both arrows are drawn explicitly (rather than from a symmetric loop) so
    that their direction is unambiguous: the first head lands on the box,
    the second lands back on the main line.
    """
    box = FancyBboxPatch((x - width / 2, y - box_h / 2), width, box_h,
                         boxstyle='round,pad=0.05,rounding_size=0.08',
                         linewidth=1.4, edgecolor='#1b1b1b', linestyle='--',
                         facecolor=HIERARCHY_COLOR, alpha=0.95, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=9.5,
            color='#1b1b1b', fontweight='bold', zorder=3)

    # the main line's edge, and the branch box's edge, both facing each other
    main_edge = box_h / 2 if y > 0 else -box_h / 2
    branch_edge = y - box_h / 2 if y > 0 else y + box_h / 2
    for (x0, y0), (x1, y1) in (
            ((source, main_edge), (x - width / 4, branch_edge)),   # in
            ((x + width / 4, branch_edge), (target, main_edge))):  # out
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=14,
            linewidth=1.2, linestyle='--', color='#555555', zorder=1))


# expansion feeds the chain just after load/format; mean construction hangs
# off cluster, just before plot/animate
side_branch(EXPANSION_LABEL, (centers[0] + centers[1]) / 2, 1.3,
            centers[0], centers[1])
side_branch(MEANS_LABEL, (centers[5] + centers[6]) / 2, -1.3,
            centers[5], centers[6])

ax.text((centers[0] + centers[-1]) / 2, 2.15,
         'Canonical hypertools pipeline order (GH #153)',
         ha='center', va='center', fontsize=13, fontweight='bold')
ax.text((centers[0] + centers[-1]) / 2, -2.35,
         'A manip LIST may interleave any stage explicitly and overrides this order; '
         'standalone stage kwargs (manip=, normalize=, reduce=, align=, cluster=) always follow it.\n'
         'Dashed side boxes run ONLY for a hierarchical (MultiIndex) DataFrame -- see docs/hierarchy.rst.',
         ha='center', va='center', fontsize=8.5, style='italic', color='#444444')

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, format='svg', bbox_inches='tight')
print(f'wrote {OUT}')
