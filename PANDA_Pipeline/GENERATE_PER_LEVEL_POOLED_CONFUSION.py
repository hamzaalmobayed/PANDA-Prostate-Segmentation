"""Recreate the supplied 4-panel patch-level confusion matrix figure.

Values are transcribed from the provided figure (NOT recomputed from slide-level
CSV). The percentage shown in each cell is a share of the corresponding panel's
four counts, rounded to 2 decimals for L2 and 1 decimal elsewhere.

Run: python GENERATE_PER_LEVEL_POOLED_CONFUSION.py
Outputs are saved next to this script (works on Windows).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

OUT = Path(__file__).resolve().parent
# [[TN, FP], [FN, TP]]; rows = actual, cols = predicted
DATA = [
    ('(A)', 'Pooled across all levels', np.array([[503020235601, 20205561962], [36123100491, 101707213750]], dtype=np.int64), 1),
    ('(B)', 'L0 (1×)', np.array([[471812503864, 18942328574], [34515344694, 94705676421]], dtype=np.int64), 1),
    ('(C)', 'L1 (4× downsample)', np.array([[30334204818, 1199788826], [1547608230, 6748586091]], dtype=np.int64), 1),
    ('(D)', 'L2 (16× downsample)', np.array([[873526919, 63444562], [60147567, 252951238]], dtype=np.int64), 2),
]

# Same visual blue-tone quadrants as source figure. Fixed mapping; no misleading
# comparison of heat intensities between levels with vastly different totals.
CMAP = ListedColormap(['#174e82', '#e6f1fa', '#e8f2fb', '#c1dcf6'])
SHADE_INDEX = np.array([[0, 1], [2, 3]])

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})
fig, axes = plt.subplots(2, 2, figsize=(15.4, 11.5), facecolor='white')
fig.suptitle('Per-level and pooled diagnostic confusion matrices',
             fontsize=20, fontweight='bold', y=0.98)

for ax, (panel, subtitle, counts, decimals) in zip(axes.flat, DATA):
    ax.imshow(SHADE_INDEX, cmap=CMAP, vmin=0, vmax=3, interpolation='nearest')
    ax.set_title(subtitle, fontsize=14.5, fontweight='bold', pad=10)
    total = int(counts.sum())
    for r in range(2):
        for c in range(2):
            count = int(counts[r, c])
            pct = 100 * count / total
            color = 'white' if (r, c) == (0, 0) else 'black'
            ax.text(c, r, f'{count:,}\n{pct:.{decimals}f}%',
                    ha='center', va='center', color=color, fontsize=11.2,
                    linespacing=1.35)
    ax.set_xticks([0, 1], ['Normal', 'Tumor'])
    ax.set_yticks([0, 1], ['Normal', 'Tumor'])
    ax.set_xlabel('Predicted', labelpad=4)
    ax.set_ylabel('Actual', labelpad=6)
    ax.tick_params(length=4, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_color('black')
    ax.text(-0.25, 1.035, panel, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=17, fontweight='bold')
    ax.set_aspect('equal')

fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.90,
                    wspace=0.32, hspace=0.39)

for ext in ('png', 'pdf', 'tif'):
    path = OUT / f'Fig_confusion_per_level_pooled.{ext}'
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print('Saved:', path)
plt.close(fig)
