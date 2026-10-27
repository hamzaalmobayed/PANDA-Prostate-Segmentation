"""Recreate Figure 13 from the user-provided reported Dice values.
Run: python GENERATE_FIG13_LITERATURE.py
Needs: matplotlib
Outputs PNG 300 dpi, PDF vector, TIFF 300 dpi in the script directory.
Values across different tasks/protocols are descriptive, not head-to-head.
Salvi/RINGS test independence is not established by these inputs.
Only PANDA has supplied numerical 95% CI limits.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = Path(__file__).resolve().parent
studies = [
    ('Chen et al. (2020)', 72.60, '#2b9bd3'),
    ('Silva-Rodríguez et al. (2020)', 77.08, '#ff821b'),
    ('Damkliang et al. (2025)\n(independent test)', 64.00, '#24b339'),
    ('Salvi et al. / RINGS (2021)', 90.16, '#ed2b2b'),
    ('This Study — PANDA', 85.22, '#8840aa'),
]
# These confidence limits belong to the PANDA held-out test only.
PANDA_CI = (84.15, 86.16)

fig, ax = plt.subplots(figsize=(13.5, 7.6))
y_positions = list(range(len(studies) - 1, -1, -1))
for (name, value, color), y in zip(studies, y_positions):
    ax.scatter(value, y, s=160, color=color, edgecolor='black', linewidth=0.9, zorder=3)
    label = f'{value:.2f}' if name.startswith(('Silva', 'Salvi', 'This')) else f'{value:.1f}'
    ax.text(value, y + 0.20, label, fontsize=12, weight='bold', ha='center', va='bottom')

panda_val = studies[-1][1]
panda_y = y_positions[-1]
ax.errorbar(panda_val, panda_y,
            xerr=[[panda_val - PANDA_CI[0]], [PANDA_CI[1] - panda_val]],
            fmt='none', ecolor='#8840aa', capsize=8, elinewidth=2.0, capthick=2.0, zorder=2)
ax.text(panda_val, panda_y - 0.23, f'({PANDA_CI[0]:.2f}–{PANDA_CI[1]:.2f})',
        ha='center', fontsize=10, color='black')

ax.set_yticks(y_positions, [s[0] for s in studies], fontsize=12)
for tick in ax.get_yticklabels():
    if tick.get_text().startswith('This Study'):
        tick.set_fontweight('bold')
ax.set_xlim(60, 92)
ax.set_ylim(-0.55, 4.55)
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.grid(axis='x', which='major', color='#d5d5d5', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)
ax.set_xlabel('Dice Score (%)', fontsize=14, fontweight='bold')
ax.set_title('Descriptive cross-task prostate histopathology segmentation Dice',
             fontsize=16, fontweight='bold', pad=20)
ax.tick_params(axis='x', labelsize=11)
fig.subplots_adjust(left=0.29, right=0.97, top=0.89, bottom=0.15)

for ext in ('png', 'pdf', 'tif'):
    dst = OUT / f'Fig13_literature_context.{ext}'
    fig.savefig(dst, dpi=300, facecolor='white')
    print('Saved:', dst)
plt.close(fig)
