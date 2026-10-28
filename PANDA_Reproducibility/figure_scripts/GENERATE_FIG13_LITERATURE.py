"""Recreate Figure 13 from the Dice values reported in the cited studies.
Run: python GENERATE_FIG13_LITERATURE.py <output_dir> [work_root]
Needs: matplotlib, Pillow, pandas
Outputs PNG and TIFF (RGB, LZW, 300 dpi, 2250 px wide), a vector PDF and the
JSON log (provenance_log.py). Values across different tasks/protocols are
descriptive, not head-to-head.

Sources (checked 28 Oct 2026):
- Chen et al., Med Phys 2020;47(3):1021-1033: average Dice 0.726 on a test set
  of 2,663 images from 32 WSIs (abstract).
- Silva-Rodriguez et al., IDEAL 2020, LNCS 12489:1-8: gland Dice 0.7708 on the
  test set, U-Net + residual blocks + augmentation (Table 2).
- Salvi et al., Artif Intell Med 2021;115:102076: Dice 90.16% on unseen images
  (abstract); split details are in the full text.
- Damkliang et al., PLoS One 2025;20(9):e0331613: mean Dice 0.637 on the
  unseen test set of 20 WSIs (Results text, Table 3; rounded to 0.64 in the
  abstract). Cross-validation Dice 0.87 (abstract). The manuscript uses 63.7
  in the text, Table 12 and both captions.
- This study: pooled fused WSI Dice and its 95% WSI-level bootstrap interval,
  read from paper_results/tables/global_wsi_metrics_95CI.csv.
Only PANDA has numerical 95% CI limits.
"""
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

from provenance_log import prov_dir, save_figure, source

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
if len(sys.argv) < 3 and "PANDA_DATA_ROOT" not in os.environ:
    sys.exit("give work_root as the second argument or set PANDA_DATA_ROOT")
WORK = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ["PANDA_DATA_ROOT"])
CI_CSV = WORK / 'paper_results' / 'tables' / 'global_wsi_metrics_95CI.csv'
ci = pd.read_csv(CI_CSV).set_index(['subset', 'metric']).loc[('fusion', 'dice')]
PANDA = round(100 * ci.value, 2)
# These confidence limits belong to the PANDA held-out test only.
PANDA_CI = (round(100 * ci.ci_low, 2), round(100 * ci.ci_high, 2))
studies = [
    ('Chen et al. (2020)', 72.60, '#2b9bd3'),
    ('Silva-Rodríguez et al. (2020)', 77.08, '#ff821b'),
    ('Damkliang et al. (2025)\n(independent test)', 63.70, '#24b339'),
    ('Salvi et al. / RINGS (2021)', 90.16, '#ed2b2b'),
    ('This Study — PANDA', PANDA, '#8840aa'),
]

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

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(prov_dir(OUT) / 'Fig13_literature_context.pdf', facecolor='white')
save_figure(fig, OUT, 'Fig13_literature_context.png', 'Fig13.tif', {
    'figure': 'Fig 13',
    'sources': {'panda_dice_ci': source(CI_CSV, WORK)},
    'values_shown': {name.replace('\n', ' '): value for name, value, _ in studies} | {'panda_95ci': list(PANDA_CI)},
    'parameters': {'literature_values': 'reported test-set Dice of the cited studies (see the script docstring)',
                   'panda_ci': f"WSI-level bootstrap, {int(ci.bootstrap_iterations)} iterations"}}, WORK, __file__)
