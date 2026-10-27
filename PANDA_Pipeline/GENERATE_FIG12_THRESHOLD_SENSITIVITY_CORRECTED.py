from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Recreates the reference figure in a clean, Windows-friendly way.
# Saves outputs next to this script.

thresholds = np.array([0.30, 0.35, 0.40, 0.43, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
dice =      np.array([0.820, 0.837, 0.845, 0.845, 0.843, 0.835, 0.820, 0.800, 0.769, 0.758])
precision = np.array([0.750, 0.794, 0.827, 0.845, 0.861, 0.886, 0.907, 0.923, 0.939, 0.950])
recall =    np.array([0.912, 0.890, 0.865, 0.845, 0.830, 0.795, 0.755, 0.714, 0.669, 0.626])

selected_threshold = 0.43
selected_idx = list(thresholds).index(selected_threshold)
selected_dice = dice[selected_idx]
selected_precision = precision[selected_idx]
selected_recall = recall[selected_idx]

plt.rcParams.update({
    'font.family': 'serif',
    'axes.titlesize': 30,
    'axes.labelsize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
})

fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

# Shaded optimal region
region = ax.axvspan(0.40, 0.45, color='#dbe8f4', alpha=0.70, zorder=0)

# Curves
line_dice, = ax.plot(thresholds, dice, color='#1f77b4', marker='o', markersize=8, linewidth=2.2, label='Dice', zorder=3)
line_precision, = ax.plot(thresholds, precision, color='#ff7f0e', marker='o', markersize=8, linewidth=2.2, label='Precision', zorder=3)
line_recall, = ax.plot(thresholds, recall, color='#2ca02c', marker='o', markersize=8, linewidth=2.2, label='Recall', zorder=3)

# Selected threshold line
sel_line = ax.axvline(selected_threshold, color='#1f77b4', linestyle='--', linewidth=2.2, zorder=2)

# Highlight selected points
ax.scatter([selected_threshold], [selected_dice], s=160, color='#1f77b4', edgecolor='black', linewidth=0.8, zorder=5)
ax.scatter([selected_threshold], [selected_precision], s=160, color='#ff7f0e', edgecolor='black', linewidth=0.8, zorder=5)
ax.scatter([selected_threshold], [selected_recall], s=160, color='#2ca02c', edgecolor='black', linewidth=0.8, zorder=5)

# Callouts
label_text = '0.845'

ax.annotate(label_text,
            xy=(selected_threshold, selected_precision), xycoords='data',
            xytext=(0.439, 0.878), textcoords='data',
            color='#ff7f0e', fontsize=20, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#ff7f0e', lw=1.3),
            arrowprops=dict(arrowstyle='-|>', color='#ff7f0e', lw=1.8))

ax.annotate(label_text,
            xy=(selected_threshold, selected_dice), xycoords='data',
            xytext=(0.457, 0.848), textcoords='data',
            color='#1f77b4', fontsize=20, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#1f77b4', lw=1.3),
            arrowprops=dict(arrowstyle='-|>', color='#1f77b4', lw=1.8))

ax.annotate(label_text,
            xy=(selected_threshold, selected_recall), xycoords='data',
            xytext=(0.441, 0.804), textcoords='data',
            color='#2ca02c', fontsize=20, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#2ca02c', lw=1.3),
            arrowprops=dict(arrowstyle='-|>', color='#2ca02c', lw=1.8))

# Axes formatting
ax.set_title('Validation Threshold Sensitivity', pad=22)
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_xlim(0.287, 0.72)
ax.set_ylim(0.60, 0.96)
ax.set_xticks([0.30, 0.35, 0.40, 0.43, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
ax.set_yticks(np.arange(0.60, 0.96, 0.05))

# Color only the 0.43 x-tick label in blue and bold, with no extra duplicate label.
fig.canvas.draw()  # ensure tick labels exist
for tick_val, tick_lbl in zip(ax.get_xticks(), ax.get_xticklabels()):
    if abs(tick_val - selected_threshold) < 1e-9:
        tick_lbl.set_color('#1f77b4')
        tick_lbl.set_fontweight('bold')
        tick_lbl.set_fontsize(18)

# Smaller legend, fully inside the axes, positioned to avoid covering the curves.
handles = [line_dice, line_precision, line_recall, sel_line, region]
labels = ['Dice', 'Precision', 'Recall', 'Selected threshold\n(0.43)', 'Optimal threshold\nregion']
legend = ax.legend(handles, labels,
                   loc='upper right', bbox_to_anchor=(0.992, 0.93),
                   frameon=True, fancybox=False, edgecolor='black',
                   borderpad=0.35, handlelength=1.9, handletextpad=0.8,
                   labelspacing=0.35, borderaxespad=0.3)
legend.get_frame().set_linewidth(1.0)

for spine in ax.spines.values():
    spine.set_linewidth(1.3)
ax.tick_params(axis='both', width=1.3, length=6)

fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.13)

output_dir = Path(__file__).resolve().parent
fig.savefig(output_dir / 'Fig12_threshold_sensitivity_corrected.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'Fig12_threshold_sensitivity_corrected.pdf', bbox_inches='tight')
fig.savefig(output_dir / 'Fig12_threshold_sensitivity_corrected.tif', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Saved: Fig12_threshold_sensitivity_corrected.png/.pdf/.tif')
