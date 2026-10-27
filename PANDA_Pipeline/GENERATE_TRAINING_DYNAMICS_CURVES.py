from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Recreates the training-dynamics figure shown in the reference image.
# Outputs are saved next to this script for Windows/Linux compatibility.

# Epochs
epochs = np.arange(1, 17)

# Curves approximated from the reference figure
training_loss = np.array([
    1.06, 0.845, 0.802, 0.822, 0.783, 0.761, 0.739, 0.756,
    0.732, 0.712, 0.700, 0.712, 0.703, 0.702, 0.692, 0.681
])

validation_loss = np.array([
    0.942, 0.695, 0.667, 0.684, 0.658, 0.636, 0.651, 0.625,
    0.643, 0.631, 0.620, 0.629, 0.6229, 0.639, 0.651, 0.641
])

dice_val = np.array([
    0.800, 0.813, 0.810, 0.807, 0.820, 0.824, 0.816, 0.821,
    0.824, 0.830, 0.8332, 0.829, 0.827, 0.8285, 0.8245, 0.8288
])

best_dice_epoch = 11
best_dice_value = 0.8332
min_loss_epoch = 13
min_loss_value = 0.6229

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.titlesize': 28,
    'axes.labelsize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
})

fig, ax1 = plt.subplots(figsize=(14, 8), dpi=300)
ax2 = ax1.twinx()

# Main curves
l1, = ax1.plot(
    epochs, training_loss,
    color='#1148ff', marker='o', markersize=7.5, linewidth=1.9,
    label='Training Loss'
)

l2, = ax1.plot(
    epochs, validation_loss,
    color='#ff7a00', marker='s', markersize=7.0, linewidth=1.8,
    linestyle=(0, (6, 3)), label='Validation Loss'
)

l3, = ax2.plot(
    epochs, dice_val,
    color='#0a7f13', marker='^', markersize=8.0, linewidth=1.9,
    label='Dice Score (Validation)'
)

# Axes setup
ax1.set_xlim(0.5, 16.5)
ax1.set_xticks(epochs)
ax1.set_xlabel('Epoch', fontweight='bold')

ax1.set_ylim(0.4, 1.2)
ax1.set_yticks(np.arange(0.4, 1.21, 0.2))
ax1.set_ylabel('Loss', color='#1148ff', fontsize=20, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#1148ff')

ax2.set_ylim(0.78, 0.84)
ax2.set_yticks(np.arange(0.78, 0.841, 0.01))
ax2.set_ylabel('Dice Score', color='#0a7f13', fontsize=20, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#0a7f13')

# Horizontal subtle grid
for y in np.linspace(0.4, 1.2, 9):
    ax1.axhline(y, color='#bfbfbf', linestyle='--', linewidth=1.0, alpha=0.6, zorder=0)

# Vertical markers
ax1.axvline(best_dice_epoch, color='#ff8c00', linestyle='--', linewidth=1.5, alpha=0.95)
ax1.axvline(min_loss_epoch, color='#b000ff', linestyle='--', linewidth=1.5, alpha=0.95)

# Title
fig.suptitle('Training Dynamics: Loss Curves and Dice Score over Epochs',
             fontweight='bold', y=0.98)

# Legend on top center
fig.legend(
    handles=[l1, l2, l3],
    labels=['Training Loss', 'Validation Loss', 'Dice Score (Validation)'],
    loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=3,
    frameon=False, handlelength=2.4, columnspacing=2.0
)

# Top annotation boxes
ax1.text(
    0.67, 1.085, 'Best Model\nEpoch 11\nDice=0.8332',
    transform=ax1.transAxes, ha='center', va='center',
    fontsize=14, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.35', fc='#fff4ea', ec='#ff8c00', lw=1.4)
)

ax1.text(
    0.83, 1.085, 'Min Loss\nEpoch 13\nLoss=0.6229',
    transform=ax1.transAxes, ha='center', va='center',
    fontsize=14, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.35', fc='#f8ebff', ec='#b000ff', lw=1.4)
)

# Bottom summary box
summary = 'Best Validation Loss: 0.6229 at Epoch 13   |   Best Dice Score: 0.8332 at Epoch 11'
fig.text(
    0.5, 0.06, summary,
    ha='center', va='center', fontsize=14, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.55', fc='#f5f5f5', ec='black', lw=1.4)
)

# Spine styling
for sp in ax1.spines.values():
    sp.set_linewidth(1.2)
for sp in ax2.spines.values():
    sp.set_linewidth(1.2)
ax1.tick_params(axis='both', width=1.2, length=5)
ax2.tick_params(axis='y', width=1.2, length=5)

fig.subplots_adjust(left=0.08, right=0.92, bottom=0.17, top=0.78)

out_dir = Path(__file__).resolve().parent
png_path = out_dir / 'Fig_training_dynamics_curves.png'
pdf_path = out_dir / 'Fig_training_dynamics_curves.pdf'
tif_path = out_dir / 'Fig_training_dynamics_curves.tif'

fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(tif_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print('Saved:')
print(png_path)
print(pdf_path)
print(tif_path)
