
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pandas as pd
import numpy as np
import numpy.ma as ma

# Load the dataset
df = pd.read_csv("airforce_data.csv")

# Convert categorical values to numerical x for grid
mission_map = {'Surveillance': 0, 'Training': 1, 'Combat': 2, 'Logistics': 3}
df['x'] = df['Mission Type'].map(mission_map)
df['y'] = df['Cyber Risk Level']

# Bin setup
x_bins = np.linspace(-0.5, 3.5, 5)
y_bins = np.linspace(-0.5, 4.5, 6)
x_centers = (x_bins[:-1] + x_bins[1:]) / 2
y_centers = (y_bins[:-1] + y_bins[1:]) / 2

# Count red (breach) and blue (non-breach) points
heat_red, _, _ = np.histogram2d(df[df['Breach History'] == 1]['x'], df[df['Breach History'] == 1]['y'], bins=[x_bins, y_bins])
heat_blue, _, _ = np.histogram2d(df[df['Breach History'] == 0]['x'], df[df['Breach History'] == 0]['y'], bins=[x_bins, y_bins])
heat_total = heat_red + heat_blue

# Proportions and masking for color accuracy
with np.errstate(divide='ignore', invalid='ignore'):
    proportion = np.true_divide(heat_red, heat_total)
    proportion[heat_total == 0] = np.nan

masked_proportion = ma.masked_invalid(proportion)

# Jittered scatterplot coordinates
x_jitter = 0.1
y_jitter = 0.1
df['x_jittered'] = df['x'] + np.random.normal(0, x_jitter, size=len(df))
df['y_jittered'] = df['y'] + np.random.normal(0, y_jitter, size=len(df))

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
cmap = LinearSegmentedColormap.from_list('custom_bwr', ['blue', 'white', 'red'], N=256)
norm = Normalize(vmin=0, vmax=1)

# Draw heatmap
im = ax.imshow(masked_proportion.T, extent=extent, origin='lower',
               cmap=cmap, norm=norm, interpolation='none', alpha=0.8)

# Overlay ratio labels (moved further to upper-left)
for i, x in enumerate(x_centers):
    for j, y in enumerate(y_centers):
        r = int(heat_red[i, j])
        b = int(heat_blue[i, j])
        total = r + b
        if total > 0:
            ax.text(x - 0.45, y + 0.4, f"{b}/{r}", ha='left', va='top', fontsize=8, color='black', alpha=0.9)

# Overlay high-contrast scatter points
for label, color in zip([0, 1], ['blue', 'red']):
    subset = df[df['Breach History'] == label]
    ax.scatter(subset['x_jittered'], subset['y_jittered'],
               color=color, edgecolors='white', linewidth=0.5,
               marker='o', s=50, label='No Breach' if label == 0 else 'Breach', alpha=0.9)

# Format plot
ax.set_xticks(range(4))
ax.set_xticklabels(['Surveillance', 'Training', 'Combat', 'Logistics'])
ax.set_yticks(range(5))
ax.set_xlabel('Mission Type')
ax.set_ylabel('Cyber Risk Level')
ax.set_title('✅ Final Heatmap: Accurate Coloring + Label Placement + High-Contrast Dots')
ax.legend(title='Breach History')
plt.tight_layout()
plt.show()
