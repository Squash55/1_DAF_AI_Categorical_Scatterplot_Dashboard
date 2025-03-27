import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma

# Load the dataset
df = pd.read_csv("airforce_data.csv")  # Replace with your actual CSV if different

# Convert mission types to x-coordinates
mission_map = {'Surveillance': 0, 'Training': 1, 'Combat': 2, 'Logistics': 3}
df['x'] = df['Mission Type'].map(mission_map)
df['y'] = df['Cyber Risk Level']

# Define categorical grid (bins match category labels)
x_bins = np.linspace(-0.5, 3.5, 5)  # Four categories: 0 to 3
y_bins = np.linspace(-0.5, 4.5, 6)  # Cyber risk levels: 0 to 4

# Compute breach counts and total counts per grid cell
heat_red, _, _ = np.histogram2d(df[df['Breach History'] == 1]['x'], df[df['Breach History'] == 1]['y'], bins=[x_bins, y_bins])
heat_all, _, _ = np.histogram2d(df['x'], df['y'], bins=[x_bins, y_bins])

# Calculate breach proportion per grid cell, mask empty cells
with np.errstate(divide='ignore', invalid='ignore'):
    proportion = np.true_divide(heat_red, heat_all)
    proportion[heat_all == 0] = np.nan  # White for empty cells

masked_proportion = ma.masked_invalid(proportion)

# Create red-white-blue colormap (0=blue, 0.5=white, 1=red)
custom_cmap = LinearSegmentedColormap.from_list('custom_bwr', ['blue', 'white', 'red'], N=256)

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 6))
extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
ax.imshow(masked_proportion.T, extent=extent, origin='lower',
          cmap=custom_cmap, alpha=0.7, aspect='auto')

# Jitter and overlay scatter points
x_jitter = 0.1
y_jitter = 0.1
df['x_jittered'] = df['x'] + np.random.normal(0, x_jitter, size=len(df))
df['y_jittered'] = df['y'] + np.random.normal(0, y_jitter, size=len(df))

for label, color in zip([0, 1], ['blue', 'red']):
    subset = df[df['Breach History'] == label]
    ax.scatter(subset['x_jittered'], subset['y_jittered'],
               color=color, edgecolor='k', alpha=0.9,
               label='No Breach' if label == 0 else 'Breach')

# Formatting
ax.set_xticks(range(4))
ax.set_xticklabels(['Surveillance', 'Training', 'Combat', 'Logistics'])
ax.set_yticks(range(5))
ax.set_xlabel('Mission Type')
ax.set_ylabel('Cyber Risk Level')
ax.set_title('Categorical Proportion Heatmap (Red = 100% Breach)')
ax.legend(title='Breach History')

# Show plot
plt.tight_layout()
plt.show()
