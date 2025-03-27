import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Air Force Proportion Heatmap", layout="wide")

# Load dataset
uploaded_file = st.file_uploader("Upload a new Air Force dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("airforce_data.csv")

# Sidebar controls
st.sidebar.markdown("## 🎛️ Heatmap Settings")
x_jitter = st.sidebar.slider("Horizontal Jitter", 0, 30, 10, 1)
y_jitter = st.sidebar.slider("Vertical Jitter", 0, 30, 10, 1)
show_heatmap = st.sidebar.checkbox("Show Breach Proportion Heatmap", value=True)
min_count = st.sidebar.slider("Minimum Count per Cell", 1, 20, 5)

# Convert mission types to numeric
mission_map = {'Surveillance': 0, 'Training': 1, 'Combat': 2, 'Logistics': 3}
df['x_base'] = df['Mission Type'].map(mission_map)
df['x'] = df['x_base'] + np.random.normal(0, x_jitter / 100, size=len(df))
df['y'] = df['Cyber Risk Level'] + np.random.normal(0, y_jitter / 100, size=len(df))

# Prepare plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot heatmap if toggled
if show_heatmap:
    x_bins = np.linspace(-0.5, 3.5, 41)
    y_bins = np.linspace(-1, 4.5, 41)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2

    heat_red, _, _ = np.histogram2d(df[df['Breach History'] == 1]['x'], df[df['Breach History'] == 1]['y'], bins=[x_bins, y_bins])
    heat_all, _, _ = np.histogram2d(df['x'], df['y'], bins=[x_bins, y_bins])

    proportion = np.full_like(heat_all, 0.5, dtype=float)
    valid_mask = heat_all >= min_count
    proportion[valid_mask] = heat_red[valid_mask] / heat_all[valid_mask]

    # Mask out invalid (sparse) cells
    masked_prop = ma.masked_where(~valid_mask, proportion)

    cmap = LinearSegmentedColormap.from_list("custom_bwr", ["blue", "white", "red"], N=256)
    ax.imshow(masked_prop.T, extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
              origin='lower', cmap=cmap, alpha=0.8, aspect='auto')

    # Overlay counts
    for i, x in enumerate(x_centers):
        for j, y in enumerate(y_centers):
            count = int(heat_all[i, j])
            if count >= min_count:
                ax.text(x, y, str(count), ha='center', va='center', fontsize=7, color='black', alpha=0.8)

# Plot scatter points with jitter
for label, color in zip([0, 1], ['blue', 'red']):
    subset = df[df['Breach History'] == label]
    ax.scatter(subset['x'], subset['y'], color=color,
               label='No Breach' if label == 0 else 'Breach',
               edgecolor='k', alpha=0.8)

# Axes
ax.set_xticks(range(4))
ax.set_xticklabels(['Surveillance', 'Training', 'Combat', 'Logistics'])
ax.set_yticks(range(5))
ax.set_xlabel("Mission Type")
ax.set_ylabel("Cyber Risk Level")
ax.set_title("Proportion-Based Heatmap with Jittered Categorical Scatter")
ax.legend(title='Breach History')

# Show plot
st.pyplot(fig)
