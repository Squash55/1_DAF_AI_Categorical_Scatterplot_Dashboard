
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

uploaded_file = st.file_uploader("Upload a new dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    file_title = uploaded_file.name.replace("_", " ").replace(".csv", "").title()
    title_text = f"📊 {file_title} Dashboard"
else:
    df = pd.read_csv("airforce_data.csv")
    title_text = "🛡️ Air Force Breach Proportion Dashboard"

st.title(title_text)

st.markdown("""
### 📘 Methods & Limitations

This dashboard uses a combination of **data-driven calculations**, **rule-based logic**, and optional **GPT-based insights** to interpret breach risks across mission types and cyber risk levels.

- Grid cells are color-coded **only when data is available**, and colored based on the **proportion of red (breach) vs blue (no breach)**.
- Cells with **50/50 proportions or low data density** appear white to avoid over-interpretation.
- **Empty cells** (no observations) are also white by design, but may visually resemble 50/50 cells.
- Rule-based insights are derived from live data aggregations.
- GPT-based summaries are tagged and generated using OpenAI, with optional refresh.

**Missing variable advisory**: This analysis is based on limited fields. Important contextual variables (e.g., mission criticality, time in service, threat posture) are not included and may affect interpretation.
""")

col1, col2 = st.columns(2)
x_jitter = col1.slider("Horizontal Jitter", 0, 30, 10, 1)
y_jitter = col2.slider("Vertical Jitter", 0, 30, 10, 1)

mission_map = {'Surveillance': 0, 'Training': 1, 'Combat': 2, 'Logistics': 3}
df['x'] = df['Mission Type'].map(mission_map)
df['y'] = df['Cyber Risk Level']
df['x_jittered'] = df['x'] + np.random.normal(0, x_jitter / 100, size=len(df))
df['y_jittered'] = df['y'] + np.random.normal(0, y_jitter / 100, size=len(df))

x_bins = np.linspace(-0.5, 3.5, 5)
y_bins = np.linspace(-0.5, 4.5, 6)
x_centers = (x_bins[:-1] + x_bins[1:]) / 2
y_centers = (y_bins[:-1] + y_bins[1:]) / 2

breach_mask = df['Breach History'] == 1
non_breach_mask = df['Breach History'] == 0
heat_red, _, _ = np.histogram2d(df[breach_mask]['x'], df[breach_mask]['y'], bins=[x_bins, y_bins])
heat_blue, _, _ = np.histogram2d(df[non_breach_mask]['x'], df[non_breach_mask]['y'], bins=[x_bins, y_bins])
heat_total = heat_red + heat_blue

with np.errstate(divide='ignore', invalid='ignore'):
    proportion = np.true_divide(heat_red, heat_total)
    proportion[heat_total == 0] = np.nan

masked_proportion = ma.masked_invalid(proportion)

fig, ax = plt.subplots(figsize=(10, 6))
extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
custom_cmap = LinearSegmentedColormap.from_list('custom_bwr', ['blue', 'white', 'red'], N=256)
ax.imshow(masked_proportion.T, extent=extent, origin='lower', cmap=custom_cmap, alpha=0.7, aspect='auto')

for i, x in enumerate(x_centers):
    for j, y in enumerate(y_centers):
        r = int(heat_red[i, j])
        b = int(heat_blue[i, j])
        if r + b > 0:
            ax.text(x - 0.3, y + 0.3, f"{b}/{r}", ha='left', va='top', fontsize=8, color='black', alpha=0.9)

for label, color in zip([0, 1], ['blue', 'red']):
    subset = df[df['Breach History'] == label]
    ax.scatter(subset['x_jittered'], subset['y_jittered'], color=color,
               edgecolor='k', label='No Breach' if label == 0 else 'Breach', alpha=0.9)

ax.set_xticks(range(4))
ax.set_xticklabels(['Surveillance', 'Training', 'Combat', 'Logistics'])
ax.set_yticks(range(5))
ax.set_xlabel('Mission Type')
ax.set_ylabel('Cyber Risk Level')
ax.set_title('Categorical Heatmap with Red/Blue Proportions per Cell')
ax.legend(title='Breach History')
st.pyplot(fig)

# 📊 Pareto Chart Section
st.subheader("📊 Breach Rate Pareto Chart by Mission × Risk Level")
grouped = df.groupby(['Mission Type', 'Cyber Risk Level'])
summary = grouped['Breach History'].agg(['mean', 'count']).reset_index()
summary['Label'] = summary['Mission Type'] + ' @ ' + summary['Cyber Risk Level'].astype(str)
summary['Breach %'] = (summary['mean'] * 100).round(1)
summary = summary.sort_values(by='mean', ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 6))
bars = ax2.barh(summary['Label'], summary['Breach %'], color='tomato', edgecolor='black')
ax2.set_xlabel('Breach Percentage (%)')
ax2.set_title('Pareto Chart: Breach Rate by Mission × Risk Level')

for bar, count in zip(bars, summary['count']):
    width = bar.get_width()
    ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f"{count} pts", va='center', fontsize=8)

plt.gca().invert_yaxis()
st.pyplot(fig2)
