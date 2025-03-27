
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy.ma as ma
import os

# 📘 Intro Section — Methods & Limitations

st.title("Air Force Cyber Breach Analysis Dashboard")

st.markdown("### 📘 Methods & Limitations")

st.markdown("""
This dashboard visualizes breach proportions across mission types and cyber risk levels using:
- A categorical heatmap with color-coded red/blue breach proportions
- Jittered scatter points with high-contrast outlines
- Cell-level red/blue ratio counts
- A Pareto chart sorted by breach percentage
- Rule-based insight summaries derived from grouped data

Missing contextual variables may limit interpretation. Future versions may include GPT or statistical flags.
""")

st.title("Air Force Breach Analysis Dashboard")

# Load data safely
csv_path = "airforce_data_clean.csv"
if not os.path.exists(csv_path):
    st.error("🚨 Dataset not found! Please upload or ensure 'airforce_data_clean.csv' is in the same folder.")
    st.stop()

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

mission_map = {'Surveillance': 0, 'Training': 1, 'Combat': 2, 'Logistics': 3}
df['x'] = df['Mission Type'].map(mission_map)
df['y'] = df['Cyber Risk Level']

x_bins = np.linspace(-0.5, 3.5, 5)
y_bins = np.linspace(-0.5, 4.5, 6)
x_centers = (x_bins[:-1] + x_bins[1:]) / 2
y_centers = (y_bins[:-1] + y_bins[1:]) / 2

heat_red, _, _ = np.histogram2d(df[df['Breach History'] == 1]['x'], df[df['Breach History'] == 1]['y'], bins=[x_bins, y_bins])
heat_blue, _, _ = np.histogram2d(df[df['Breach History'] == 0]['x'], df[df['Breach History'] == 0]['y'], bins=[x_bins, y_bins])
heat_total = heat_red + heat_blue

with np.errstate(divide='ignore', invalid='ignore'):
    proportion = np.true_divide(heat_red, heat_total)
    proportion[heat_total == 0] = np.nan
masked_proportion = ma.masked_invalid(proportion)

x_jitter = st.slider("Horizontal Jitter", 0, 30, 10) / 100
y_jitter = st.slider("Vertical Jitter", 0, 30, 10) / 100
df['x_jittered'] = df['x'] + np.random.normal(0, x_jitter, size=len(df))
df['y_jittered'] = df['y'] + np.random.normal(0, y_jitter, size=len(df))

fig, ax = plt.subplots(figsize=(10, 6))
extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
cmap = LinearSegmentedColormap.from_list('custom_bwr', ['blue', 'white', 'red'], N=256)
norm = Normalize(vmin=0, vmax=1)
im = ax.imshow(masked_proportion.T, extent=extent, origin='lower',
               cmap=cmap, norm=norm, interpolation='none', alpha=0.8)

for i, x in enumerate(x_centers):
    for j, y in enumerate(y_centers):
        r = int(heat_red[i, j])
        b = int(heat_blue[i, j])
        total = r + b
        if total > 0:
            ax.text(x - 0.45, y + 0.4, f"{b}/{r}", ha='left', va='top', fontsize=8, color='black', alpha=0.9)

for label, color in zip([0, 1], ['blue', 'red']):
    subset = df[df['Breach History'] == label]
    ax.scatter(subset['x_jittered'], subset['y_jittered'],
               color=color, edgecolors='white', linewidth=0.5,
               marker='o', s=50, label='No Breach' if label == 0 else 'Breach', alpha=0.9)

ax.set_xticks(range(4))
ax.set_xticklabels(['Surveillance', 'Training', 'Combat', 'Logistics'])
ax.set_yticks(range(5))
ax.set_xlabel('Mission Type')
ax.set_ylabel('Cyber Risk Level')
ax.set_title('Categorical Breach Proportion Heatmap')
ax.legend(title='Breach History', loc='upper left', bbox_to_anchor=(1.02, 1))
st.pyplot(fig)

# 📊 Group and summarize for Pareto
grouped = df.groupby(['Mission Type', 'Cyber Risk Level'])
summary = grouped['Breach History'].agg(['mean', 'count']).reset_index()
summary['Label'] = summary['Mission Type'] + ' @ ' + summary['Cyber Risk Level'].astype(str)
summary['Breach %'] = (summary['mean'] * 100).round(1)
summary = summary.sort_values(by='Breach %', ascending=False)

# Pareto Chart
st.subheader("📊 Breach Rate Pareto Chart by Mission × Risk Level")
fig2, ax2 = plt.subplots(figsize=(10, 6))
bars = ax2.barh(summary['Label'], summary['Breach %'], color='tomato', edgecolor='black')
ax2.set_xlabel('Breach Percentage (%)')
ax2.set_title('Pareto Chart: Breach Rate by Mission × Risk Level')
for bar, count in zip(bars, summary['count']):
    width = bar.get_width()
    ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f"{count} pts", va='center', fontsize=8)
ax2.invert_yaxis()
st.pyplot(fig2)

# Rule-Based Insights
st.markdown("### 🧠 Rule-Based Insights")
high_risk = summary.loc[summary['mean'] > 0.5]
low_risk = summary.loc[summary['mean'] <= 0.5]
if not high_risk.empty:
    st.markdown("#### 🔴 High-Risk Areas")
    for _, row in high_risk.iterrows():
        st.write(f"• `{row['Label']}` shows a high breach rate of **{row['Breach %']}%**.")
if not low_risk.empty:
    st.markdown("#### 🔵 Lower-Risk Areas")
    for _, row in low_risk.iterrows():
        st.write(f"• `{row['Label']}` has a relatively low breach rate of **{row['Breach %']}%**.")


# 🔬 Fisher's Exact Test Flags (Statistical Significance)
st.markdown("### 🔬 Statistical Test Flags")
alpha = 0.05
significant_cells = []

for i, x in enumerate(x_centers):
    for j, y in enumerate(y_centers):
        red = int(heat_red[i, j])
        blue = int(heat_blue[i, j])
        other_red = heat_red.sum() - red
        other_blue = heat_blue.sum() - blue

        if (red + blue > 0) and (other_red + other_blue > 0):
            table = [[red, blue], [other_red, other_blue]]
            _, p = stats.fisher_exact(table)
            if p < alpha:
                significant_cells.append((i, j))
                ax.text(x, y, "🔺", ha='center', va='center', fontsize=12, color='black')

if significant_cells:
    st.success(f"{len(significant_cells)} quadrant(s) show statistically significant differences (p < 0.05).")
else:
    st.info("No quadrant shows a statistically significant breach difference (p < 0.05).")

# 🧠 Golden Questions
st.markdown("### ❓ Golden Questions")
golden_questions = []
top = summary.sort_values(by="Breach %", ascending=False).head(3)
for _, row in top.iterrows():
    if row["Breach %"] > 50:
        golden_questions.append(f"Why is `{row['Label']}` showing a breach rate of {row['Breach %']}%?")
if golden_questions:
    for q in golden_questions:
        st.write("•", q)
else:
    st.write("No high-breach patterns detected that warrant immediate questioning.")

# ✅ Golden Answers
st.markdown("### ✅ Golden Answers")
for _, row in top.iterrows():
    if row["Breach %"] > 70 and row["count"] >= 5:
        st.write(f"• `{row['Label']}` is the most breach-prone cell, with a statistically meaningful breach rate of **{row['Breach %']}%** across {row['count']} records.")
