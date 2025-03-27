
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy.ma as ma
import scipy.stats as stats

st.title("Air Force Cyber Breach Analysis Dashboard")

st.markdown("""### 📘 Methods & Limitations

This dashboard uses synthetic data to simulate cyber breach patterns across mission types and cyber risk levels.

- Heatmap shows red/blue proportions of breach outcomes
- Fisher’s Exact Test is used to flag statistically significant breach differences (p < 0.05)
- Golden Questions are generated based on the most breach-prone quadrants
- Golden Answers are only offered when the data clearly supports them
- This version does not require file upload and uses internal test data
""")

# Synthetic data
np.random.seed(42)
df = pd.DataFrame({
    'Mission Type': np.random.choice(['Surveillance', 'Training', 'Combat', 'Logistics'], size=200),
    'Cyber Risk Level': np.random.randint(0, 5, size=200),
    'Breach History': np.random.choice([0, 1], size=200, p=[0.7, 0.3])
})

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

x_jitter = 0.1
y_jitter = 0.1
df['x_jittered'] = df['x'] + np.random.normal(0, x_jitter, size=len(df))
df['y_jittered'] = df['y'] + np.random.normal(0, y_jitter, size=len(df))

fig, ax = plt.subplots(figsize=(10, 6))
extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
cmap = LinearSegmentedColormap.from_list('custom_bwr', ['blue', 'white', 'red'], N=256)
norm = Normalize(vmin=0, vmax=1)
ax.imshow(masked_proportion.T, extent=extent, origin='lower', cmap=cmap, norm=norm, interpolation='none', alpha=0.8)

mission_types = ['Surveillance', 'Training', 'Combat', 'Logistics']
coord_to_label = {(i, j): f"{mission_types[i]} @ {j}" for i in range(4) for j in range(5)}
significant_labels = []

for i, x in enumerate(x_centers):
    for j, y in enumerate(y_centers):
        r = int(heat_red[i, j])
        b = int(heat_blue[i, j])
        total = r + b
        if total > 0:
            ax.text(x - 0.45, y + 0.4, f"{b}/{r}", ha='left', va='top', fontsize=8, color='black', alpha=0.9)
        other_r = heat_red.sum() - r
        other_b = heat_blue.sum() - b
        if total > 0 and (other_r + other_b > 0):
            _, p = stats.fisher_exact([[r, b], [other_r, other_b]])
            if p < 0.05:
                ax.text(x, y, "🔺", ha='center', va='center', fontsize=14, color='black')
                significant_labels.append(coord_to_label.get((i, j), f"@({i},{j})"))

for label, color in zip([0, 1], ['blue', 'red']):
    subset = df[df['Breach History'] == label]
    ax.scatter(subset['x_jittered'], subset['y_jittered'], color=color, edgecolors='white', linewidth=0.5, s=50, alpha=0.9)

ax.set_xticks(range(4))
ax.set_xticklabels(mission_types)
ax.set_yticks(range(5))
ax.set_xlabel('Mission Type')
ax.set_ylabel('Cyber Risk Level')
ax.set_title('Categorical Heatmap of Breach Proportions')
ax.legend(['No Breach', 'Breach'], loc='upper left', bbox_to_anchor=(1.02, 1))
st.pyplot(fig)

# Stat output
st.markdown("### 🔬 Statistically Significant Quadrants")
if significant_labels:
    st.success(f"{len(significant_labels)} quadrant(s) show significant breach differences (p < 0.05).")
    for label in significant_labels:
        st.write("🔺", label)
else:
    st.info("No quadrant showed a statistically significant difference.")

# Pareto chart
st.subheader("📊 Breach Rate Pareto Chart")
grouped = df.groupby(['Mission Type', 'Cyber Risk Level'])
summary = grouped['Breach History'].agg(['mean', 'count']).reset_index()
summary['Label'] = summary['Mission Type'] + ' @ ' + summary['Cyber Risk Level'].astype(str)
summary['Breach %'] = (summary['mean'] * 100).round(1)
summary = summary.sort_values(by='Breach %', ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 6))
bars = ax2.barh(summary['Label'], summary['Breach %'], color='tomato', edgecolor='black')
for bar, count in zip(bars, summary['count']):
    width = bar.get_width()
    ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f"{count} pts", va='center', fontsize=8)
ax2.set_xlabel('Breach Percentage (%)')
ax2.set_title('Pareto Chart: Breach Rate by Mission × Risk Level')
ax2.invert_yaxis()
st.pyplot(fig2)

# Golden Questions
st.markdown("### ❓ Golden Questions")
top = summary.head(3)
questions = []
for _, row in top.iterrows():
    if row["Breach %"] > 50:
        questions.append(f"Why is `{row['Label']}` showing a breach rate of {row['Breach %']}%?")
if questions:
    for q in questions:
        st.write("•", q)
else:
    st.write("No high-breach patterns detected.")

# Golden Answers
st.markdown("### ✅ Golden Answers (Evidence-Based)")
answers = []
for _, row in top.iterrows():
    if row["Breach %"] > 70 and row["count"] >= 5:
        answers.append(f"• `{row['Label']}` shows the most breach-prone condition, with a rate of **{row['Breach %']}%** across {row['count']} records.")
if answers:
    for a in answers:
        st.markdown(a)
else:
    st.info("Golden Answers are only provided when the data clearly supports them.")
