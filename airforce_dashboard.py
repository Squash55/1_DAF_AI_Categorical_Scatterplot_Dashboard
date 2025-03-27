
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

st.set_page_config(page_title="Categorical Heatmap with Proportions", layout="wide")
st.title("🛡️ Air Force Breach Proportion Dashboard")

uploaded_file = st.file_uploader("Upload a new dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("airforce_data.csv")

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
            ax.text(x, y, f"{b}/{r}", ha='center', va='center', fontsize=8, color='black', alpha=0.9)

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

st.subheader("📊 Rule-Based Insights")
grouped = df.groupby('Mission Type')
for name, group in grouped:
    rate = group['Breach History'].mean()
    high_risk = group[group['Cyber Risk Level'] >= 2]
    high_rate = high_risk['Breach History'].mean() if not high_risk.empty else 0
    st.markdown(f"🧠 **{name}** missions have a breach rate of **{rate:.0%}**, "
                f"and **{high_rate:.0%}** at higher risk levels.")

if openai.api_key:
    st.subheader("🤖 GPT-Based Insights")
    prompt = f"Generate executive insights about breach rates and mission types using this data schema: {list(df.columns)}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        for line in response['choices'][0]['message']['content'].split("\n"):
            if line.strip():
                st.markdown("🤖 " + line.strip())
    except Exception as e:
        st.error(f"GPT Error: {e}")
else:
    st.info("OpenAI key not detected. Add it in .env as OPENAI_API_KEY=sk-... to enable GPT insights.")
