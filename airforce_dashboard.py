import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai
import os
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap

# Load OpenAI key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Air Force Risk Dashboard", layout="wide")

# Upload or use default data
uploaded_file = st.file_uploader("Upload a new Air Force dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("airforce_data.csv")

# Dashboard header
st.markdown("""
## 📘 Analysis Methods & Auto-Update Capabilities

This dashboard uses **rule-based logic** and **optional GPT-based AI** to generate insights.  
Interpretations auto-update whenever data changes. 

- 🧠 **Rule-Based Insights**: Aggregations by mission type + thresholds  
- 🤖 **GPT Insights**: Optional LLM-based summaries  
- 📉 **Heatmap**: Shows breach proportion only for cells with ≥5 points (white = balanced or low data)
""", unsafe_allow_html=True)

# Jitter sliders
col1, col2 = st.columns(2)
x_jitter = col1.slider("Horizontal Jitter", 0, 30, 10)
y_jitter = col2.slider("Vertical Jitter", 0, 30, 10)

# Map mission types to numbers
mission_map = {'Surveillance': 0, 'Training': 1, 'Combat': 2, 'Logistics': 3}
df['x'] = df['Mission Type'].map(mission_map) + np.random.normal(0, x_jitter / 100, size=len(df))
df['y'] = df['Cyber Risk Level'] + np.random.normal(0, y_jitter / 100, size=len(df))

# Heatmap toggle
show_heatmap = st.checkbox("🟦 Show Data-Driven Heatmap", value=True)

# Start plot
fig, ax = plt.subplots(figsize=(10, 6))

# Optional: Draw proportion-based heatmap (only for cells with ≥5)
if show_heatmap:
    x_bins = np.linspace(-0.5, 3.5, 41)
    y_bins = np.linspace(-1, 4.5, 41)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    heat_red, _, _ = np.histogram2d(df[df['Breach History'] == 1]['x'], df[df['Breach History'] == 1]['y'], bins=[x_bins, y_bins])
    heat_all, _, _ = np.histogram2d(df['x'], df['y'], bins=[x_bins, y_bins])
    with np.errstate(divide='ignore', invalid='ignore'):
        prop = np.where(heat_all >= 5, heat_red / heat_all, 0.5)

    # Custom blue-white-red colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_bwr', ['blue', 'white', 'red'], N=256)
    extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
    ax.imshow(prop.T, extent=extent, origin='lower', cmap=custom_cmap, alpha=0.25, aspect='auto')

    # Overlay counts
    for i, x in enumerate(x_centers):
        for j, y in enumerate(y_centers):
            count = int(heat_all[i, j])
            if count > 0:
                ax.text(x, y, str(count), ha='center', va='center', fontsize=7, color='black', alpha=0.7)

# Plot the actual scatter points
for label, color in zip([0, 1], ['blue', 'red']):
    subset = df[df['Breach History'] == label]
    ax.scatter(subset['x'], subset['y'], color=color,
               label='No Breach' if label == 0 else 'Breach', edgecolor='k', alpha=0.8)

# Axis formatting
ax.set_xticks(range(4))
ax.set_xticklabels(['Surveillance', 'Training', 'Combat', 'Logistics'], rotation=15)
ax.set_yticks(range(5))
ax.set_ylabel('Cyber Risk Level')
ax.set_xlabel('Mission Type')
ax.legend(title='Breach History')
st.pyplot(fig)

# Rule-based insights
def generate_rule_based_insights(df):
    grouped = df.groupby('Mission Type')
    results = []
    for mission, group in grouped:
        breach_rate = group['Breach History'].mean()
        high_risk = group[group['Cyber Risk Level'] >= 2]
        high_risk_rate = high_risk['Breach History'].mean() if not high_risk.empty else 0
        results.append(
            f"🧠 [Rule-Based] **{mission}** missions show **{breach_rate:.0%} overall breach rate**, "
            f"with **{high_risk_rate:.0%} at Cyber Risk Level 2–3.**"
        )
    return results

# GPT-based insights
def generate_gpt_insights(df):
    if openai.api_key is None:
        return ["⚠️ GPT API key not found."]
    prompt = f"You are reviewing this Air Force mission dataset: {list(df.columns)}. Generate executive insights about breach trends by mission type and cyber risk."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return ["🤖 [GPT] " + line for line in response['choices'][0]['message']['content'].split('\n') if line.strip()]
    except Exception as e:
        return [f"⚠️ GPT error: {e}"]

# Output both interpretation types
st.subheader("📊 Auto-Generated Interpretations")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🧠 Rule-Based")
    for insight in generate_rule_based_insights(df):
        st.markdown(insight)
with col2:
    st.markdown("### 🤖 GPT-Based")
    for insight in generate_gpt_insights(df):
        st.markdown(insight)

# Missing data advisory
if openai.api_key:
    disclaimer_prompt = f"This dataset has columns: {list(df.columns)}. What additional fields should be included in future versions for better breach analysis? Tag each as High, Medium, or Optional."
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": disclaimer_prompt}],
            temperature=0.3
        )
        st.subheader("📉 Missing Data Disclaimer (AI-Generated)")
        for line in resp['choices'][0]['message']['content'].split('\n'):
            st.markdown("🛠️ " + line)
    except Exception as e:
        st.error(f"GPT error: {e}")
