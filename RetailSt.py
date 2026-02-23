import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

rfm = pd.read_csv('rfm.csv')
anomalies_viz = pd.read_csv('anomalies_viz.csv')
rfm_plot_c = pd.read_csv('rfm_plot.csv')


st.title("RFM Dashboard")

# ---------- FIRST PLOT ----------
st.subheader("Data Distribution")

dist_option = st.selectbox(
    "Select transformation",
    ["Regular", "Log Transformed"],
    key="dist_select"
)

# Prepare data (only compute log when needed)
if dist_option == "Log Transformed":
    rfm_plot = rfm.copy()
    rfm_plot['Recency'] = np.log1p(rfm_plot['Recency'])
    rfm_plot['Frequency'] = np.log1p(rfm_plot['Frequency'])
    rfm_plot['Monetary'] = np.log1p(rfm_plot['Monetary'])
else:
    rfm_plot = rfm

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot
sns.histplot(rfm_plot['Recency'], bins=30, ax=axes[0])
axes[0].set_title(
    'Recency (Log Transformed)' if dist_option == "Log Transformed"
    else 'Recency Distribution'
)

sns.histplot(rfm_plot['Frequency'], bins=30, ax=axes[1])
axes[1].set_title(
    'Frequency (Log Transformed)' if dist_option == "Log Transformed"
    else 'Frequency Distribution'
)

sns.histplot(rfm_plot['Monetary'], bins=30, ax=axes[2])
axes[2].set_title(
    'Monetary (Log Transformed)' if dist_option == "Log Transformed"
    else 'Monetary Distribution'
)

plt.tight_layout()
st.pyplot(fig)


#--------------------------------------------------------------------------------------


# ---------- SECOND PLOT ----------
st.subheader("Customer Segmentation")

highlight_option = st.selectbox(
    "Highlight segment",
    ["All", "Low Engagement", "High Engagement", "Anomalies"],
    key="cluster_select"
)

fig2, ax = plt.subplots(figsize=(10, 6))

# --- Base plot (always shown lightly) ---
ax.scatter(
    rfm_plot_c['PC1'],
    rfm_plot_c['PC2'],
    c=rfm_plot_c['Cluster'],
    cmap='tab10',
    alpha=0.25 if highlight_option != "All" else 0.6,
    label='KMeans Clusters'
)

# --- Highlight logic ---
if highlight_option == "Low Engagement":
    low_mask = rfm_plot_c['Cluster'] == 0  # ⚠️ adjust if your label differs
    ax.scatter(
        rfm_plot_c.loc[low_mask, 'PC1'],
        rfm_plot_c.loc[low_mask, 'PC2'],
        color='blue',
        alpha=0.9,
        label='Low Engagement (Highlighted)'
    )

elif highlight_option == "High Engagement":
    high_mask = rfm_plot_c['Cluster'] == 1  # ⚠️ adjust if needed
    ax.scatter(
        rfm_plot_c.loc[high_mask, 'PC1'],
        rfm_plot_c.loc[high_mask, 'PC2'],
        color='orange',
        alpha=0.9,
        label='High Engagement (Highlighted)'
    )

# --- Anomalies (always visible but emphasized when selected) ---
ax.scatter(
    anomalies_viz['PC1'],
    anomalies_viz['PC2'],
    facecolors='none',
    edgecolors='red',
    s=150 if highlight_option == "Anomalies" else 120,
    linewidths=2 if highlight_option == "Anomalies" else 1.5,
    label='DBSCAN Anomalies'
)

ax.set_title('Customer Segmentation with Anomaly Overlay')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig2)