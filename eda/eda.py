"""
EDA.py

Performs exploratory data analysis on campaign_data.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_outputs")

# Load dataset
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "campaign_data.csv"))

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Basic info
print("Shape of dataset:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values:\n", df.isnull().sum())
print("Data types:\n", df.dtypes)

# Summary stats
df.describe(include='all').to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))

# CTR distribution by product type
product_types = df['product_type'].unique()
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='product_type', y='click_through_rate')
plt.title("CTR Distribution by Product Type")
plt.xlabel("Product Type")
plt.ylabel("CTR")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ctr_by_product_type.png"))
plt.clf()

# Budget vs CTR by channel
channels = df['channel'].unique()
fig, axes = plt.subplots(1, len(channels), figsize=(20, 4), sharey=True)
for ax, channel in zip(axes, channels):
    subset = df[df['channel'] == channel]
    sns.scatterplot(data=subset, x='budget_usd', y='click_through_rate', ax=ax, alpha=0.6)
    ax.set_title(channel)
    ax.set_xlabel('Budget')
    ax.set_ylabel('CTR')
plt.suptitle("Budget vs CTR by Channel", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "budget_vs_ctr_by_channel.png"))
plt.clf()

# Half-triangle correlation heatmap
corr = df.select_dtypes(include='number').drop(columns=['campaign_id']).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    annot_kws={"size": 8}
)
plt.title("Correlation Heatmap (Lower Triangle Only)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_half.png"))
plt.clf()

print(f"EDA complete. Plots and stats saved to '{OUTPUT_DIR}'.")
