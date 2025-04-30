"""
EDA.py

Performs exploratory data analysis on campaign_data.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import sys

# Add project root to sys.path for module import support if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load dataset
df = pd.read_csv("../data/campaign_data.csv")


# Create output directory
os.makedirs("eda_outputs", exist_ok=True)

# Basic info
print("Shape of dataset:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values:\n", df.isnull().sum())
print("Data types:\n", df.dtypes)

# Summary stats
df.describe(include='all').to_csv("eda_outputs/summary_statistics.csv")

# CTR vs Conversion Rate by Product Type — subplots
product_types = df['product_type'].unique()
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

for ax, product in zip(axes, product_types):
    subset = df[df['product_type'] == product]
    sns.scatterplot(data=subset, x='click_through_rate', y='conversion_rate', ax=ax, alpha=0.6)
    ax.set_title(product)
    ax.set_xlabel('CTR')
    ax.set_ylabel('Conversion Rate')

plt.suptitle("CTR vs Conversion Rate by Product Type", fontsize=14)
plt.tight_layout()
plt.savefig("eda_outputs/ctr_vs_cr_by_product.png")
plt.clf()

# Budget vs Conversions by Channel — subplots
channels = df['channel'].unique()
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

for ax, channel in zip(axes, channels):
    subset = df[df['channel'] == channel]
    sns.scatterplot(data=subset, x='budget_usd', y='conversions', ax=ax, alpha=0.6)
    ax.set_title(channel)
    ax.set_xlabel('Budget')
    ax.set_ylabel('Conversions')

plt.suptitle("Budget vs Conversions by Channel", fontsize=14)
plt.tight_layout()
plt.savefig("eda_outputs/budget_vs_conversions_by_channel.png")
plt.clf()

# Half-triangle correlation heatmap — larger plot, smaller font
corr = df.select_dtypes(include='number').corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    annot_kws={"size": 8}
)
plt.title("Correlation Heatmap (Lower Triangle Only)", fontsize=16)
plt.savefig("eda_outputs/correlation_heatmap_half.png")
plt.clf()

print( "EDA complete. Plots and stats saved to 'eda_outputs/' folder.")
