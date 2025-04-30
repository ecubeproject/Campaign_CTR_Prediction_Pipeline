import pandas as pd
import numpy as np
import random

import os
import sys

# Add project root to sys.path for module import support if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
n_rows = 10000
product_types = ['Shoes', 'Smartphones', 'Laptops', 'Furniture', 'Books']
audience_ages = list(range(18, 65))
audience_genders = ['Male', 'Female', 'Other']
locations = ['New York', 'California', 'Texas', 'Florida', 'Illinois']
channels = ['Google Ads', 'Facebook', 'Instagram', 'LinkedIn', 'YouTube']
durations = [7, 14, 30, 45, 60]  # in days
budgets = np.round(np.random.uniform(500, 10000, n_rows), 2)

# Synthetic DataFrame
df = pd.DataFrame({
    'campaign_id': range(1, n_rows + 1),
    'product_type': np.random.choice(product_types, n_rows),
    'audience_age': np.random.choice(audience_ages, n_rows),
    'audience_gender': np.random.choice(audience_genders, n_rows),
    'location': np.random.choice(locations, n_rows),
    'channel': np.random.choice(channels, n_rows),
    'duration_days': np.random.choice(durations, n_rows),
    'budget_usd': budgets
})

# Generate performance metrics based on some randomized logic
df['impressions'] = np.random.randint(1000, 100000, size=n_rows)
df['click_through_rate'] = np.round(np.random.uniform(0.01, 0.15, size=n_rows), 4)
df['clicks'] = (df['impressions'] * df['click_through_rate']).astype(int)
df['conversion_rate'] = np.round(np.random.uniform(0.01, 0.10, size=n_rows), 4)
df['conversions'] = (df['clicks'] * df['conversion_rate']).astype(int)
df['cost_per_click'] = np.round(df['budget_usd'] / df['clicks'].replace(0, 1), 2)

df.to_csv("../data/campaign_data.csv", index=False)
