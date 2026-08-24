import pandas as pd
import numpy as np

import os
import sys

# Absolute paths so this script runs the same way regardless of cwd
# (matches preprocessing.py's / model_training.py's approach).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters.
# NOTE: audience_age is generated as the same buckets the Streamlit app's
# dropdown offers (not individual ages) -- the app never asks for a precise
# age, so training on precise ages would leave the model unable to match
# any category the app actually sends it.
n_rows = 10000
product_types = ['Shoes', 'Smartphones', 'Laptops', 'Furniture', 'Books']
audience_age_buckets = ['18-24', '25-34', '35-44', '45-54', '55+']
audience_genders = ['Male', 'Female', 'Other']
locations = ['New York', 'California', 'Texas', 'Florida', 'Illinois']
channels = ['Google Ads', 'Facebook', 'Instagram', 'LinkedIn', 'YouTube']
durations = [7, 14, 30, 45, 60]  # in days

product_type_arr = np.random.choice(product_types, n_rows)
audience_age_arr = np.random.choice(audience_age_buckets, n_rows)
audience_gender_arr = np.random.choice(audience_genders, n_rows)
location_arr = np.random.choice(locations, n_rows)
channel_arr = np.random.choice(channels, n_rows)
duration_arr = np.random.choice(durations, n_rows)
budget_arr = np.round(np.random.uniform(500, 10000, n_rows), 2)

df = pd.DataFrame({
    'campaign_id': range(1, n_rows + 1),
    'product_type': product_type_arr,
    'audience_age': audience_age_arr,
    'audience_gender': audience_gender_arr,
    'location': location_arr,
    'channel': channel_arr,
    'duration_days': duration_arr,
    'budget_usd': budget_arr,
})

# ---------------------------------------------------------------------------
# Generate click_through_rate from the campaign-planning attributes above,
# with effects loosely modeled on real digital-advertising patterns. This
# replaces the old fully-random target (which had no relationship to any
# input feature) with a signal-bearing one, so the trained model actually
# learns something rather than fitting noise.
#
# audience_gender is intentionally left with no effect on CTR -- it's kept
# as a feature (real campaign datasets carry weak/non-predictive fields
# too) but no CTR-driving relationship is invented for it.
# ---------------------------------------------------------------------------
BASE_CTR = 0.035

# Search intent converts better than social/display browsing.
channel_effect = {
    'Google Ads': 0.020,
    'YouTube': 0.006,
    'Facebook': 0.000,
    'Instagram': -0.003,
    'LinkedIn': -0.010,
}

# Product x age-bucket affinity: younger audiences respond better to
# shoes/electronics, older audiences respond better to furniture/books.
product_age_effect = {
    ('Shoes', '18-24'): 0.018, ('Shoes', '25-34'): 0.010, ('Shoes', '35-44'): 0.000,
    ('Shoes', '45-54'): -0.006, ('Shoes', '55+'): -0.010,
    ('Smartphones', '18-24'): 0.020, ('Smartphones', '25-34'): 0.014, ('Smartphones', '35-44'): 0.004,
    ('Smartphones', '45-54'): -0.004, ('Smartphones', '55+'): -0.010,
    ('Laptops', '18-24'): 0.006, ('Laptops', '25-34'): 0.012, ('Laptops', '35-44'): 0.010,
    ('Laptops', '45-54'): 0.002, ('Laptops', '55+'): -0.004,
    ('Furniture', '18-24'): -0.010, ('Furniture', '25-34'): 0.004, ('Furniture', '35-44'): 0.014,
    ('Furniture', '45-54'): 0.012, ('Furniture', '55+'): 0.006,
    ('Books', '18-24'): 0.002, ('Books', '25-34'): 0.006, ('Books', '35-44'): 0.008,
    ('Books', '45-54'): 0.010, ('Books', '55+'): 0.012,
}

# More ad-saturated / competitive metro markets convert slightly worse.
location_effect = {
    'California': -0.003,
    'New York': -0.003,
    'Texas': 0.002,
    'Florida': 0.002,
    'Illinois': 0.000,
}

product_age_arr = np.array([product_age_effect[(p, a)] for p, a in zip(product_type_arr, audience_age_arr)])
channel_eff_arr = np.array([channel_effect[c] for c in channel_arr])
location_eff_arr = np.array([location_effect[l] for l in location_arr])

# Diminishing-returns budget effect (bigger budgets buy better ad
# placement/targeting, but with saturation) and mild ad-fatigue decay as a
# campaign runs longer.
budget_effect = 0.014 * np.log1p(budget_arr / 1000.0) - 0.014 * np.log1p(500 / 1000.0)
duration_effect = -0.00025 * (duration_arr - durations[0])

noise = np.random.normal(0, 0.006, n_rows)

ctr = (BASE_CTR + channel_eff_arr + product_age_arr + location_eff_arr
       + budget_effect + duration_effect + noise)
df['click_through_rate'] = np.round(np.clip(ctr, 0.002, 0.30), 4)

data_path = os.path.join(PROJECT_ROOT, "data", "campaign_data.csv")
df.to_csv(data_path, index=False)
print(f"Generated {n_rows} rows. CTR range: "
      f"{df['click_through_rate'].min():.4f} - {df['click_through_rate'].max():.4f}, "
      f"mean: {df['click_through_rate'].mean():.4f}")
