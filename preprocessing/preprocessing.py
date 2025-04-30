import os
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load dataset
data_path = os.path.join(PROJECT_ROOT, "data", "campaign_data.csv")
df = pd.read_csv(data_path)
print(f"\nLoaded data from: {data_path}")
print(f"Original columns: {df.columns.tolist()}")

# Drop ID and Target column
drop_cols = ['campaign_id', 'click_through_rate']
X = df.drop(columns=drop_cols)
y = df['click_through_rate']

print(f"\n🎯 Columns dropped: {drop_cols}")
print(f"Features used for preprocessing: {X.columns.tolist()}")
print(f"Target variable: {y.name}")

# Fix common misclassified columns
for col in ['audience_age', 'audience_gender', 'product_type', 'location', 'channel']:
    X[col] = X[col].astype(str)

# Detect column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {numeric_features}")
print(f"Categorical columns: {categorical_features}")

# Pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Fit and transform
X_processed = preprocessor.fit_transform(X)

print("\nPreprocessor fitted successfully.")
print("Columns seen by preprocessor (feature_names_in_):")
print(list(preprocessor.feature_names_in_))

# Save fitted preprocessor
preprocessed_dir = os.path.join(PROJECT_ROOT, "preprocessed")
os.makedirs(preprocessed_dir, exist_ok=True)
preprocessor_path = os.path.join(preprocessed_dir, "preprocessor.pkl")
joblib.dump(preprocessor, preprocessor_path)
print(f"Preprocessor saved to: {preprocessor_path}")

# Save processed data
encoded_cat_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
processed_feature_names = numeric_features + list(encoded_cat_cols)
X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)
df_cleaned = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)

output_csv_path = os.path.join(preprocessed_dir, "campaign_data_preprocessed.csv")
df_cleaned.to_csv(output_csv_path, index=False)
print(f"Transformed data saved to: {output_csv_path}")
